# -*- coding:utf-8 -*-
"""
模型训练和测试运行文件
包含K折交叉验证训练流程和集成模型测试流程
"""

import warnings
# 抑制RDKit的弃用警告
warnings.filterwarnings('ignore', message='.*GetValence.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
# 抑制RDKit的logging警告
logging.getLogger('rdkit').setLevel(logging.ERROR)

# 抑制RDKit的警告输出（必须在导入rdkit之前）
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
import random
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import MIFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
# from utils.dataset import reset_placeholder_stats, get_placeholder_stats, print_placeholder_stats_summary
from utils.protein_3d_featurizer import protein_3d_init
import torch_geometric.loader as pyg_loader

# 默认计算设备（已废弃，实际使用由main.py传入的device参数）
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.backends.cuda
# 禁用 Flash Attention 和高效 SDPA backend，强制使用数学模式
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# 禁用 cuBLAS TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# 禁用 inductor 自动 JIT 编译
torch._inductor.config.triton.cudagraphs = False

def run_MIF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device):
    """
    运行MIF-DTI模型的K折交叉验证训练和测试流程
    
    Args:
        SEED: 随机种子
        DATASET: 数据集名称
        MODEL: 模型类
        K_Fold: K折交叉验证的折数
        LOSS: 损失函数类型
        device: 计算设备
    """
    # 设置随机种子以确保可复现性
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 初始化超参数
    hp = hyperparameter()
    
    # 自动检测子口袋预处理文件（如果未指定且文件存在）
    if hp.use_subpocket and hp.subpocket_cache_path is None:
        default_cache_path = f'./DataSets/Preprocessed/{DATASET}-subpocket.pt'
        if os.path.exists(default_cache_path):
            print(f"检测到子口袋预处理文件，自动使用: {default_cache_path}")
            hp.subpocket_cache_path = default_cache_path

    # 从文本文件加载数据集
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    # 设置损失函数权重（用于处理类别不平衡）
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    # 数据集划分：BD2D数据集有固定的划分方式
    if DATASET == "BD2D":
        split_pos = 52010
        train_data_list = data_list[:split_pos]
        test_data_list = data_list[split_pos:]
        # 打乱训练数据
        print("data shuffle")
        train_data_list = shuffle_dataset(train_data_list, SEED)
    else:
        # 打乱数据
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)

        # 划分训练验证集和测试集（80%训练验证，20%测试）
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    # 数据准备：加载或生成蛋白质和配体（药物）的图表示
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        # 如果预处理文件存在，直接加载
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        # 否则从序列生成蛋白质图
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        # 如果预处理文件存在，直接加载
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        # 否则从SMILES生成配体图
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    # 加载或生成蛋白质3D特征（如果启用）
    prot_3d_dict = None
    if hp.use_3d and hp.pdb_dir is not None:
        prot_3d_path = f'./DataSets/Preprocessed/{DATASET}-protein-3d.pkl'
        if os.path.exists(prot_3d_path):
            print('Loading Protein 3D Structure data...')
            prot_3d_dict = joblib.load(prot_3d_path)
        else:
            print('Initialising Protein 3D Structure from PDB files...')
            protein_names = list(set([item.split(' ')[-2] for item in data_list]))
            prot_3d_dict = protein_3d_init(protein_names, hp.pdb_dir, use_esm=True)
            joblib.dump(prot_3d_dict, prot_3d_path)
    elif hp.use_3d and hp.prot_3d_cache_path is not None:
        print('Loading Protein 3D Structure from cache...')
        prot_3d_dict = joblib.load(hp.prot_3d_cache_path)

    # 清空GPU缓存
    torch.cuda.empty_cache()

    # 初始化指标列表（用于存储K折交叉验证的结果）
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # K折交叉验证循环
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        # 获取当前折的训练集和验证集
        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        # # 重置占位符统计（每个fold开始时重置）
        # reset_placeholder_stats()
        # 创建数据集对象（将文本数据转换为图数据）
        train_dataset = ProteinMoleculeDataset(
            train_dataset, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path
        )
        valid_dataset = ProteinMoleculeDataset(
            valid_dataset, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path
        )
        test_dataset = ProteinMoleculeDataset(
            test_data_list, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path
        )
        # # 输出占位符统计汇总
        # print_placeholder_stats_summary()
        train_size = len(train_dataset)

        # 创建数据加载器
        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
                                    
        # 创建模型
        model = MODEL(device=device, use_subpocket=hp.use_subpocket, pocket_num=hp.pocket_num)

        # 初始化模型权重
        weight_p, bias_p = [], []
        # 对权重矩阵使用Xavier初始化
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 分离权重和偏置参数（用于不同的权重衰减）
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        # 创建优化器和学习率调度器
        # 权重参数使用权重衰减，偏置参数不使用权重衰减
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        # 循环学习率调度器（CyclicLR）
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        # 创建损失函数
        # if LOSS == 'PolyLoss':
        #     Loss = PolyLoss(weight_loss=weight_loss,
        #                     DEVICE=device, epsilon=hp.loss_epsilon)
        # else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

        # 设置输出文件路径
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        # 创建早停机制
        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        # 开始训练
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            # 如果早停条件满足，退出训练
            if early_stopping.early_stop == True:
                break

            # 每个epoch开始时重置统计
            # reset_placeholder_stats()

            # 训练阶段
            train_losses_in_epoch = []
            model.train()
            for data in train_loader:
                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                data = data.to(device)
                predicted_y= model(data)
                # NaN 检测
                if torch.isnan(predicted_y).any():
                    print(f"[NaN WARNING] Epoch {epoch}, batch produced NaN output. Max: {predicted_y.max()}, Min: {predicted_y.min()}")
                    print(f"  drug_repr shapes at block outputs (sample): check model internals")
                    torch.save(predicted_y, f"/tmp/nan_output_epoch{epoch}.pt")
                # 计算损失
                train_loss = Loss(predicted_y, data.cls_y)
                train_losses_in_epoch.append(train_loss.item())
                # 反向传播
                train_loss.backward()
                # 梯度裁剪，防止数值爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 更新参数
                optimizer.step()

                # 更新学习率
                scheduler.step()
            # 计算一个epoch的平均训练损失
            train_loss_a_epoch = np.average(train_losses_in_epoch)

            # 验证阶段
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []  # 真实标签、预测标签、预测分数
            with torch.no_grad():
                for data in valid_loader:
                    # 前向传播（不计算梯度）
                    data = data.to(device)
                    valid_scores = model(data)
                    
                    valid_labels = data.cls_y
                    # 计算验证损失
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    # 转换为numpy数组用于计算指标
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]  # 正类概率

                    # 收集所有批次的预测结果
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            # 计算验证集指标
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            # # 获取当前epoch的统计信息
            # stats = get_placeholder_stats()
            
            # 打印训练进度和验证指标
            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)
            
            # 打印数据使用统计
            # if stats['total_used'] > 0 or stats['total_created'] > 0:
            #     print(f'  [Data Stats] prot_3d: {stats["prot_3d_used"]} used, {stats["prot_3d_data"]} placeholders | '
            #           f'subpocket: {stats["subpocket_used"]} used, {stats["subpocket_data"]} placeholders')

            # 保存检查点并根据早停机制决定是否继续训练
            early_stopping(AUC_dev, model, epoch)

        # 加载最佳检查点
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{device}.pth', weights_only=True))

        # 在训练集、验证集和测试集上测试模型
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, MIF=True)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, MIF=True)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, MIF=True)
        # 保存当前折的测试结果
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        # 将结果写入文件
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    # 显示K折交叉验证的平均结果
    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)
    

def ensemble_run_MIF_model(SEED, DATASET, K_Fold, device):
    """
    运行集成MIF-DTI模型测试流程
    加载K折训练好的模型，在测试集上进行集成预测
    
    Args:
        SEED: 随机种子
        DATASET: 数据集名称
        K_Fold: K折数量（需要加载K个模型）
        device: 计算设备
    """
    # 设置随机种子
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 初始化超参数
    hp = hyperparameter()
    
    # 自动检测子口袋预处理文件（如果未指定且文件存在）
    if hp.use_subpocket and hp.subpocket_cache_path is None:
        default_cache_path = f'./DataSets/Preprocessed/{DATASET}-subpocket.pt'
        if os.path.exists(default_cache_path):
            print(f"检测到子口袋预处理文件，自动使用: {default_cache_path}")
            hp.subpocket_cache_path = default_cache_path

    # 加载数据集
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    # 设置损失函数权重
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    # 打乱数据
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    # 划分测试集（使用与训练时相同的划分方式）
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 数据准备：加载或生成蛋白质和配体图
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()  
    
          
    

    # 加载或生成蛋白质3D特征（如果启用）
    prot_3d_dict = None
    if hp.use_3d and hp.pdb_dir is not None:
        prot_3d_path = f'./DataSets/Preprocessed/{DATASET}-protein-3d.pkl'
        if os.path.exists(prot_3d_path):
            print('Loading Protein 3D Structure data...')
            prot_3d_dict = joblib.load(prot_3d_path)
        else:
            print('Initialising Protein 3D Structure from PDB files...')
            protein_names = list(set([item.split(' ')[-2] for item in data_list]))
            prot_3d_dict = protein_3d_init(protein_names, hp.pdb_dir, use_esm=True)
            joblib.dump(prot_3d_dict, prot_3d_path)
    elif hp.use_3d and hp.prot_3d_cache_path is not None:
        print('Loading Protein 3D Structure from cache...')
        prot_3d_dict = joblib.load(hp.prot_3d_cache_path)
    
    # 重置占位符统计
    # reset_placeholder_stats()
    test_dataset = ProteinMoleculeDataset(
        test_data_list, ligand_dict, protein_dict, device=device,
        use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
        use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
        subpocket_cache_path=hp.subpocket_cache_path
    )
    # 输出占位符统计汇总
    # print_placeholder_stats_summary()
    
    # test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                                  collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    # 加载K折训练好的模型
    model = []
    for i in range(K_Fold):
        model.append(MIFDTI(use_3d=hp.use_3d, use_subpocket=hp.use_subpocket, pocket_num=hp.pocket_num).to(device))
        # 加载对应折的模型权重（需要先运行K折训练）
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device)))
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MIFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    # 创建损失函数（用于测试，实际不用于反向传播）
    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=device, epsilon=hp.loss_epsilon)

    # 使用集成模型在测试集上进行预测
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, MIF=True)
    
    # 显示集成模型的测试结果
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
    
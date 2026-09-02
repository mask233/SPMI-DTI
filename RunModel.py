# -*- coding:utf-8 -*-
"""
模型训练和测试运行文件
包含K折交叉验证训练流程和集成模型测试流程
"""

import os
CUDA_DEBUG = os.environ.get('DTI_CUDA_DEBUG', '0') == '1'
STRICT_DETERMINISM = os.environ.get('DTI_STRICT_DETERMINISM', '0') == '1'
if CUDA_DEBUG:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
import sys
import random
import joblib
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from utils.TestModel import prediction_metrics, save_prediction_records, test_model
from utils.ShowResult import show_result
from utils.split_manifest import file_sha256, save_split_manifest
from utils import protein_init, ligand_init, ProteinMoleculeDataset
from utils.dataset import load_prepared_subpocket_cache
# from utils.dataset import reset_placeholder_stats, get_placeholder_stats, print_placeholder_stats_summary
from utils.protein_3d_featurizer import protein_3d_init
import torch_geometric.loader as pyg_loader
import torch_geometric  # 用于获取版本号

# 默认计算设备（已废弃，实际使用由main.py传入的device参数）
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.backends.cuda
# The slower math-only path remains available for strict reproducibility
# checks, but is not the default used for runtime/memory benchmarking.
if STRICT_DETERMINISM:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, '_inductor'):
        torch._inductor.config.triton.cudagraphs = False


def enable_cuda_stability_mode():
    """Prefer synchronous, conservative CUDA kernels for crash diagnosis."""
    global CUDA_DEBUG, STRICT_DETERMINISM
    CUDA_DEBUG = True
    STRICT_DETERMINISM = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, '_inductor'):
        torch._inductor.config.triton.cudagraphs = False
    print("[CUDA STABILITY] synchronous execution and conservative SDP kernels enabled", flush=True)


def set_experiment_seed(seed):
    """Set all RNGs used here; CUDA kernels may still be version-dependent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These settings improve repeatability without claiming bitwise identity
    # across different PyTorch/CUDA/GPU stacks.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _mean_detached_losses(losses):
    """Match the historical NumPy float64 mean with one device transfer."""
    if not losses:
        return float("nan")
    values = torch.stack(losses).detach().cpu().numpy().astype(np.float64, copy=False)
    return float(np.average(values))


def _assert_finite_scalar(value, message):
    """Fail on NaN/Inf with one scalar check instead of repeated diagnostics."""
    condition = torch.isfinite(value).all()
    if not condition.item():
        raise FloatingPointError(message)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _save_latest_training_state(path, model, optimizer, scheduler, epoch,
                                early_stopping, metadata):
    """Atomically save a resumable epoch boundary without replacing best weights."""
    state = {
        "schema_version": 1,
        "metadata": metadata,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "early_stopping": {
            "best_score": float(early_stopping.best_score),
            "counter": int(early_stopping.counter),
            "early_stop": bool(early_stopping.early_stop),
        },
        "rng_state": _capture_rng_state(),
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_latest_training_state(path, model, optimizer, scheduler,
                                early_stopping, expected_metadata):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported resume-state schema: {path}")
    metadata = state.get("metadata", {})
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Resume-state metadata mismatch in {path}: {mismatches}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    early_state = state["early_stopping"]
    early_stopping.best_score = early_state["best_score"]
    early_stopping.counter = early_state["counter"]
    early_stopping.early_stop = early_state["early_stop"]
    _restore_rng_state(state["rng_state"])
    return int(state["epoch"]) + 1


# ============================================================
# 全局日志配置：同时输出到控制台和文件
# ============================================================
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_{LOG_TIMESTAMP}.log")

class TeeLogger:
    """同时将输出写入控制台和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8', buffering=1)  # 行缓冲
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# 安装全局日志重定向
_tee = TeeLogger(LOG_FILE)
sys.stdout = _tee
print(f"[LOG] 日志文件: {LOG_FILE}")
print(f"[LOG] 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[LOG] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, PyG: {torch_geometric.__version__}")

def run_MIF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device,
                  DATA_DIR=None, PREPROCESS_DIR=None, RESUME=False,
                  NOTIFIER=None, DEBUG_DATA_VALIDATION=False,
                  RUNTIME_OPTIMIZATION=True):
    """
    运行MIF-DTI模型的K折交叉验证训练和测试流程
    
    Args:
        SEED: 随机种子
        DATASET: 数据集名称
        MODEL: 模型类
        K_Fold: K折交叉验证的折数
        LOSS: 损失函数类型
        device: 计算设备
        DATA_DIR: 包含 <DATASET>.txt 的目录；None 时由 config/env/default 解析
        PREPROCESS_DIR: 蛋白、配体和子口袋缓存目录
        RESUME: 若存在逐 epoch 状态，则从下一个 epoch 继续
        NOTIFIER: 可选的非致命通知对象
        DEBUG_DATA_VALIDATION: 启用逐张量同步数值检查（仅用于诊断）
        RUNTIME_OPTIMIZATION: 使用严格等价优化后的运行热路径
    """
    # 设置随机种子以确保可复现性
    set_experiment_seed(SEED)

    # 初始化超参数
    hp = hyperparameter(DATASET, data_dir=DATA_DIR, preprocess_dir=PREPROCESS_DIR)
    data_dir = hp.data_dir
    preprocess_dir = hp.preprocess_dir
    os.makedirs(preprocess_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Preprocessed cache directory: {preprocess_dir}")
    
    # 自动检测子口袋预处理文件（如果未指定且文件存在）
    if hp.use_subpocket and hp.subpocket_cache_path is None:
        default_cache_path = os.path.join(preprocess_dir, f'{DATASET}-subpocket.pt')
        if os.path.exists(default_cache_path):
            print(f"检测到子口袋预处理文件，自动使用: {default_cache_path}")
            hp.subpocket_cache_path = default_cache_path

    shared_subpocket_cache = None
    if (hp.use_subpocket and hp.subpocket_cache_path is not None
            and os.path.isfile(hp.subpocket_cache_path)):
        shared_subpocket_cache = load_prepared_subpocket_cache(
            hp.subpocket_cache_path, max_subpockets=hp.pocket_num
        )

    # 从文本文件加载数据集
    print("Train in " + DATASET)
    print("load data")
    dir_input = os.path.join(data_dir, f'{DATASET}.txt')
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")
    dataset_sha256 = file_sha256(dir_input)

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
        # BUGFIX: 原代码 data_list[split_pos:-1] 会丢失最后一个测试样本，
        # 现改为 data_list[split_pos:]，测试集占完整 20%
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))
    seed_output_dir = f"./{DATASET}/seed_{SEED}"
    save_split_manifest(
        os.path.join(seed_output_dir, "outer_split_manifest.csv"),
        {"trainval": train_data_list, "outer_test": test_data_list},
        seed=SEED,
        fold=None,
    )

    # 数据准备：加载或生成蛋白质和配体（药物）的图表示
    protein_path = os.path.join(preprocess_dir, f'{DATASET}-protein.pkl')
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

    ligand_path = os.path.join(preprocess_dir, f'{DATASET}-ligand-hi.pkl')
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
        prot_3d_path = os.path.join(preprocess_dir, f'{DATASET}-protein-3d.pkl')
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
    # Each inner-fold checkpoint is evaluated on the same outer test split.
    # Keep its results only as diagnostics; seed-level reporting below averages
    # probabilities per sample across checkpoints before computing metrics.
    outer_test_fold_records = []

    # K折交叉验证循环
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        # 获取当前折的训练集和验证集
        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        save_path = "./" + DATASET + f"/seed_{SEED}/" + "{}".format(i_fold+1)
        save_split_manifest(
            os.path.join(save_path, "split_manifest.csv"),
            {"train": train_dataset, "valid": valid_dataset, "outer_test": test_data_list},
            seed=SEED,
            fold=i_fold + 1,
        )
        # # 重置占位符统计（每个fold开始时重置）
        # reset_placeholder_stats()
        # 创建数据集对象（将文本数据转换为图数据）
        train_dataset = ProteinMoleculeDataset(
            train_dataset, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path,
            subpocket_obj=shared_subpocket_cache,
        )
        valid_dataset = ProteinMoleculeDataset(
            valid_dataset, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path,
            subpocket_obj=shared_subpocket_cache,
        )
        test_dataset = ProteinMoleculeDataset(
            test_data_list, ligand_dict, protein_dict, device=device,
            use_3d=hp.use_3d, pdb_dir=hp.pdb_dir, prot_3d_obj=prot_3d_dict,
            use_subpocket=hp.use_subpocket, subpocket_dir=hp.subpocket_dir, pocket_num=hp.pocket_num,
            subpocket_cache_path=hp.subpocket_cache_path,
            subpocket_obj=shared_subpocket_cache,
        )
        # # 输出占位符统计汇总
        # print_placeholder_stats_summary()
        train_size = len(train_dataset)

        # 创建数据加载器
        loader_stability = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True, **loader_stability)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=False, **loader_stability)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=False, **loader_stability)
                                    
        # 创建模型
        model = MODEL(
            device=device,
            use_subpocket=hp.use_subpocket,
            pocket_num=hp.pocket_num,
            debug_data_validation=DEBUG_DATA_VALIDATION,
            runtime_optimization=RUNTIME_OPTIMIZATION,
        )

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
        # BUGFIX: 原代码忽略 LOSS 参数、固定使用 CELoss，而 main.py 传入 'PolyLoss'，造成论文/代码不一致。
        # 现按 LOSS 参数实际选择；论文 Eq.(9)(10) 应描述为 CE（2 类 logits + softmax）。
        if LOSS == 'PolyLoss':
            Loss = PolyLoss(weight_loss=weight_loss,
                            DEVICE=device, epsilon=hp.loss_epsilon)
        else:
            Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

        # 设置输出文件路径
        # BUGFIX: 加入 seed 维度，避免多种子运行时 checkpoint 相互覆盖
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        # 创建早停机制
        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        resume_path = os.path.join(save_path, "latest_training_state.pt")
        resume_metadata = {
            "dataset": DATASET,
            "dataset_sha256": dataset_sha256,
            "seed": int(SEED),
            "fold": int(i_fold + 1),
            "inner_folds": int(K_Fold),
            "loss": LOSS,
            "model": MODEL.__name__,
            # Do not silently continue one formal run under a different CUDA
            # execution policy.  A diagnostic stability run must start from
            # its own state file (or resume another stability run).
            "cuda_stability": bool(CUDA_DEBUG and STRICT_DETERMINISM),
            "runtime_optimization": bool(RUNTIME_OPTIMIZATION),
        }
        start_epoch = 1
        if RESUME and os.path.isfile(resume_path):
            start_epoch = _load_latest_training_state(
                resume_path, model, optimizer, scheduler, early_stopping, resume_metadata
            )
            print(f"[RESUME] fold={i_fold + 1}, continuing from epoch {start_epoch}")

        # 开始训练
        print('Training...')
        for epoch in range(start_epoch, hp.Epoch + 1):
            # 如果早停条件满足，退出训练
            if early_stopping.early_stop == True:
                break

            # 每个epoch开始时重置统计
            # reset_placeholder_stats()
            
            # GPU 显存监控（每个 epoch 打印）
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                if CUDA_DEBUG:
                    torch.cuda.synchronize(device)
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"[GPU Mem] Epoch {epoch}: allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB")

            # 训练阶段
            train_losses_in_epoch = []
            model.train()
            
            print(f"[TRAIN] Epoch {epoch}: starting DataLoader...", flush=True)
            for batch_idx, data in enumerate(train_loader):
                if CUDA_DEBUG:
                    print(f"[CUDA DEBUG] epoch={epoch} batch={batch_idx} collated", flush=True)
                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                data = data.to(device)
                if CUDA_DEBUG and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                predicted_y = model(data)
                if CUDA_DEBUG and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                if DEBUG_DATA_VALIDATION and not torch.isfinite(predicted_y).all().item():
                    raise FloatingPointError(
                        f"Non-finite logits at epoch {epoch}, batch {batch_idx}."
                    )
                # 计算损失
                train_loss = Loss(predicted_y, data.cls_y)
                _assert_finite_scalar(
                    train_loss,
                    f"Non-finite loss at epoch {epoch}, batch {batch_idx}.",
                )
                train_losses_in_epoch.append(train_loss.detach())
                
                # 反向传播
                train_loss.backward()
                if CUDA_DEBUG and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 更新参数
                optimizer.step()
                scheduler.step()
                if CUDA_DEBUG and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    print(f"[CUDA DEBUG] epoch={epoch} batch={batch_idx} completed", flush=True)
                
                del data, predicted_y, train_loss
                    
            # 每个 epoch 结束后清理 GPU 缓存
            if CUDA_DEBUG:
                torch.cuda.empty_cache()
            # 计算一个epoch的平均训练损失
            train_loss_a_epoch = _mean_detached_losses(train_losses_in_epoch)

            # 验证阶段
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []  # 真实标签、预测标签、预测分数
            with torch.no_grad():
                for batch_idx, data in enumerate(valid_loader):
                    data = data.to(device)
                    if CUDA_DEBUG and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    valid_scores = model(data)
                    if CUDA_DEBUG and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    
                    valid_labels = data.cls_y
                    # 计算验证损失
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.detach())
                    # 转换为numpy数组用于计算指标
                    valid_labels_np = valid_labels.to('cpu').data.numpy()
                    valid_scores_np = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    if not np.isfinite(valid_scores_np).all():
                        raise FloatingPointError(
                            f"Non-finite validation scores at epoch {epoch}, batch {batch_idx}."
                        )
                    valid_predictions = np.argmax(valid_scores_np, axis=1)
                    valid_scores_np = valid_scores_np[:, 1]  # 正类概率

                    # 收集所有批次的预测结果
                    Y.extend(valid_labels_np)
                    P.extend(valid_predictions)
                    S.extend(valid_scores_np)
                    
                    # 释放 GPU 显存引用
                    del data, valid_scores, valid_labels
            # 验证完成后清理 GPU 缓存
            if CUDA_DEBUG:
                torch.cuda.empty_cache()

            # 计算验证集指标
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = _mean_detached_losses(valid_losses_in_epoch)

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
            if RESUME:
                _save_latest_training_state(
                    resume_path, model, optimizer, scheduler, epoch,
                    early_stopping, resume_metadata,
                )

        # 加载最佳检查点
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{device}.pth', weights_only=True))

        # 在训练集、验证集和测试集上测试模型
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1,
            MIF=True, seed=SEED, fold=i_fold + 1, split="train")
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1,
            MIF=True, seed=SEED, fold=i_fold + 1, split="valid")
        test_output = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1,
            MIF=True, seed=SEED, fold=i_fold + 1, split="outer_test", return_predictions=True)
        (testset_test_stable_results, fold_accuracy, fold_precision, fold_recall,
         fold_auc, fold_prc, fold_test_records) = test_output
        outer_test_fold_records.append(fold_test_records)
        # 保存当前折的测试结果
        # 将结果写入文件
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'w', encoding='utf-8') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')
        if NOTIFIER is not None:
            NOTIFIER.send(
                "DTI Fold Done",
                "Fold completed successfully.\n"
                f"dataset: {DATASET}\nseed: {SEED}\nfold: {i_fold + 1}/{K_Fold}\n"
                f"AUC: {fold_auc:.4f}  PRC: {fold_prc:.4f}\n"
                f"Accuracy: {fold_accuracy:.4f}  Recall: {fold_recall:.4f}  "
                f"Precision: {fold_precision:.4f}",
            )

    # 显示K折交叉验证的平均结果

    # BUGFIX: 返回该 seed 下 K 折的测试集平均指标（用于多种子汇总 mean±std）
    if len(outer_test_fold_records) != K_Fold:
        raise RuntimeError("Missing outer-test predictions from one or more inner folds.")
    reference_records = outer_test_fold_records[0]
    if not reference_records:
        raise RuntimeError("Outer test evaluation produced no predictions.")
    score_matrix = []
    for fold_number, records in enumerate(outer_test_fold_records, start=1):
        if len(records) != len(reference_records):
            raise RuntimeError(f"Fold {fold_number} changed outer-test prediction count.")
        for ref, current in zip(reference_records, records):
            if (ref["sample_index"], ref["sample_id"], ref["y_true"]) != (
                    current["sample_index"], current["sample_id"], current["y_true"]):
                raise RuntimeError(f"Fold {fold_number} changed outer-test sample alignment.")
        score_matrix.append([record["y_score"] for record in records])

    mean_scores = np.mean(np.asarray(score_matrix, dtype=float), axis=0)
    y_true = [record["y_true"] for record in reference_records]
    y_pred = (mean_scores >= 0.5).astype(int).tolist()
    seed_metrics = prediction_metrics(y_true, y_pred, mean_scores)
    ensemble_records = []
    for record, mean_score, prediction in zip(reference_records, mean_scores, y_pred):
        ensemble_records.append({**record, "y_score": float(mean_score), "y_pred": int(prediction)})
    save_prediction_records(
        ensemble_records,
        os.path.join(seed_output_dir, f"{DATASET}_outer_test_ensemble_predictions_seed_{SEED}.csv"),
        seed=SEED, fold="probability_mean_over_inner_folds", split="outer_test")
    with open(os.path.join(seed_output_dir, "outer_test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": DATASET,
            "seed": SEED,
            "inner_folds": K_Fold,
            "protocol": "fixed outer 20% test; probability mean over inner-fold checkpoints",
            "runtime_optimization": bool(RUNTIME_OPTIMIZATION),
            "n_outer_test": len(reference_records),
            "metrics": seed_metrics,
        }, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("[Outer test] Seed-level metrics are computed after averaging each sample's "
          f"positive-class probability across {K_Fold} inner-fold checkpoints: {seed_metrics}")
    return seed_metrics
    

def ensemble_run_MIF_model(SEED, DATASET, K_Fold, device, LOSS="CE",
                           DATA_DIR=None, PREPROCESS_DIR=None):
    """
    运行集成MIF-DTI模型测试流程
    加载K折训练好的模型，在测试集上进行集成预测
    
    Args:
        SEED: 随机种子
        DATASET: 数据集名称
        K_Fold: K折数量（需要加载K个模型）
        device: 计算设备
        DATA_DIR: 包含 <DATASET>.txt 的目录；None 时由 config/env/default 解析
        PREPROCESS_DIR: 蛋白、配体和子口袋缓存目录
    """
    # 设置随机种子
    set_experiment_seed(SEED)

    # 初始化超参数
    hp = hyperparameter(DATASET, data_dir=DATA_DIR, preprocess_dir=PREPROCESS_DIR)
    data_dir = hp.data_dir
    preprocess_dir = hp.preprocess_dir
    os.makedirs(preprocess_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Preprocessed cache directory: {preprocess_dir}")
    
    # 自动检测子口袋预处理文件（如果未指定且文件存在）
    if hp.use_subpocket and hp.subpocket_cache_path is None:
        default_cache_path = os.path.join(preprocess_dir, f'{DATASET}-subpocket.pt')
        if os.path.exists(default_cache_path):
            print(f"检测到子口袋预处理文件，自动使用: {default_cache_path}")
            hp.subpocket_cache_path = default_cache_path

    # 加载数据集
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = os.path.join(data_dir, f'{DATASET}.txt')
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
    # BUGFIX: 与训练流程保持一致，测试集占完整 20%（原代码丢失最后一个样本）
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/seed_{SEED}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 数据准备：加载或生成蛋白质和配体图
    protein_path = os.path.join(preprocess_dir, f'{DATASET}-protein.pkl')
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = os.path.join(preprocess_dir, f'{DATASET}-ligand-hi.pkl')
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
        prot_3d_path = os.path.join(preprocess_dir, f'{DATASET}-protein-3d.pkl')
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
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=False)

    # 加载K折训练好的模型
    model = []
    for i in range(K_Fold):
        model.append(MIFDTI(device=device, use_subpocket=hp.use_subpocket, pocket_num=hp.pocket_num).to(device))
        # 加载对应折的模型权重（需要先运行K折训练）
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/seed_{SEED}/' + f'{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device)))
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MIFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    # 创建损失函数（用于测试，实际不用于反向传播）
    # BUGFIX: 与训练流程保持一致，使用 CELoss
    if LOSS == 'PolyLoss':
        Loss = PolyLoss(weight_loss=weight_loss, DEVICE=device, epsilon=hp.loss_epsilon)
    else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

    # 使用集成模型在测试集上进行预测
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold,
            MIF=True, seed=SEED, fold="ensemble", split="outer_test")
    
    # 显示集成模型的测试结果
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
    

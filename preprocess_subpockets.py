# -*- coding:utf-8 -*-
"""
预处理脚本：提前提取所有蛋白质的子口袋特征并保存到磁盘
这样在训练时可以快速加载，而不需要每次都重新计算
"""

import os
import sys
import torch
from tqdm import tqdm
import warnings

# 抑制RDKit的警告
warnings.filterwarnings('ignore', message='.*GetValence.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.getLogger('rdkit').setLevel(logging.ERROR)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.subpocket_featurizer import process_protein_subpocket
from config import hyperparameter


def extract_protein_names(dataset_file):
    """
    从数据集文件中提取所有唯一的蛋白质名称
    
    Args:
        dataset_file: 数据集文件路径
        
    Returns:
        set: 蛋白质名称集合
    """
    protein_names = set()
    with open(dataset_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(' ')
            if len(items) >= 2:
                # 蛋白质名称在第二个位置（索引1）
                protein_name = items[1]
                protein_names.add(protein_name)
    return protein_names


def preprocess_subpockets(dataset_name, pdb_dir, subpocket_dir, max_subpockets=30, 
                         output_dir=None, force_reprocess=False):
    """
    预处理所有蛋白质的子口袋特征
    
    Args:
        dataset_name: 数据集名称（如 'Davis', 'KIBA'）
        pdb_dir: PDB文件目录
        subpocket_dir: 子口袋文件目录
        max_subpockets: 最大子口袋数量
        output_dir: 输出目录（如果为None，使用默认路径）
        force_reprocess: 是否强制重新处理（即使缓存文件已存在）
    """
    # 设置输出路径
    if output_dir is None:
        output_dir = './DataSets/Preprocessed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{dataset_name}-subpocket.pt')
    
    # 检查缓存文件是否存在
    if os.path.exists(output_path) and not force_reprocess:
        print(f"缓存文件已存在: {output_path}")
        print("如果要重新处理，请使用 force_reprocess=True")
        response = input("是否加载现有缓存？(y/n): ")
        if response.lower() != 'y':
            print("取消操作")
            return
        else:
            # 加载并显示统计信息
            subpocket_dict = torch.load(output_path)
            print(f"\n已加载 {len(subpocket_dict)} 个蛋白质的子口袋特征")
            valid_count = sum(1 for v in subpocket_dict.values() if v is not None)
            print(f"其中 {valid_count} 个有有效的子口袋数据")
            return
    
    # 读取数据集文件
    dataset_file = f'./DataSets/{dataset_name}.txt'
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")
    
    print(f"正在从 {dataset_file} 提取蛋白质名称...")
    protein_names = extract_protein_names(dataset_file)
    print(f"找到 {len(protein_names)} 个唯一的蛋白质")
    
    # 预处理每个蛋白质
    subpocket_dict = {}
    failed_proteins = []
    
    print(f"\n开始预处理子口袋特征（最大子口袋数: {max_subpockets}）...")
    print(f"PDB目录: {pdb_dir}")
    print(f"子口袋目录: {subpocket_dir}")
    print("-" * 80)
    
    for protein_name in tqdm(sorted(protein_names), desc="处理蛋白质"):
        try:
            subpocket_data = process_protein_subpocket(
                protein_name, 
                pdb_dir, 
                subpocket_dir, 
                max_subpockets=max_subpockets
            )
            subpocket_dict[protein_name] = subpocket_data
        except Exception as e:
            print(f"\n警告: 处理 {protein_name} 时出错: {e}")
            failed_proteins.append(protein_name)
            subpocket_dict[protein_name] = None
    
    # 保存结果
    print(f"\n保存预处理结果到 {output_path}...")
    torch.save(subpocket_dict, output_path)
    print(f"保存完成！")
    
    # 统计信息
    total = len(subpocket_dict)
    valid = sum(1 for v in subpocket_dict.values() if v is not None)
    failed = len(failed_proteins)
    
    print(f"\n预处理统计:")
    print(f"  总蛋白质数: {total}")
    print(f"  成功处理: {valid}")
    print(f"  失败/跳过: {failed}")
    
    if failed_proteins:
        print(f"\n失败的蛋白质列表:")
        for p in failed_proteins[:10]:  # 只显示前10个
            print(f"  - {p}")
        if len(failed_proteins) > 10:
            print(f"  ... 还有 {len(failed_proteins) - 10} 个")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python preprocess_subpockets.py <dataset_name> [--force]")
        print("示例: python preprocess_subpockets.py Davis")
        print("      python preprocess_subpockets.py Davis --force  # 强制重新处理")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    force_reprocess = '--force' in sys.argv
    
    # 从配置文件获取路径
    hp = hyperparameter(dataset_name)
    pdb_dir = hp.pdb_dir
    subpocket_dir = hp.subpocket_dir
    max_subpockets = hp.pocket_num
    
    # 如果配置中的路径是特定数据集的，需要调整
    # 这里假设路径已经正确配置
    print(f"数据集: {dataset_name}")
    print(f"PDB目录: {pdb_dir}")
    print(f"子口袋目录: {subpocket_dir}")
    print(f"最大子口袋数: {max_subpockets}")
    print()
    
    preprocess_subpockets(
        dataset_name=dataset_name,
        pdb_dir=pdb_dir,
        subpocket_dir=subpocket_dir,
        max_subpockets=max_subpockets,
        force_reprocess=force_reprocess
    )


if __name__ == '__main__':
    main()

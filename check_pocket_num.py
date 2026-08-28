# -*- coding:utf-8 -*-
"""
检查数据集中每个蛋白质的实际子口袋数量
用于验证 pocket_num=30 是否合适
"""

import os
import sys
import torch
from collections import Counter
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import hyperparameter
from utils.subpocket_featurizer import process_protein_subpocket


def count_subpockets_in_directory(subpocket_dir, protein_name):
    """
    统计指定蛋白质目录中的子口袋文件数量
    
    Args:
        subpocket_dir: 子口袋根目录
        protein_name: 蛋白质名称
        
    Returns:
        int: 子口袋数量
    """
    base_dir = os.path.join(subpocket_dir, protein_name)
    if not os.path.exists(base_dir):
        return 0
    
    subpocket_count = 0
    pocket_number = 1
    
    while pocket_number < 10:  # 最多检查10个pocket
        subpocket_number = 1
        found_any = False
        
        while subpocket_number <= 30:  # 每个pocket最多30个子口袋
            subpocket_file = os.path.join(base_dir, f"subpocket_{pocket_number}_{subpocket_number}.pdb")
            if os.path.exists(subpocket_file) and os.path.isfile(subpocket_file):
                subpocket_count += 1
                found_any = True
                subpocket_number += 1
            else:
                if subpocket_number == 1:
                    # 如果第一个子口袋不存在，说明这个pocket不存在
                    break
                else:
                    # 如果中间某个子口袋不存在，继续下一个pocket
                    break
        
        if not found_any:
            break
        
        pocket_number += 1
    
    return subpocket_count


def check_pocket_distribution(dataset_name, subpocket_dir, max_check=1000):
    """
    检查数据集中子口袋数量的分布
    
    Args:
        dataset_name: 数据集名称
        subpocket_dir: 子口袋目录
        max_check: 最多检查的蛋白质数量
    """
    # 读取数据集文件
    dataset_file = f'./DataSets/{dataset_name}.txt'
    if not os.path.exists(dataset_file):
        print(f"数据集文件不存在: {dataset_file}")
        return
    
    # 提取蛋白质名称
    protein_names = set()
    with open(dataset_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(' ')
            if len(items) >= 2:
                protein_name = items[1]
                protein_names.add(protein_name)
    
    print(f"找到 {len(protein_names)} 个唯一的蛋白质")
    print(f"检查前 {min(max_check, len(protein_names))} 个蛋白质...")
    print("-" * 80)
    
    # 统计子口袋数量
    pocket_counts = []
    valid_proteins = 0
    invalid_proteins = 0
    
    for protein_name in tqdm(list(protein_names)[:max_check], desc="统计子口袋"):
        count = count_subpockets_in_directory(subpocket_dir, protein_name)
        if count > 0:
            pocket_counts.append(count)
            valid_proteins += 1
        else:
            invalid_proteins += 1
    
    if len(pocket_counts) == 0:
        print("没有找到有效的子口袋数据！")
        return
    
    # 统计信息
    print("\n" + "=" * 80)
    print("子口袋数量统计:")
    print("=" * 80)
    print(f"有效蛋白质数: {valid_proteins}")
    print(f"无效蛋白质数: {invalid_proteins}")
    print(f"总检查数: {valid_proteins + invalid_proteins}")
    print()
    
    print(f"子口袋数量分布:")
    counter = Counter(pocket_counts)
    for count in sorted(counter.keys()):
        num_proteins = counter[count]
        percentage = (num_proteins / len(pocket_counts)) * 100
        print(f"  {count:3d} 个子口袋: {num_proteins:4d} 个蛋白质 ({percentage:5.2f}%)")
    
    print()
    print(f"最小值: {min(pocket_counts)}")
    print(f"最大值: {max(pocket_counts)}")
    print(f"平均值: {sum(pocket_counts) / len(pocket_counts):.2f}")
    print(f"中位数: {sorted(pocket_counts)[len(pocket_counts) // 2]}")
    
    # 检查是否超过30
    over_30 = sum(1 for c in pocket_counts if c > 30)
    under_30 = sum(1 for c in pocket_counts if c < 30)
    exactly_30 = sum(1 for c in pocket_counts if c == 30)
    
    print()
    print("=" * 80)
    print("关于 pocket_num=30 的兼容性:")
    print("=" * 80)
    print(f"子口袋数 > 30 的蛋白质: {over_30} ({over_30/len(pocket_counts)*100:.2f}%)")
    print(f"子口袋数 = 30 的蛋白质: {exactly_30} ({exactly_30/len(pocket_counts)*100:.2f}%)")
    print(f"子口袋数 < 30 的蛋白质: {under_30} ({under_30/len(pocket_counts)*100:.2f}%)")
    
    if over_30 > 0:
        print(f"\n⚠️  警告: 有 {over_30} 个蛋白质的子口袋数超过30，会被截断！")
        print("   建议: 考虑增加 pocket_num 或检查数据预处理")
    
    if max(pocket_counts) < 30:
        print(f"\n✓ 所有蛋白质的子口袋数都小于30，pocket_num=30 是安全的")
        print(f"  可以考虑降低 pocket_num 以减少内存占用")
    
    # 检查实际处理后的数量（使用process_protein_subpocket）
    print()
    print("=" * 80)
    print("检查实际处理后的子口袋数量（使用process_protein_subpocket）:")
    print("=" * 80)
    
    hp = hyperparameter(dataset_name)
    processed_counts = []
    
    sample_proteins = list(protein_names)[:min(100, len(protein_names))]
    for protein_name in tqdm(sample_proteins, desc="处理样本"):
        try:
            subpocket_data = process_protein_subpocket(
                protein_name, 
                hp.pdb_dir, 
                subpocket_dir, 
                max_subpockets=30
            )
            if subpocket_data is not None:
                # 计算实际的子口袋数量
                if hasattr(subpocket_data, 'ptr'):
                    num_graphs = len(subpocket_data.ptr) - 1
                else:
                    # 如果没有ptr，尝试从batch推断
                    if hasattr(subpocket_data, 'batch'):
                        num_graphs = len(torch.unique(subpocket_data.batch))
                    else:
                        num_graphs = 0
                processed_counts.append(num_graphs)
        except Exception as e:
            pass
    
    if len(processed_counts) > 0:
        print(f"\n处理后的子口袋数量统计（样本数: {len(processed_counts)}）:")
        print(f"  最小值: {min(processed_counts)}")
        print(f"  最大值: {max(processed_counts)}")
        print(f"  平均值: {sum(processed_counts) / len(processed_counts):.2f}")
        print(f"  中位数: {sorted(processed_counts)[len(processed_counts) // 2]}")
        
        counter_processed = Counter(processed_counts)
        print(f"\n  分布:")
        for count in sorted(counter_processed.keys())[:20]:  # 只显示前20个
            num = counter_processed[count]
            print(f"    {count:3d} 个子口袋: {num:3d} 个蛋白质")


def main():
    if len(sys.argv) < 2:
        print("用法: python check_pocket_num.py <dataset_name> [max_check]")
        print("示例: python check_pocket_num.py Davis 1000")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    max_check = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    hp = hyperparameter(dataset_name)
    subpocket_dir = hp.subpocket_dir
    
    if subpocket_dir is None:
        print("错误: subpocket_dir 未配置")
        sys.exit(1)
    
    print(f"数据集: {dataset_name}")
    print(f"子口袋目录: {subpocket_dir}")
    print()
    
    check_pocket_distribution(dataset_name, subpocket_dir, max_check)


if __name__ == '__main__':
    main()

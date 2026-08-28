#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查子口袋文件是否能被正确解析
"""

import os
import numpy as np

def parse_pdb_for_bounding_box(pdb_file):
    """从PDB文件中解析边界框"""
    with open(pdb_file, 'r') as file:
        x_coords, y_coords, z_coords = [], [], []
        for line in file:
            if line.startswith("ATOM"):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                except (ValueError, IndexError) as e:
                    continue
        if len(x_coords) == 0:
            return None, None
        min_coords = np.array([min(x_coords), min(y_coords), min(z_coords)])
        max_coords = np.array([max(x_coords), max(y_coords), max(z_coords)])
    return min_coords, max_coords


def check_subpocket_files(protein_names, subpocket_dir):
    """检查子口袋文件"""
    results = {}
    
    for protein_name in protein_names:
        base_dir = os.path.join(subpocket_dir, protein_name)
        if not os.path.exists(base_dir):
            results[protein_name] = {
                'status': '目录不存在',
                'files_found': 0,
                'files_parsed': 0,
                'errors': []
            }
            continue
        
        # 查找所有子口袋文件
        subpocket_files = []
        for pocket_num in range(1, 11):
            for subpocket_num in range(1, 31):
                file_path = os.path.join(base_dir, f"subpocket_{pocket_num}_{subpocket_num}.pdb")
                if os.path.exists(file_path):
                    subpocket_files.append((pocket_num, subpocket_num, file_path))
        
        # 解析文件
        parsed_count = 0
        errors = []
        for pocket_num, subpocket_num, file_path in subpocket_files[:10]:  # 只检查前10个
            box_min, box_max = parse_pdb_for_bounding_box(file_path)
            if box_min is None or box_max is None:
                errors.append(f"subpocket_{pocket_num}_{subpocket_num}.pdb: 无法解析边界框")
            else:
                parsed_count += 1
        
        results[protein_name] = {
            'status': '正常' if len(subpocket_files) > 0 else '无文件',
            'files_found': len(subpocket_files),
            'files_parsed': parsed_count,
            'errors': errors[:5]  # 只显示前5个错误
        }
    
    return results


if __name__ == "__main__":
    # 要检查的蛋白质列表
    protein_names = [
        'P16471', 'P08172', 'Q15118', 'P14867', 'Q15825',
        'P04183', 'P23677', 'P78352', 'P23458', 'P24046',
        'P48169', 'Q6TME4', 'P09237', 'O75251'
    ]
    
    subpocket_dir = "./DataSets/pdb/BIOSNAP_subpocket/"
    
    print("=" * 80)
    print("检查子口袋文件")
    print("=" * 80)
    print()
    
    results = check_subpocket_files(protein_names, subpocket_dir)
    
    for protein_name, result in results.items():
        print(f"【{protein_name}】")
        print(f"  状态: {result['status']}")
        print(f"  找到文件数: {result['files_found']}")
        print(f"  成功解析: {result['files_parsed']}")
        if result['errors']:
            print(f"  错误:")
            for error in result['errors']:
                print(f"    - {error}")
        print()


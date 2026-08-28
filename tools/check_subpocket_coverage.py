# -*- coding: utf-8 -*-
"""
检查数据集蛋白的子口袋/PDB 结构覆盖率（大修 P0-06 配套）
检查来源（按优先级）:
  1. 子口袋缓存: ./DataSets/Preprocessed/{ds}-subpocket.pt 和 ./DataSets/Preprocessed_old_backup/{ds}-subpocket.pt
  2. 子口袋目录: ./DataSets/pdb/{ds}_subpocket/**/subpocket_*.pdb
  3. PDB 结构目录: ./DataSets/pdb/{ds}_pdb/**/*.pdb（只有全结构，无子口袋）

输出: 有子口袋 / 仅有PDB结构 / 两者皆无(虚拟口袋) 的数量与名单

用法:
  python tools/check_subpocket_coverage.py [数据集名，默认全部]
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')


def protein_names_of(dataset):
    names = set()
    path = f'./DataSets/{dataset}.txt'
    for l in open(path, encoding='utf-8', errors='replace'):
        p = l.strip().split(' ')
        if len(p) >= 5:
            names.add(p[1])
    return names


def cache_covered(dataset):
    """从子口袋缓存 .pt 获取有子口袋数据的蛋白集合"""
    covered = set()
    for sub in ['Preprocessed', 'Preprocessed_old_backup']:
        cache = f'./DataSets/{sub}/{dataset}-subpocket.pt'
        if os.path.exists(cache):
            import torch
            d = torch.load(cache, map_location='cpu')
            for k, v in d.items():
                if v is not None and hasattr(v, 'x') and v.x is not None and v.x.numel() > 0:
                    covered.add(k)
            print(f'  缓存 {cache}: {len(d)} 条目, 有效 {len([1 for v in d.values() if v is not None])}')
    return covered


def dir_covered(dataset):
    """从子口袋 PDB 文件目录统计有子口袋的蛋白"""
    covered = set()
    d = f'./DataSets/pdb/{dataset}_subpocket'
    if os.path.isdir(d):
        n_dirs = 0
        for name in os.listdir(d):
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                n_dirs += 1
                files = os.listdir(sub)
                if any(f.startswith('subpocket') and f.endswith('.pdb') for f in files):
                    covered.add(name)
        print(f'  子口袋目录 {d}: {n_dirs} 个子目录, 含 subpocket 文件的 {len(covered)} 个')
    return covered


def pdb_covered(dataset):
    """统计有 PDB 全结构文件的蛋白（不含子口袋分解）"""
    covered = set()
    d = f'./DataSets/pdb/{dataset}_pdb'
    if os.path.isdir(d):
        n_dirs = 0
        for name in os.listdir(d):
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                n_dirs += 1
                if any(f.endswith('.pdb') for f in os.listdir(sub)):
                    covered.add(name)
        print(f'  PDB 结构目录 {d}: {n_dirs} 个子目录, 含 pdb 文件的 {len(covered)} 个')
    return covered


def check_dataset(dataset):
    names = protein_names_of(dataset)
    print(f'[{dataset}] txt 蛋白总数: {len(names)}')
    subp = cache_covered(dataset) | dir_covered(dataset)
    pdb = pdb_covered(dataset)
    only_pdb = (pdb - subp) & names
    neither = names - subp - pdb
    print(f'[{dataset}] 有子口袋: {len(subp & names)} ({100 * len(subp & names) / max(len(names), 1):.1f}%) | '
          f'仅PDB结构无子口袋: {len(only_pdb)} | 两者皆无(虚拟口袋): {len(neither)} ({100 * len(neither) / max(len(names), 1):.1f}%)')
    if only_pdb:
        print(f'  仅PDB结构名单: {sorted(only_pdb)}')
    if neither:
        print(f'  虚拟口袋名单: {sorted(neither)}')


if __name__ == '__main__':
    dss = sys.argv[1:] if len(sys.argv) > 1 else ['Davis', 'DrugBank', 'BIOSNAP']
    for ds in dss:
        print('=' * 70)
        check_dataset(ds)

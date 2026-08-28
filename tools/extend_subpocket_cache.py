# -*- coding: utf-8 -*-
"""
增量扩展子口袋缓存（大修专用）
=========================================
背景: BIOSNAP 清洗后新增了 31 个映射出的蛋白 ID，旧缓存里没有它们的条目。
      其中 28 个在 subpocket 目录中有注释文件，只有 3 个没有（会生成虚拟口袋）。

用法:
  python tools/extend_subpocket_cache.py BIOSNAP <subpocket_dir> <pdb_dir>

  例如:
  python tools/extend_subpocket_cache.py BIOSNAP \
      /home/zjx/00_Code/MIF+SP_v2_final_大修/DataSets/pdb/BIOSNAP_subpocket \
      /home/zjx/00_Code/MIF+SP_v2_final_大修/DataSets/pdb/BIOSNAP_pdb

行为:
  1. 读取旧缓存 ./DataSets/Preprocessed_old_backup/{ds}-subpocket.pt（若存在）
     或 ./DataSets/Preprocessed/{ds}-subpocket.pt
  2. 找出 txt 中不在缓存里的蛋白名，逐个用 process_protein_subpocket 生成
  3. 合并后写回 ./DataSets/Preprocessed/{ds}-subpocket.pt
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
# 确保能 import 项目根目录下的 utils 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ds = sys.argv[1]
    subpocket_dir = sys.argv[2] if len(sys.argv) > 2 else None
    pdb_dir = sys.argv[3] if len(sys.argv) > 3 else None

    import torch
    from utils.subpocket_featurizer import process_protein_subpocket

    txt_path = f'./DataSets/{ds}.txt'
    names = []
    for l in open(txt_path, encoding='utf-8', errors='replace'):
        p = l.strip().split(' ')
        if len(p) >= 5:
            names.append(p[1])
    names = sorted(set(names))
    print(f'[{ds}] txt 蛋白数: {len(names)}')

    # 找旧缓存
    cache = None
    for cand in [f'./DataSets/Preprocessed_old_backup/{ds}-subpocket.pt',
                 f'./DataSets/Preprocessed/{ds}-subpocket.pt']:
        if os.path.exists(cand):
            cache = cand
            break
    if cache:
        print(f'加载旧缓存: {cache}')
        d = torch.load(cache, map_location='cpu')
    else:
        print('未找到旧缓存，从空开始')
        d = {}

    # 需要补充的：不在缓存里，或缓存值为 None（上次生成失败/维度崩溃留下的占位）
    missing = [n for n in names if n not in d or d.get(n) is None]
    print(f'缓存已有: {len(d)}，需补充: {len(missing)}')
    if missing:
        print('缺失名单:', missing)
    if not missing:
        print('无需补充')
        sys.exit(0)

    if not subpocket_dir or not pdb_dir:
        print('需要提供 <subpocket_dir> <pdb_dir> 参数（见用法）')
        sys.exit(1)

    failed = []
    for name in missing:
        try:
            data = process_protein_subpocket(name, pdb_dir, subpocket_dir, max_subpockets=30)
            d[name] = data
            n_graphs = len(data.ptr) - 1 if hasattr(data, 'ptr') and data.ptr is not None else '?'
            print(f'  {name}: 生成 {n_graphs} 个子口袋图')
        except Exception as e:
            print(f'  {name}: 失败 -> {str(e)[:100]}')
            failed.append(name)
            d[name] = None

    os.makedirs('./DataSets/Preprocessed', exist_ok=True)
    out = f'./DataSets/Preprocessed/{ds}-subpocket.pt'
    torch.save(d, out)
    print(f'已保存合并缓存: {out}（总条目 {len(d)}）')
    if failed:
        print('失败名单（训练时自动落虚拟口袋）:', failed)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
数据集清洗脚本（大修 P0-05）
================================
1. BIOSNAP: 移除 NO-PROT-ID 行（无蛋白 ID -> 无法关联子口袋/3D 结构，避免虚拟占位污染）
   （若能通过 tools/map_noprotid.py 补到 ID，可先补再清洗）
2. 去重（规则 B，推荐）:
   - 同一 (drug, protein) 对出现多行且标签一致 -> 保留第一行，删除其余（防止 train/test 泄漏）
   - 同一对出现多行且标签矛盾 -> 该对全部删除（不人为裁决标签，避免偏置）
3. 输出统计报告，可选 --apply 直接替换 DataSets/*.txt（原文件备份为 *.txt.orig_backup）

用法:
  python clean_datasets.py                # 干跑，只打印统计
  python clean_datasets.py --apply        # 实际替换 DataSets/*.txt
  python clean_datasets.py --rule A       # 使用规则 A（冲突对保留第一条）
"""
import os
import sys
import shutil
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataSets')


def load_lines(dataset):
    with open(os.path.join(BASE, f'{dataset}.txt'), 'r', encoding='utf-8', errors='replace') as f:
        return f.read().strip().split('\n')


def pair_key(parts):
    # drug 标识: DrugBank 用 DB ID; BIOSNAP 无 drug ID，用 SMILES 作为药物标识
    return (parts[2], parts[1])


def stats(lines):
    drugs, prots, pos, neg = set(), set(), 0, 0
    for l in lines:
        p = l.split(' ')
        # BIOSNAP 无 drug ID，用 SMILES 作为药物标识；其余用第 1 列 ID
        drug_id = p[2] if p[0].startswith('NO-') else p[0]
        drugs.add(drug_id)
        prots.add(p[1])
        if p[4] == '1':
            pos += 1
        else:
            neg += 1
    return len(drugs), len(prots), pos, neg


def clean(dataset, rule='B', apply=False, remove_noprotid=True):
    lines = load_lines(dataset)
    n0 = len(lines)

    removed_no_id = []
    kept_no_id = []
    if remove_noprotid and dataset == 'BIOSNAP':
        kept_no_id = [l for l in lines if not l.split(' ')[1].startswith('NO-')]
        removed_no_id = [l for l in lines if l.split(' ')[1].startswith('NO-')]
        lines = kept_no_id
        print(f'[BIOSNAP] 移除 NO-PROT-ID 行: {len(removed_no_id)}（涉及唯一序列 '
              f'{len(set(l.split(" ")[3] for l in removed_no_id))}，其中正样本 '
              f'{sum(1 for l in removed_no_id if l.split(" ")[4] == "1")}）')

    groups = defaultdict(list)
    for l in lines:
        groups[pair_key(l.split(' '))].append(l)

    out, n_dup_del, n_conflict_del = [], 0, 0
    for k, v in groups.items():
        labels = {x.split(' ')[4] for x in v}
        if len(labels) > 1:
            if rule == 'B':
                n_conflict_del += len(v)
                continue
            else:  # 规则 A: 保留第一条
                out.append(v[0])
                n_dup_del += len(v) - 1
        else:
            out.append(v[0])
            n_dup_del += len(v) - 1

    print(f'[{dataset}] 规则{rule}: 原 {n0} 行 -> {len(out)} 行 '
          f'(去重删 {n_dup_del} 行, 冲突对删 {n_conflict_del} 行)')

    d, p, pos, neg = stats(out)
    print(f'[{dataset}] 最终统计: drugs={d} prots={p} pos={pos} neg={neg} pos:neg={pos / max(neg, 1):.3f}')

    if apply:
        src = os.path.join(BASE, f'{dataset}.txt')
        bak = os.path.join(BASE, f'{dataset}.txt.orig_backup')
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
        with open(src, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        print(f'[{dataset}] 已写入 {src}（原文件备份为 {bak}）')
    return out


if __name__ == '__main__':
    apply = '--apply' in sys.argv
    rule = 'A' if '--rule' in sys.argv and sys.argv[sys.argv.index('--rule') + 1] == 'A' else 'B'
    for ds in ['Davis', 'DrugBank', 'BIOSNAP']:
        print('=' * 70)
        clean(ds, rule=rule, apply=apply)

# -*- coding: utf-8 -*-
"""
把 map_noprotid.py 命中的 UniProt accession 写回 BIOSNAP.txt（替换 NO-PROT-ID）
未命中的行保持 NO-PROT-ID，交给 clean_datasets.py 移除

用法:
  python tools/apply_noprotid_mapping.py [BIOSNAP.txt路径] [mapping.json路径]
"""
import sys
import os
import json
import shutil


def main():
    biosnap = sys.argv[1] if len(sys.argv) > 1 else './DataSets/BIOSNAP.txt'
    mapping_file = sys.argv[2] if len(sys.argv) > 2 else './tools/noprotid_mapping.json'
    if not os.path.exists(mapping_file):
        print(f'未找到映射文件: {mapping_file}，请先运行 tools/map_noprotid.py')
        sys.exit(1)

    with open(mapping_file, encoding='utf-8') as f:
        mapping = json.load(f)

    # seq -> accession（命中才替换）
    # isoform accession（如 P60174-3）规范化为主 accession（P60174），
    # 与子口袋缓存/UniProt 主条目键保持一致
    import re as _re
    def canonical(acc):
        if not acc:
            return acc
        return _re.sub(r'-\d+$', '', acc)
    seq2acc = {}
    for k, v in mapping.items():
        if v.get('status') == 'hit' and v.get('seq'):
            seq2acc[v['seq']] = canonical(v['accession'])
    print(f'映射表: {len(seq2acc)} 条序列 -> accession（已规范化 isoform）')

    with open(biosnap, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    replaced = 0
    out = []
    for l in lines:
        parts = l.split(' ')
        if len(parts) >= 5 and parts[1].startswith('NO-'):
            acc = seq2acc.get(parts[3])
            if acc:
                parts[1] = acc
                replaced += 1
        out.append(' '.join(parts))

    bak = biosnap + '.pre_id_backup'
    if not os.path.exists(bak):
        shutil.copy2(biosnap, bak)
    with open(biosnap, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print(f'已替换 {replaced} 行（原文件备份: {bak}）')

    remain = sum(1 for l in out if len(l.split(' ')) >= 5 and l.split(' ')[1].startswith('NO-'))
    print(f'剩余 NO-PROT-ID 行: {remain}（如 >0 用 clean_datasets.py 移除）')


if __name__ == '__main__':
    main()

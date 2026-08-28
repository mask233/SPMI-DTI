# -*- coding: utf-8 -*-
"""
NO-PROT-ID 序列补 ID 脚本（在服务器上运行）
====================================================
方法 1: EBI Proteins API 精确序列匹配（md5 查询，快）
        注意: 必须带 Accept: application/json 头，否则返回 XML 导致解析失败
方法 2: NCBI BLAST (swissprot)（兜底，慢，适合方法 1 未命中/有突变的序列）

输出: tools/noprotid_mapping.json
  {seq_index: {'status': 'hit'|'no_hit', 'accession': ..., 'identity': ..., 'seq': ...}}

用法:
  python tools/map_noprotid.py [BIOSNAP.txt路径]
"""
import sys
import os
import json
import time
import hashlib
import re
import requests

sys.stdout.reconfigure(encoding='utf-8')


def load_noprotid_seqs(biosnap_path):
    seqs = []
    for l in open(biosnap_path, encoding='utf-8', errors='replace'):
        parts = l.strip().split(' ')
        if len(parts) >= 5 and parts[1].startswith('NO-'):
            seqs.append(parts[3])
    seen = set()
    uniq = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def ebi_md5_search(seq):
    """EBI Proteins API 精确序列匹配。返回 hits 列表或 None"""
    md5 = hashlib.md5(seq.upper().encode()).hexdigest()
    r = requests.get(
        'https://www.ebi.ac.uk/proteins/api/proteins',
        params={'size': 10, 'md5': md5},
        headers={'Accept': 'application/json'},
        timeout=60,
    )
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not data:
        return None
    hits = []
    for e in data:
        org = None
        try:
            org = e.get('organism', {}).get('names', [{}])[0].get('value')
        except Exception:
            pass
        reviewed = bool(e.get('reviewed'))
        hits.append({
            'accession': e.get('accession'),
            'name': e.get('name'),
            'organism': org,
            'reviewed': reviewed,
        })
    # 排序：优先 Homo sapiens，其次 reviewed（Swiss-Prot 手工审编条目）
    def rank(h):
        is_human = bool(h['organism'] and 'homo sapiens' in str(h['organism']).lower())
        return (0 if is_human else 1, 0 if h['reviewed'] else 1)
    hits.sort(key=rank)
    return hits


def ncbi_blast(seqs, per_seq_timeout=60):
    """NCBI BLAST 逐条比对，每条限时 per_seq_timeout 秒，超时自动换下一条。

    返回 {seq_index_str: hit_info}；超时的条目 status='timeout'，并打印 RID 供稍后手动查询。
    """
    s = requests.Session()
    mapping = {}
    for idx, seq in enumerate(seqs):
        print(f'\n  BLAST [{idx + 1}/{len(seqs)}] 提交（序列长度 {len(seq)}）...', flush=True)
        try:
            put = s.post(
                'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi',
                data={'CMD': 'Put', 'PROGRAM': 'blastp', 'DATABASE': 'swissprot',
                      'QUERY': f'>q\n{seq}\n', 'NUM_ALIGNMENTS': '1', 'FORMAT_TYPE': 'JSON2_S'},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=60,
            )
        except Exception as e:
            print(f'  提交失败: {str(e)[:80]}，跳过')
            mapping[str(idx)] = {'status': 'no_hit'}
            continue

        m = re.search(r'QBlastInfoBegin(.*?)QBlastInfoEnd', put.text, re.S)
        rid = None
        if m:
            for line in m.group(1).splitlines():
                if line.strip().startswith('RID ='):
                    rid = line.split('=', 1)[1].strip()
        if not rid:
            print('  提交解析失败，跳过')
            mapping[str(idx)] = {'status': 'no_hit'}
            continue

        status = ''
        t0 = time.time()
        while time.time() - t0 < per_seq_timeout:
            try:
                g = s.get(
                    'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi',
                    params={'CMD': 'Get', 'FORMAT_OBJECT': 'SearchInfo', 'RID': rid},
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=30,
                )
                m2 = re.search(r'Status=(\w+)', g.text)
                status = m2.group(1) if m2 else ''
            except Exception:
                status = ''
            print(f'    状态: {status}（已用 {int(time.time() - t0)}s / {per_seq_timeout}s）', flush=True)
            if status in ('READY', 'FAILED', 'UNKNOWN'):
                break
            time.sleep(10)

        if status != 'READY':
            mapping[str(idx)] = {'status': 'timeout', 'rid': rid}
            print(f'    超时跳过。可稍后在浏览器查看: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&RID={rid}')
            continue

        # 结果下载（快速重试）
        data = None
        for attempt in range(3):
            try:
                res = s.get(
                    'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi',
                    params={'CMD': 'Get', 'FORMAT_TYPE': 'JSON2_S', 'RID': rid},
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=45,
                )
                if res.status_code == 200 and len(res.text) > 200:
                    data = res.json()
                    break
            except Exception as e:
                print(f'    下载重试 {attempt}: {str(e)[:80]}')
            time.sleep(5)
        if data is None:
            mapping[str(idx)] = {'status': 'timeout', 'rid': rid}
            print(f'    结果下载失败，跳过。RID={rid}')
            continue

        hits = None
        for rep in data.get('BlastOutput2', []):
            hits = rep.get('report', {}).get('results', {}).get('hits') or []
            break
        if not hits:
            mapping[str(idx)] = {'status': 'no_hit'}
            print('    无命中')
            continue
        d = (hits[0].get('description') or [{}])[0]
        hsp = (hits[0].get('hsps') or [{}])[0]
        mapping[str(idx)] = {
            'status': 'hit',
            'accession': d.get('accession'),
            'title': (d.get('title') or '')[:80],
            'identity': hsp.get('identity'),
            'align_len': hsp.get('align_len'),
            'gaps': hsp.get('gaps'),
        }
        print(f'    命中: {d.get("accession")} id={hsp.get("identity")}/{hsp.get("align_len")}')
    return mapping


def main():
    biosnap = sys.argv[1] if len(sys.argv) > 1 else './DataSets/BIOSNAP.txt'
    seqs = load_noprotid_seqs(biosnap)
    print(f'NO-PROT-ID 唯一序列数: {len(seqs)}')

    mapping = {}
    nohit = []  # (原始索引, 序列)
    for i, seq in enumerate(seqs):
        hits = None
        for attempt in range(2):
            try:
                hits = ebi_md5_search(seq)
                if hits:
                    break
            except Exception as e:
                print(f'  EBI retry {attempt}: {str(e)[:80]}')
            time.sleep(5)
        if hits:
            h0 = hits[0]
            extra = ''
            if len(hits) > 1:
                extra = f'（另有 {len(hits)-1} 个其他物种/条目: ' + ', '.join(h['accession'] for h in hits[1:4]) + '）'
            mapping[str(i)] = {
                'status': 'hit', 'method': 'ebi', 'seq': seq,
                'accession': h0['accession'], 'name': h0['name'],
                'organism': h0['organism'], 'reviewed': h0['reviewed'],
            }
            print(f'{i}: EBI hit {h0["accession"]} {h0["name"]} ({h0["organism"]}, reviewed={h0["reviewed"]}) {extra}')
        else:
            nohit.append((i, seq))
            print(f'{i}: EBI no hit')
        time.sleep(1)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noprotid_mapping.json')

    def save_mapping():
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print('saved:', out)

    # EBI 阶段结束后立即存盘：即使 BLAST 卡住被中断，EBI 命中结果也不丢失
    save_mapping()

    if nohit:
        print(f'\nEBI 未命中 {len(nohit)} 条，转 NCBI BLAST（每条限时 60 秒）...')
        bm = ncbi_blast([s for _, s in nohit], per_seq_timeout=60)
        for j, (orig_i, seq) in enumerate(nohit):
            v = bm.get(str(j))
            if v and v.get('status') == 'hit':
                mapping[str(orig_i)] = {
                    'status': 'hit', 'method': 'ncbi_blast', 'seq': seq,
                    'accession': v.get('accession'), 'title': v.get('title'),
                    'identity': v.get('identity'), 'align_len': v.get('align_len'),
                    'gaps': v.get('gaps'),
                }
                print(f'{orig_i}: NCBI hit {v.get("accession")} id={v.get("identity")}/{v.get("align_len")}')
            elif v and v.get('status') == 'timeout':
                mapping[str(orig_i)] = {'status': 'timeout', 'seq': seq, 'rid': v.get('rid')}
                print(f'{orig_i}: BLAST 超时（RID={v.get("rid")}，可稍后手动查；否则按删除处理）')
            else:
                mapping[str(orig_i)] = {'status': 'no_hit', 'seq': seq}
                print(f'{orig_i}: NO HIT (建议删除)')
    else:
        print('\n全部命中，无需 BLAST')

    save_mapping()
    hit_n = sum(1 for v in mapping.values() if v.get('status') == 'hit')
    print(f'\n汇总: 命中 {hit_n}/{len(seqs)}，未命中 {len(seqs) - hit_n}')
    print('下一步: 用命中的 accession 替换 BIOSNAP.txt 中的 NO-PROT-ID，'
          '未命中的行用 clean_datasets.py 移除')


if __name__ == '__main__':
    main()

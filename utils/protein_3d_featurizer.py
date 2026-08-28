# -*- coding:utf-8 -*-
"""
蛋白质3D特征提取器
基于PDB文件提取蛋白质的3D结构特征，转换为PyTorch Geometric格式
"""

import os
import sys
import contextlib
from functools import partial
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, coalesce
from typing import Optional, Union
import pandas as pd
from pathlib import Path
from biopandas.pdb import PandasPdb

# 抑制graphein的stdout输出
@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    yield
    sys.stdout = original_stdout

with suppress_stdout():
    from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_k_nn_edges, add_peptide_bonds
    from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor
    from graphein.protein.features.sequence.embeddings import compute_esm_embedding
    from graphein.protein.features.sequence.utils import subset_by_node_feature_value
    from graphein.protein.graphs import (
        initialise_graph_with_metadata, 
        add_nodes_to_graph,
        annotate_node_metadata, 
        annotate_graph_metadata, 
        annotate_edge_metadata, 
        compute_edges
    )
    from graphein.protein.graphs import (
        process_dataframe, 
        deprotonate_structure, 
        convert_structure_to_centroids, 
        remove_insertions
    )

import networkx as nx


def general_amino_acid_property(n: str, d: dict, return_array: bool = False):
    """计算氨基酸的通用属性特征"""
    def onehot(idx, size):
        onehot = torch.zeros(size)
        onehot[idx] = 1
        return np.array(onehot)

    residue_charge = {'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80, 
                     'THR': -0.80, 'TYR': -0.80, 'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37, 
                     'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37, 'PRO': 0.0, 
                     'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65}

    residue_polarity = {'CYS': 'polar', 'HIS': 'polar', 'ASN': 'polar', 'GLN': 'polar', 
                       'SER': 'polar', 'THR': 'polar', 'TYR': 'polar', 'TRP': 'polar',
                       'ALA': 'apolar', 'PHE': 'apolar', 'GLY': 'apolar', 'ILE': 'apolar', 
                       'VAL': 'apolar', 'MET': 'apolar', 'PRO': 'apolar', 'LEU': 'apolar',
                       'GLU': 'neg_charged', 'ASP': 'neg_charged', 'LYS': 'pos_charged', 
                       'ARG': 'pos_charged'}

    polarity_encoding = {'apolar': 0, 'polar': 1, 'neg_charged': 2, 'pos_charged': 3}

    amino_acid = d.get("residue_name", "UNK")
    features = []
    features.append(residue_charge.get(amino_acid, 0.0))
    polarity = residue_polarity.get(amino_acid, 'apolar')
    encoded_polarity = onehot(polarity_encoding[polarity], len(polarity_encoding))
    features.extend(encoded_polarity)
    
    if return_array:
        features = np.array(features)
    
    d["amino_acid_property"] = features
    return features


def compute_esm_embedding_long(sequence, representation="residue", model_name="esm1b_t33_650M_UR50S", output_layer=33, chunk_size=1022):
    """
    计算长序列的ESM嵌入（支持超过1022的序列）
    参考SP-DTI的实现
    """
    seq_length = len(sequence)
    embeddings = []
    token_dim = None
    overlap_size = 100

    num_chunks = max(1, (seq_length - overlap_size) // (chunk_size - overlap_size) + 1)

    for i in range(num_chunks):
        start_idx = i * (chunk_size - overlap_size)
        end_idx = min(start_idx + chunk_size, seq_length)

        chunk_embedding = compute_esm_embedding(sequence[start_idx:end_idx], representation, model_name, output_layer)
        chunk_embedding = chunk_embedding[0, 1:-1]  # 移除首尾token

        if token_dim is None:
            token_dim = chunk_embedding.shape[1]

        embeddings.append(chunk_embedding)

    # 合并所有chunk的嵌入
    combined_embedding = embeddings[0]
    for i in range(1, len(embeddings)):
        overlap_with_prev = overlap_size if i * (chunk_size - overlap_size) < seq_length else 0
        if overlap_with_prev > 0:
            combined_embedding[-overlap_with_prev:] = (combined_embedding[-overlap_with_prev:] + embeddings[i][:overlap_with_prev]) / 2
            combined_embedding = np.concatenate([combined_embedding, embeddings[i][overlap_with_prev:]], axis=0)
        else:
            combined_embedding = np.concatenate([combined_embedding, embeddings[i]], axis=0)

    # 添加空的首尾token（graphein会移除）
    empty_start_token = np.zeros((1, token_dim))
    empty_end_token = np.zeros((1, token_dim))
    combined_embedding = np.concatenate([empty_start_token, combined_embedding, empty_end_token], axis=0)

    combined_embedding = combined_embedding[np.newaxis, :]
    return combined_embedding


def esm_residue_embedding(G: nx.Graph, model_name: str = "esm1b_t33_650M_UR50S", output_layer: int = 33) -> nx.Graph:
    """
    为图的每个节点添加ESM嵌入特征
    参考SP-DTI的实现，确保节点顺序与序列顺序对齐
    """
    for chain in G.graph.get("chain_ids", ["A"]):  # 默认使用链A
        sequence = G.graph.get(f"sequence_{chain}", "")
        if not sequence:
            # 如果没有序列信息，尝试从图元数据获取
            sequence = G.graph.get("sequence", "")
        
        if sequence:
            embedding = compute_esm_embedding_long(
                sequence,
                representation="residue",
                model_name=model_name,
                output_layer=output_layer,
            )
            # 移除首尾token
            embedding = embedding[0, 1:-1]
            
            # 获取同一链的所有节点（按顺序）
            subgraph = subset_by_node_feature_value(G, "chain_id", chain)
            
            # 按节点顺序分配ESM嵌入（关键对齐步骤）
            for i, (n, d) in enumerate(subgraph.nodes(data=True)):
                if i < len(embedding):
                    G.nodes[n]["esm_embedding"] = embedding[i]
                else:
                    # 如果节点数超过嵌入数，使用零向量
                    G.nodes[n]["esm_embedding"] = np.zeros(embedding.shape[1])
        else:
            # 如果没有序列信息，为所有节点添加零向量
            print(f"Warning: No sequence found for chain {chain}, using zero embeddings")
            for n in G.nodes():
                if "esm_embedding" not in G.nodes[n]:
                    G.nodes[n]["esm_embedding"] = np.zeros(1280)  # 默认ESM维度
    
    return G


def read_pdb_to_dataframe(path: Optional[Union[str, os.PathLike]] = None) -> pd.DataFrame:
    """读取PDB文件到DataFrame"""
    if path is None:
        raise ValueError("PDB file path must be specified!")
    
    if isinstance(path, Path):
        path = os.fsdecode(path)
    
    if path.endswith(".pdb") or path.endswith(".pdb.gz") or path.endswith(".ent"):
        atomic_df = PandasPdb().read_pdb(path)
        return atomic_df.df['ATOM']
    else:
        raise ValueError(f"Unsupported file format: {path}")


def create_protein_3d_graph(protein_name: str, pdb_dir: str, use_esm: bool = False) -> Data:
    """
    从PDB文件创建蛋白质3D结构图（PyTorch Geometric格式）
    参考SP-DTI的实现，确保节点顺序与序列顺序对齐，并在节点级别融合ESM嵌入
    
    Args:
        protein_name: 蛋白质名称/ID
        pdb_dir: PDB文件目录
        use_esm: 是否使用ESM嵌入（需要序列信息）
    
    Returns:
        PyTorch Geometric Data对象，包含：
        - x: 节点特征 [num_nodes, feat_dim] (融合了3D结构特征和ESM嵌入)
        - edge_index: 边索引 [2, num_edges]
        - edge_attr: 边属性 [num_edges, edge_feat_dim]
        - pos: 节点3D坐标 [num_nodes, 3]
        - esm_embedding: ESM嵌入 [num_nodes, esm_dim] (如果use_esm=True)
    """
    pdb_path = os.path.join(pdb_dir, f"{protein_name}.pdb")
    
    if not os.path.exists(pdb_path):
        # 尝试其他可能的路径格式
        alt_path = os.path.join(pdb_dir, protein_name, f"{protein_name}.pdb")
        if os.path.exists(alt_path):
            pdb_path = alt_path
        else:
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # 读取PDB文件
    atomic_df = read_pdb_to_dataframe(pdb_path)
    
    # 使用graphein构建图
    with suppress_stdout():
        # 处理PDB数据
        atomic_df = process_dataframe(atomic_df)
        atomic_df = deprotonate_structure(atomic_df)
        atomic_df = remove_insertions(atomic_df)
        
        # 初始化图（使用centroid粒度，确保每个残基一个节点）
        # 注意：需要先转换为centroid格式
        atomic_df = convert_structure_to_centroids(atomic_df)
        G = initialise_graph_with_metadata(atomic_df)
        
        # 添加节点
        G = add_nodes_to_graph(G, atomic_df)
        
        # 注释图元数据（包含序列信息）
        G = annotate_graph_metadata(G, atomic_df)
        
        # 注释节点元数据（氨基酸特征等）
        G = annotate_node_metadata(G, [amino_acid_one_hot, hydrogen_bond_acceptor, hydrogen_bond_donor, general_amino_acid_property])
        
        # 如果使用ESM，添加ESM嵌入（在节点级别对齐）
        if use_esm:
            G = esm_residue_embedding(G)
        
        # 计算边（基于距离、氢键、KNN等）
        G = compute_edges(
            G,
            funcs=[
                add_peptide_bonds,
                add_hydrogen_bond_interactions,
                partial(add_k_nn_edges, k=10),  # K近邻边
            ]
        )
        
        # 注释边元数据
        G = annotate_edge_metadata(G, atomic_df)
    
    # 提取节点特征（按节点顺序，确保与序列对齐）
    node_features = []
    node_positions = []
    esm_embeddings = []
    node_mapping = {}  # 从graphein节点ID到索引的映射
    
    # 按节点顺序提取特征（确保与序列顺序一致）
    for idx, (node_id, node_data) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = idx
        
        # 氨基酸one-hot编码
        aa_onehot = amino_acid_one_hot(node_id, node_data)
        
        # 氢键供体/受体
        hb_donor = hydrogen_bond_donor(node_id, node_data)
        hb_acceptor = hydrogen_bond_acceptor(node_id, node_data)
        
        # 氨基酸属性
        aa_property = general_amino_acid_property(node_id, node_data)
        
        # 组合3D结构特征
        feat_3d = np.concatenate([
            aa_onehot,
            [hb_donor, hb_acceptor],
            aa_property
        ])
        node_features.append(feat_3d)
        
        # 提取ESM嵌入（如果存在）
        if use_esm and 'esm_embedding' in node_data:
            esm_emb = node_data['esm_embedding']
            if isinstance(esm_emb, torch.Tensor):
                esm_emb = esm_emb.cpu().numpy()
            esm_embeddings.append(esm_emb)
        
        # 提取3D坐标（使用CA原子坐标，如果没有则使用质心）
        if 'coords' in node_data:
            pos = node_data['coords']
        elif 'ca_coords' in node_data:
            pos = node_data['ca_coords']
        elif 'centroid' in node_data:
            pos = node_data['centroid']
        else:
            # 使用残基质心
            pos = np.array([0.0, 0.0, 0.0])  # 默认值
        
        node_positions.append(pos)
    
    node_features = np.array(node_features)
    node_positions = np.array(node_positions)
    
    # 如果使用ESM，在节点级别融合ESM嵌入和3D结构特征
    if use_esm and len(esm_embeddings) > 0:
        esm_embeddings = np.array(esm_embeddings)
        # 确保ESM嵌入数量与节点数量匹配
        if len(esm_embeddings) == len(node_features):
            # 节点级别融合：将ESM嵌入作为额外特征存储
            # 注意：这里不直接拼接，而是在模型中融合，以保持灵活性
            pass
        else:
            print(f"Warning: ESM embedding count ({len(esm_embeddings)}) != node count ({len(node_features)})")
            # 如果数量不匹配，使用零向量填充
            if len(esm_embeddings) < len(node_features):
                esm_dim = esm_embeddings.shape[1] if len(esm_embeddings) > 0 else 1280
                padding = np.zeros((len(node_features) - len(esm_embeddings), esm_dim))
                esm_embeddings = np.concatenate([esm_embeddings, padding], axis=0)
            else:
                esm_embeddings = esm_embeddings[:len(node_features)]
    else:
        esm_embeddings = None
    
    # 提取边信息
    edge_list = []
    edge_attrs = []
    
    for u, v, edge_data in G.edges(data=True):
        if u in node_mapping and v in node_mapping:
            edge_list.append([node_mapping[u], node_mapping[v]])
            
            # 边属性：距离
            if 'distance' in edge_data:
                dist = edge_data['distance']
            else:
                # 计算欧氏距离
                pos_u = node_positions[node_mapping[u]]
                pos_v = node_positions[node_mapping[v]]
                dist = np.linalg.norm(pos_u - pos_v)
            
            # 边类型编码
            edge_type = 0  # 默认
            if edge_data.get('kind') == 'peptide_bond':
                edge_type = 1
            elif edge_data.get('kind') == 'hydrogen_bond':
                edge_type = 2
            elif edge_data.get('kind') == 'k_nn':
                edge_type = 3
            
            edge_attrs.append([dist, edge_type])
    
    # 转换为torch tensor
    node_features = torch.FloatTensor(node_features)
    node_positions = torch.FloatTensor(node_positions)
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attrs)
    
    # 转换为无向图
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
    
    # 创建PyTorch Geometric Data对象
    data_dict = {
        'x': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'pos': node_positions,
        'num_nodes': len(node_features)
    }
    
    # 如果使用ESM，添加ESM嵌入
    if use_esm and esm_embeddings is not None:
        data_dict['esm_embedding'] = torch.FloatTensor(esm_embeddings)
    
    data = Data(**data_dict)
    
    return data


def protein_3d_init(protein_names: list, pdb_dir: str, use_esm: bool = False):
    """
    批量初始化蛋白质3D特征
    
    Args:
        protein_names: 蛋白质名称列表
        pdb_dir: PDB文件目录
        use_esm: 是否使用ESM嵌入
    
    Returns:
        dict: {protein_name: Data对象}
    """
    result_dict = {}
    
    from tqdm import tqdm
    
    for protein_name in tqdm(protein_names, desc="Processing 3D structures"):
        try:
            data = create_protein_3d_graph(protein_name, pdb_dir, use_esm)
            result_dict[protein_name] = data
        except Exception as e:
            print(f"Error processing {protein_name}: {e}")
            # 创建空的Data对象作为占位符
            result_dict[protein_name] = Data(
                x=torch.zeros(1, 33),  # 默认特征维度
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 2),
                pos=torch.zeros(1, 3),
                num_nodes=1
            )
    
    return result_dict


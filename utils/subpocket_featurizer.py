# -*- coding:utf-8 -*-
"""
子口袋特征化工具
从PDB文件中提取子口袋并构建图结构（使用PyTorch Geometric）
基于SP-DTI的子口袋处理逻辑，但使用PyTorch Geometric而非DGL
"""

import warnings
# 抑制RDKit的弃用警告
warnings.filterwarnings('ignore', message='.*GetValence.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
# 抑制RDKit的logging警告
logging.getLogger('rdkit').setLevel(logging.ERROR)

import torch
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# 抑制RDKit的警告输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import networkx as nx
from torch_geometric.data import Data, Batch
from utils.subpocket_batch import (
    combine_subpocket_graphs,
    split_subpocket_batch,
    validate_flat_subpocket_batch,
)
import deepchem

# 模块级 logger（可通过环境变量 SUBPOCKET_LOG_FILE 输出到文件）
logger = logging.getLogger("subpocket_featurizer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_file = os.environ.get("SUBPOCKET_LOG_FILE", None)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    # 默认也输出到控制台
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


def one_of_k_encoding(x, allowable_set):
    """One-hot编码"""
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot编码，未知值映射到最后一个元素"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# 原子特征维度（与 atom_feature() 返回值一致）:
#   元素(10) + 度(9) + H数(5) + 价态(6) + 芳香性(1) = 31
# BUGFIX: 原注释误写为 (10, 6, 5, 6, 1) = 28，导致虚拟图/真实图维度不一致；
# 实际全部旧缓存与模型 input_dim=31 一致，统一为 31
ATOM_FEATURE_DIM = 31


def make_virtual_subpocket_graph():
    """Return one explicit padding slot for a protein with no real pocket."""
    return Data(
        x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pocket_number=torch.tensor([-1.0], dtype=torch.float32),
    )


def ensure_subpocket_slots(subpocket_data, max_subpockets=30):
    """Normalize a cached protein to exactly ``max_subpockets`` graph slots.

    Existing caches are accepted as Data or Batch objects.  Missing values are
    represented explicitly as virtual graphs rather than disappearing during
    PyG collation, which preserves the outer DTI sample boundary.
    """
    if subpocket_data is None:
        graphs = []
    elif isinstance(subpocket_data, Batch):
        graphs = split_subpocket_batch(subpocket_data)
    elif isinstance(subpocket_data, Data):
        graphs = [subpocket_data]
    else:
        raise TypeError(f"Unsupported cached subpocket type: {type(subpocket_data)!r}")

    if len(graphs) > max_subpockets:
        raise RuntimeError(
            f"Cached protein has {len(graphs)} subpocket graphs; expected at most {max_subpockets}."
        )
    normalized_graphs = []
    for graph_index, graph in enumerate(graphs):
        x = getattr(graph, 'x', None)
        if x is None or x.dim() != 2 or x.size(1) != ATOM_FEATURE_DIM:
            shape = None if x is None else tuple(x.shape)
            raise ValueError(
                f"Cached subpocket graph {graph_index} has feature shape {shape}; "
                f"expected [N, {ATOM_FEATURE_DIM}]."
            )
        pocket_number = getattr(graph, 'pocket_number', None)
        # Older caches may contain a one-node, all-zero placeholder carrying a
        # positive pocket number. No valid atom can have an all-zero one-hot
        # feature vector, so normalize that legacy placeholder to masked padding.
        if (x.size(0) == 1 and torch.count_nonzero(x).item() == 0
                and pocket_number is not None):
            graph = make_virtual_subpocket_graph()
        normalized_graphs.append(graph)
    graphs = normalized_graphs + [
        make_virtual_subpocket_graph() for _ in range(max_subpockets - len(normalized_graphs))
    ]
    return combine_subpocket_graphs(graphs)


class PreparedSubpocketCache(dict):
    """Marker mapping whose entries already satisfy the fixed-slot contract."""

    def __init__(self, *args, max_subpockets=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_subpockets = int(max_subpockets)


def prepare_subpocket_cache(cache, max_subpockets=30):
    """Normalize and strictly validate every protein once on CPU.

    Reusing this mapping across train/validation/test datasets removes repeated
    cache splitting, padding, validation and disk loading from ``__getitem__``.
    """
    if isinstance(cache, PreparedSubpocketCache):
        if cache.max_subpockets != int(max_subpockets):
            raise ValueError(
                f"Prepared cache uses {cache.max_subpockets} slots, "
                f"requested {max_subpockets}."
            )
        return cache
    if not isinstance(cache, dict):
        raise TypeError(f"Expected a dict subpocket cache, got {type(cache)!r}.")

    prepared = PreparedSubpocketCache(max_subpockets=max_subpockets)
    for key, value in cache.items():
        normalized = ensure_subpocket_slots(value, max_subpockets=max_subpockets)
        validate_flat_subpocket_batch(
            normalized,
            expected_graphs=max_subpockets,
            feature_width=ATOM_FEATURE_DIM,
            validate_values=True,
        )
        prepared[key] = normalized
    return prepared


def atom_feature(atom):
    """提取原子特征"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 9, 5, 6, 1) --> total 31


def get_atom_feature(atoms):
    """批量提取原子特征"""
    H = []
    for atom in atoms:
        H.append(atom_feature(atom))
    H = np.array(H)
    return H


def parse_pdb_for_bounding_box(pdb_file):
    """从PDB文件中解析边界框"""
    with open(pdb_file, 'r') as file:
        x_coords, y_coords, z_coords = [], [], []
        for line in file:
            if line.startswith("ATOM"):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
        if len(x_coords) == 0:
            return None, None
        min_coords = np.array([min(x_coords), min(y_coords), min(z_coords)])
        max_coords = np.array([max(x_coords), max(y_coords), max(z_coords)])
    return min_coords, max_coords


def atoms_within_box(molecule, box_min, box_max):
    """获取边界框内的原子索引"""
    conformer = molecule.GetConformer()
    positions = conformer.GetPositions()
    indices = []
    for idx, pos in enumerate(positions):
        if np.all(pos >= box_min) and np.all(pos <= box_max):
            indices.append(idx)
    return indices


def networkx_to_pyg(nx_graph, node_features, pocket_number):
    """将NetworkX图转换为PyTorch Geometric Data对象"""
    # 获取节点和边
    nodes = list(nx_graph.nodes())
    edges = list(nx_graph.edges())
    
    if len(nodes) == 0:
        # An empty pocket is padding, not a real pocket position. Keeping the
        # positive pocket number would bypass the attention mask.
        return make_virtual_subpocket_graph()
    
    # 创建节点到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 构建边索引
    edge_list = []
    for u, v in edges:
        edge_list.append([node_to_idx[u], node_to_idx[v]])
        edge_list.append([node_to_idx[v], node_to_idx[u]])  # 无向图
    
    if len(edge_list) == 0:
        # 如果没有边，创建自环
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 节点特征
    node_feat_list = []
    for node in nodes:
        node_feat_list.append(node_features[node])
    
    x = torch.tensor(np.array(node_feat_list), dtype=torch.float32)
    
    # 添加自环
    num_nodes = len(nodes)
    self_loops = torch.arange(num_nodes).repeat(2, 1)
    if edge_index.size(1) > 0:
        edge_index = torch.cat([edge_index, self_loops], dim=1)
    else:
        edge_index = self_loops
    
    # 创建Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        pocket_number=torch.full((num_nodes,), pocket_number, dtype=torch.float32)
    )
    
    return data


pk = deepchem.dock.ConvexHullPocketFinder()


def process_protein_subpocket(protein_name, pdb_dir, subpocket_dir, max_subpockets=30):
    """
    处理蛋白质子口袋：从PDB文件和子口袋文件中提取子口袋并构建图
    
    Args:
        protein_name: 蛋白质名称
        pdb_dir: PDB文件目录
        subpocket_dir: 子口袋文件目录
        max_subpockets: 最大子口袋数量
        
    Returns:
        Batch对象：包含所有子口袋图的批次数据
    """
    long_data_dir = subpocket_dir + "long/"
    csv_file_path = os.path.join(long_data_dir, f"{protein_name}.csv")
    
    constructed_graphs = []
    logger.info(f"Start processing protein={protein_name} pdb_dir={pdb_dir} subpocket_dir={subpocket_dir} max_subpockets={max_subpockets}")
    
    if os.path.exists(csv_file_path):
        # 使用CSV文件（长格式）
        print(f"Processing subpockets for {protein_name} (long format)")
        df = pd.read_csv(csv_file_path)
        all_pdb = set(df["segment_folder_name"])
        complete_pdb_file_dict = {}
        am_dict = {}

        for pdb_name in all_pdb:
            pdb_path = os.path.join(pdb_dir, protein_name, f"{pdb_name}.pdb")
            try:
                complete_pdb_file_dict[pdb_name] = Chem.MolFromPDBFile(pdb_path)
                if complete_pdb_file_dict[pdb_name] is None:
                    # 尝试不使用sanitize
                    complete_pdb_file_dict[pdb_name] = Chem.MolFromPDBFile(pdb_path, sanitize=False)
                    if complete_pdb_file_dict[pdb_name] is None:
                        raise ValueError(f"Could not read molecule from {pdb_path}")
                am_dict[pdb_name] = GetAdjacencyMatrix(complete_pdb_file_dict[pdb_name])
            except Exception as e:
                # 捕获RDKit的"Bad input file"等错误
                print(f"Warning: Could not read PDB file {pdb_path}: {e}")
                # 跳过这个pdb文件，继续处理其他的
                continue

        subpocket_count = 0
        pocket_number = 1

        for index, row in df.iterrows():
            if pocket_number >= 10:
                break
            if subpocket_count >= max_subpockets:
                break
            folder = row["segment_folder_name"]
            base_dir = os.path.join(long_data_dir, protein_name, folder)
            pocket_index = row["pocket_index"]
            subpocket_number = 1
            batched_graphs = []
            
            while True:
                subpocket_file = os.path.join(base_dir, f"subpocket_{pocket_index}_{subpocket_number}.pdb")
                if subpocket_count >= max_subpockets:
                    break
                if not os.path.exists(subpocket_file):
                    break
                else:
                    print(f"  Processing {subpocket_file}")
                    # 检查是否成功加载了对应的PDB文件
                    if folder not in complete_pdb_file_dict:
                        print(f"  Warning: PDB file for {folder} was not loaded, skipping this subpocket")
                        subpocket_number += 1
                        continue
                    complete_molecule = complete_pdb_file_dict[folder]
                    am = am_dict[folder]
                    box_min, box_max = parse_pdb_for_bounding_box(subpocket_file)
                    if box_min is None:
                        subpocket_number += 1
                        continue
                    
                    atom_indices = atoms_within_box(complete_molecule, box_min, box_max)
                    if len(atom_indices) == 0:
                        subpocket_number += 1
                        continue
                    
                    ami = am[np.ix_(atom_indices, atom_indices)]
                    g = nx.from_numpy_array(ami)
                    
                    # 移除小连通分量
                    for component in list(nx.connected_components(g)):
                        if len(component) <= 10:
                            g.remove_nodes_from(component)

                    remaining_node_indices = list(g.nodes)
                    if len(remaining_node_indices) == 0:
                        subpocket_number += 1
                        continue
                    
                    selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                    H = get_atom_feature(selected_atoms)

                    # 转换为PyTorch Geometric Data
                    data = networkx_to_pyg(g, {idx: H[i] for i, idx in enumerate(remaining_node_indices)}, pocket_number)

                    if data.x.size(0) == 0:
                        subpocket_number += 1
                        continue

                    if subpocket_number <= 14:
                        constructed_graphs.append(data)
                    elif subpocket_number > 20:
                        break
                    else:
                        batched_graphs.append(data)

                    subpocket_number += 1
                    subpocket_count += 1

            # 合并批次图
            if len(batched_graphs) > 0:
                if len(batched_graphs) == 1:
                    constructed_graphs.append(batched_graphs[0])
                else:
                    # 合并多个图
                    merged_data = combine_subpocket_graphs(batched_graphs)
                    constructed_graphs.append(merged_data)

            pocket_number += 1
    else:
        # 使用标准格式（短格式）
        print(f"Processing subpockets for {protein_name} (standard format)")
        complete_pdb_file = os.path.join(pdb_dir, f"{protein_name}.pdb")
        complete_molecule = Chem.MolFromPDBFile(complete_pdb_file)
        if complete_molecule is None:
            complete_molecule = Chem.MolFromPDBFile(complete_pdb_file, sanitize=False)
            if complete_molecule is None:
                raise ValueError(f"Could not read molecule from {complete_pdb_file}")
        am = GetAdjacencyMatrix(complete_molecule)

        base_dir = os.path.join(subpocket_dir, protein_name)
        subpocket_count = 0
        pocket_number = 1
        exit = False
        
        while subpocket_count < max_subpockets and not exit:
            subpocket_number = 1
            batched_graphs = []
            
            while subpocket_number <= max_subpockets and subpocket_count < max_subpockets:
                subpocket_file = os.path.join(base_dir, f"subpocket_{pocket_number}_{subpocket_number}.pdb")
                if not os.path.exists(subpocket_file):
                    if subpocket_number == 1:
                        exit = True
                        break
                    break

                box_min, box_max = parse_pdb_for_bounding_box(subpocket_file)
                if box_min is None:
                    if subpocket_number == 1:
                        exit = True
                        break
                    subpocket_number += 1
                    continue
                
                atom_indices = atoms_within_box(complete_molecule, box_min, box_max)
                if len(atom_indices) == 0:
                    subpocket_number += 1
                    continue

                ami = am[np.ix_(atom_indices, atom_indices)]
                g = nx.from_numpy_array(ami)
                
                # 移除小连通分量
                for component in list(nx.connected_components(g)):
                    if len(component) <= 10:
                        g.remove_nodes_from(component)

                remaining_node_indices = list(g.nodes)
                if len(remaining_node_indices) == 0:
                    subpocket_number += 1
                    continue
                
                selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                H = get_atom_feature(selected_atoms)

                # 转换为PyTorch Geometric Data
                data = networkx_to_pyg(g, {idx: H[i] for i, idx in enumerate(remaining_node_indices)}, pocket_number)

                if data.x.size(0) == 0:
                    subpocket_number += 1
                    continue
                    
                if subpocket_number <= 14:
                    constructed_graphs.append(data)
                elif subpocket_number > 20:
                    break
                else:
                    batched_graphs.append(data)
                
                subpocket_count += 1
                subpocket_number += 1

            # 合并批次图
            if len(batched_graphs) > 0:
                if len(batched_graphs) == 1:
                    constructed_graphs.append(batched_graphs[0])
                else:
                    merged_data = combine_subpocket_graphs(batched_graphs)
                    constructed_graphs.append(merged_data)

            pocket_number += 1
            if pocket_number >= 10:
                break

        # 如果没有找到子口袋，尝试使用deepchem，否则回退为虚拟口袋
        # NOTE: deepchem 的某些底层实现可能在运行时触发本进程级别的段错误（C 扩展崩溃）。
        # 为了稳健性，默认**禁用** deepchem，只有当环境变量 USE_DEEPCHEM=1 时才尝试调用它。
        if len(constructed_graphs) == 0:
            pdb_file = os.path.join(pdb_dir, f"{protein_name}.pdb")
            print(f"No Subpockets Found for {protein_name}! ", end="")
            use_deepchem = os.environ.get("USE_DEEPCHEM", "0") == "1"
            if not use_deepchem:
                print("deepchem disabled (USE_DEEPCHEM!=1); creating virtual pockets")
                pockets = []
            else:
                print("attempting deepchem (USE_DEEPCHEM=1)")
                # 先做安全性检查：确保 RDKit 成功解析并有坐标/原子/构象
                m = None
                try:
                    m = Chem.MolFromPDBFile(pdb_file)
                    if m is None:
                        m = Chem.MolFromPDBFile(complete_pdb_file, sanitize=False)
                except Exception as _e:
                    print(f"  Warning: RDKit failed to parse {pdb_file}: {_e}")
                    m = None

                pockets = []
                if m is None:
                    print("  Warning: molecule is None, skipping deepchem")
                else:
                    try:
                        n_atoms = m.GetNumAtoms() if hasattr(m, "GetNumAtoms") else 0
                        conformers = m.GetConformers() if hasattr(m, "GetConformers") else []
                        if n_atoms == 0 or len(conformers) == 0:
                            print("  Warning: molecule has no atoms or conformers, skipping deepchem")
                        else:
                            # 尝试调用 deepchem（注意：此调用可能触发本进程崩溃）
                            try:
                                pockets = pk.find_pockets(pdb_file)
                            except Exception as e:
                                # 捕获 Python 异常（无法捕获 C 层段错误）
                                print(f"  deepchem.find_pockets raised exception: {e}, falling back to virtual pockets")
                                pockets = []
                    except Exception as e:
                        print(f"  Unexpected error before calling deepchem: {e}; skipping deepchem")
                        pockets = []

            # 如果 pockets 非空，按原逻辑转换为图；否则直接回退（后续会生成虚拟口袋）
            if len(pockets) > 0:
                am = GetAdjacencyMatrix(m)
                logger.info(f"{protein_name}: deepchem returned {len(pockets)} pockets")
                try:
                    c2 = m.GetConformers()[0]
                    d2 = np.array(c2.GetPositions())
                except Exception:
                    d2 = None

                for bound_box in pockets:
                    if len(constructed_graphs) >= 4:
                        break
                    if d2 is None:
                        continue
                    x_min = bound_box.x_range[0]
                    x_max = bound_box.x_range[1]
                    y_min = bound_box.y_range[0]
                    y_max = bound_box.y_range[1]
                    z_min = bound_box.z_range[0]
                    z_max = bound_box.z_range[1]

                    idxs = []
                    for idx, atom_cord in enumerate(d2):
                        if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                            idxs.append(idx)

                    if len(idxs) == 0:
                        continue

                    ami = am[np.array(idxs)[:, None], np.array(idxs)]
                    g = nx.from_numpy_array(ami)

                    for component in list(nx.connected_components(g)):
                        if len(component) <= 10:
                            g.remove_nodes_from(component)

                    remaining_node_indices = list(g.nodes)
                    if len(remaining_node_indices) == 0:
                        continue
                    
                    selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                    H = get_atom_feature(selected_atoms)

                    data = networkx_to_pyg(g, {idx: H[i] for i, idx in enumerate(remaining_node_indices)}, 1)
                    constructed_graphs.append(data)

    # 填充到最大数量
    while len(constructed_graphs) < max_subpockets:
        virtual_data = Data(
            x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            pocket_number=torch.tensor([-1.0], dtype=torch.float32)
        )
        constructed_graphs.append(virtual_data)

    # 批处理所有图
    batch_data = combine_subpocket_graphs(constructed_graphs[:max_subpockets])
    return batch_data

import torch.utils.data
from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Data
import pickle
import torch.utils.data
import os
from utils.DataSetsFunction import label_sequence, label_smiles
from utils.protein_3d_featurizer import create_protein_3d_graph
from utils.subpocket_featurizer import (
    PreparedSubpocketCache,
    ensure_subpocket_slots,
    prepare_subpocket_cache,
    process_protein_subpocket,
)

# # 模块级别的计数器，用于统计占位符创建次数和实际使用的PDB数量（减少警告输出）
# _placeholder_stats = {
#     'subpocket_data': 0,
#    'prot_3d_data': 0,
#     'total_created': 0,
#     'subpocket_used': 0,  # 实际使用的subpocket数量
#     'prot_3d_used': 0,    # 实际使用的prot_3d数量
#     'total_used': 0       # 总的使用数量
# }

# def reset_placeholder_stats():
#     """重置占位符统计计数器"""
#     global _placeholder_stats
#     _placeholder_stats = {
#         'subpocket_data': 0,
#         'prot_3d_data': 0,
#         'total_created': 0,
#         'subpocket_used': 0,
#         'prot_3d_used': 0,
#         'total_used': 0
#     }

# def get_placeholder_stats():
#     """获取占位符统计信息"""
#     return _placeholder_stats.copy()

# def print_placeholder_stats_summary():
#     """打印占位符统计汇总（在数据集加载完成后调用）"""
#     global _placeholder_stats
#     if _placeholder_stats['total_created'] > 0 or _placeholder_stats['total_used'] > 0:
#         print(f"\n[INFO] Data Statistics:")
#         print(f"  - subpocket_data used: {_placeholder_stats['subpocket_used']}, placeholders: {_placeholder_stats['subpocket_data']}")
#         print(f"  - prot_3d_data used: {_placeholder_stats['prot_3d_used']}, placeholders: {_placeholder_stats['prot_3d_data']}")
#         print(f"  - Total used: {_placeholder_stats['total_used']}, Total placeholders: {_placeholder_stats['total_created']}")
#         print()


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only existed.
        return torch.load(path, map_location="cpu")


def load_prepared_subpocket_cache(path, max_subpockets=30):
    """Load, normalize and validate one cache for sharing across datasets."""
    print(f"从预处理文件加载并预检子口袋特征: {path}")
    raw_cache = _torch_load_cpu(path)
    if not isinstance(raw_cache, dict):
        raise TypeError(f"Expected a dict subpocket cache, got {type(raw_cache)!r}.")
    valid_count = sum(1 for value in raw_cache.values() if value is not None)
    prepared = prepare_subpocket_cache(raw_cache, max_subpockets=max_subpockets)
    print(f"已加载 {len(prepared)} 个蛋白质的子口袋特征（其中 {valid_count} 个有效）")
    return prepared

class ProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, mol_obj, prot_obj, device='cpu', 
                 use_3d=False, pdb_dir=None, prot_3d_obj=None,
                 use_subpocket=False, subpocket_dir=None, pocket_num=30, 
                 subpocket_cache_path=None, subpocket_obj=None):
        super(ProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data,list):
            self.pairs = sequence_data
        else:
            raise Exception("provide list object")
        
        ## MOLECULES
        if isinstance(mol_obj, dict):
            self.mols = mol_obj
        elif isinstance(mol_obj, str):
            with open(mol_obj, 'rb') as f:
                self.mols = pickle.load(f)
        else:
            raise Exception("provide dict mol object or pickle path")


        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = prot_obj
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj)
        else:
            raise Exception("provide dict mol object or pickle path")

        ## PROTEIN 3D STRUCTURES
        self.use_3d = use_3d
        self.pdb_dir = pdb_dir
        if use_3d:
            if isinstance(prot_3d_obj, dict):
                self.prots_3d = prot_3d_obj
            elif isinstance(prot_3d_obj, str):
                self.prots_3d = torch.load(prot_3d_obj) if prot_3d_obj.endswith('.pt') else pickle.load(open(prot_3d_obj, 'rb'))
            elif pdb_dir is not None:
                # 延迟加载：在需要时从PDB文件创建
                self.prots_3d = {}
            else:
                raise Exception("For 3D features, provide prot_3d_obj dict/path or pdb_dir")
        else:
            self.prots_3d = None

        ## SUBPOCKET STRUCTURES
        self.use_subpocket = use_subpocket
        self.subpocket_dir = subpocket_dir
        self.pocket_num = pocket_num
        self.subpocket_cache_loaded = False  # 标记是否从预处理文件加载
        if use_subpocket:
            if subpocket_dir is None and subpocket_cache_path is None and subpocket_obj is None:
                raise Exception("For subpocket features, provide subpocket_dir or subpocket_cache_path")
            if subpocket_obj is not None:
                self.prots_subpocket = prepare_subpocket_cache(
                    subpocket_obj, max_subpockets=self.pocket_num
                )
                self.subpocket_cache_loaded = True
            # 优先从预处理文件加载（如果存在）
            elif subpocket_cache_path is not None and os.path.exists(subpocket_cache_path):
                self.prots_subpocket = load_prepared_subpocket_cache(
                    subpocket_cache_path, max_subpockets=self.pocket_num
                )
                self.subpocket_cache_loaded = True
            else:
                # 延迟加载：在需要时从子口袋文件创建
                self.prots_subpocket = PreparedSubpocketCache(
                    max_subpockets=self.pocket_num
                )
            self._missing_subpocket = ensure_subpocket_slots(None, self.pocket_num)
        else:
            self.prots_subpocket = None
            self._missing_subpocket = None

        self.device = device

        for _, v in self.mols.items():
            v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
            v['atom_feature'] = v['atom_feature'].float()
            adj = v['bond_feature'].long()
            mol_edge_index =  adj.nonzero(as_tuple=False).t().contiguous()
            v['atom_edge_index'] = mol_edge_index
            v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long()
            v['atom_num_nodes'] = v['atom_idx'].shape[0]
            v['smiles_x'] = torch.tensor(label_smiles(v['smiles'], MAX_SMI_LEN=200)).reshape(1, -1)


        for _, v in self.prots.items():
            v['seq_feat'] = v['seq_feat'].float()
            v['token_representation'] = v['token_representation'].float()
            v['num_nodes'] = len(v['seq'])
            v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
            v['edge_weight'] = v['edge_weight'].float()
            v['seq_x'] = torch.tensor(label_sequence(v['seq'], MAX_SEQ_LEN=1500)).reshape(1, -1)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Extract data
        items = self.pairs[idx].split()
        if len(items) < 5:
            raise ValueError(f"Malformed DTI row at index {idx}: expected at least 5 fields.")
        drug_id, protein_id = items[-5], items[-4]
        mol_key, prot_key = items[-3], items[-2]
        # prot_key is the protein sequence, but PDB files are named by protein name (items[1])
        prot_name = items[1] if len(items) >= 2 else prot_key  # protein name for PDB file lookup
        cls_y = torch.tensor(int(items[-1])).long()
            
        mol = self.mols[mol_key]
        prot = self.prots[prot_key]
        
        ## atom
        mol_x = mol['atom_idx']
        mol_x_feat = mol['atom_feature']
        mol_edge_index  = mol['atom_edge_index']
        mol_edge_attr = mol['atom_edge_attr']
        mol_num_nodes = mol['atom_num_nodes']
        mol_smiles_x = mol['smiles_x']

        ## Prot
        prot_seq = prot['seq']
        prot_seq_x = prot['seq_x']
        prot_node_aa = prot['seq_feat']
        prot_node_evo = prot['token_representation']
        prot_num_nodes = prot['num_nodes']
        prot_node_pos = prot['node_pos']
        prot_edge_index = prot['edge_index']
        prot_edge_weight = prot['edge_weight']

        ## Protein 3D structure (if enabled)
        prot_3d_data = None
        if self.use_3d:
            if prot_key in self.prots_3d:
                prot_3d_data = self.prots_3d[prot_key]
            elif self.pdb_dir is not None:
                # 延迟加载：从PDB文件创建
                # 使用蛋白质名称（prot_name）而不是序列（prot_key）来查找PDB文件
                # 启用ESM嵌入，确保节点顺序对齐（参考SP-DTI）
                try:
                    prot_3d_data = create_protein_3d_graph(prot_name, self.pdb_dir, use_esm=True)
                    self.prots_3d[prot_key] = prot_3d_data
                except Exception as e:
                    print(f"Warning: Could not load 3D structure for {prot_name}: {e}")
                    # 如果ESM失败，尝试不使用ESM
                    try:
                        prot_3d_data = create_protein_3d_graph(prot_name, self.pdb_dir, use_esm=False)
                        self.prots_3d[prot_key] = prot_3d_data
                    except Exception as e2:
                        print(f"Warning: Could not load 3D structure without ESM for {prot_name}: {e2}")
                        prot_3d_data = None
        
        # # 统计实际使用的prot_3d数量
        # if prot_3d_data is not None:
        #     _placeholder_stats['prot_3d_used'] += 1
        #     _placeholder_stats['total_used'] += 1

        ## Subpocket structure (if enabled)
        # 确保 subpocket_data 总是被设置，即使为 None，以便批处理时所有对象都有相同的属性
        subpocket_data = None
        if self.use_subpocket:
            # 优先从预处理数据中查找（先尝试 prot_name，再尝试 prot_key）
            if prot_name in self.prots_subpocket:
                subpocket_data = self.prots_subpocket[prot_name]
            elif prot_key in self.prots_subpocket:
                subpocket_data = self.prots_subpocket[prot_key]
            # 如果已从预处理文件加载，就不再尝试从 subpocket_dir 加载
            elif not self.subpocket_cache_loaded and self.subpocket_dir is not None:
                # 延迟加载：从子口袋文件创建
                # 使用蛋白质名称（prot_name）而不是序列（prot_key）来查找PDB文件
                try:
                    subpocket_data = process_protein_subpocket(prot_name, self.pdb_dir, self.subpocket_dir, self.pocket_num)
                    subpocket_data = ensure_subpocket_slots(
                        subpocket_data, self.pocket_num
                    )
                    self.prots_subpocket[prot_key] = subpocket_data
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to construct subpocket features for {prot_name}; "
                        "refusing to silently substitute a missing-data fallback."
                    ) from e
        
        # # 统计实际使用的subpocket数量
        # if subpocket_data is not None:
        #     _placeholder_stats['subpocket_used'] += 1
        #     _placeholder_stats['total_used'] += 1

        # 构建数据字典，确保所有可选属性都被设置
        if self.use_subpocket:
            # Cache entries were normalized and strictly validated once during
            # dataset construction. Reuse their flat tensors for every pair.
            if subpocket_data is None:
                subpocket_data = self._missing_subpocket

        data_dict = {
            ## MOLECULE
            'mol_x': mol_x,
            'mol_smiles_x': mol_smiles_x,
            'mol_x_feat': mol_x_feat,
            'mol_edge_index': mol_edge_index,
            'mol_edge_attr': mol_edge_attr,
            'mol_num_nodes': mol_num_nodes,
            'mol_node_levels': mol['node_levels'],
            ## PROTEIN
            'prot_node_aa': prot_node_aa,
            'prot_node_evo': prot_node_evo,
            'prot_seq_x': prot_seq_x,
            'prot_node_pos': prot_node_pos,
            'prot_seq': prot_seq,
            'prot_edge_index': prot_edge_index,
            'prot_edge_weight': prot_edge_weight,
            'prot_num_nodes': prot_num_nodes,
            ## Y output
            'cls_y': cls_y,
            ## keys
            'drug_id': drug_id,
            'protein_id': protein_id,
            'mol_key': mol_key,
            'prot_key': prot_key,
        }
        
        # 总是设置可选属性，即使为 None，以确保批处理时所有对象都有相同的属性
        data_dict['prot_3d_data'] = prot_3d_data
        data_dict['subpocket_data'] = subpocket_data
        
        # DEBUG: 验证传递给 MultiGraphData 的参数
        if 'subpocket_data' not in data_dict or 'prot_3d_data' not in data_dict:
            print(f"[ERROR __getitem__] Missing required keys in data_dict!")
            print(f"  data_dict keys: {list(data_dict.keys())}")
            print(f"  idx: {idx}, mol_key: {mol_key}, prot_key: {prot_key}")
            raise RuntimeError("Missing required keys in data_dict")
        
        out = MultiGraphData(**data_dict)
        
        # DEBUG: 验证返回对象的 _store 状态（关键检查）
        missing_keys = []
        if 'subpocket_data' not in out._store:
            missing_keys.append('subpocket_data')
        if 'prot_3d_data' not in out._store:
            missing_keys.append('prot_3d_data')
        
        if missing_keys:
            print(f"[ERROR __getitem__] Returned object missing attributes in _store!")
            print(f"  idx: {idx}, mol_key: {mol_key}, prot_key: {prot_key}")
            print(f"  missing keys: {missing_keys}")
            print(f"  out._store keys: {list(out._store.keys())}")
            print(f"  subpocket_data in data_dict: {'subpocket_data' in data_dict}, value: {data_dict.get('subpocket_data')}")
            print(f"  prot_3d_data in data_dict: {'prot_3d_data' in data_dict}, value: {data_dict.get('prot_3d_data')}")
            raise RuntimeError(f"Returned object missing attributes: {missing_keys}")
        
        return out

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    # 类级别的调试标志
    _debug_enabled = False  # 设置为 True 来启用详细调试
    
    @staticmethod
    def _is_placeholder(obj):
        """检查一个对象是否是占位符（用于表示 None 值的空 Data 对象）"""
        if obj is None:
            return True
        from torch_geometric.data import Data
        if isinstance(obj, Data) and hasattr(obj, 'num_nodes') and obj.num_nodes == 0:
            # 检查是否是占位符（只有 num_nodes=0，没有其他有意义的属性）
            if len(obj._store) <= 1:  # 只有 num_nodes
                return True
        return False
    
    @classmethod
    def validate_batch_objects(cls, data_list):
        """验证一批对象是否都包含必需的属性（用于调试）"""
        issues = []
        for i, obj in enumerate(data_list):
            if not isinstance(obj, cls):
                issues.append(f"Object {i} is not a {cls.__name__} instance")
                continue
            if 'subpocket_data' not in obj._store:
                issues.append(f"Object {i} missing 'subpocket_data' in _store")
            if 'prot_3d_data' not in obj._store:
                issues.append(f"Object {i} missing 'prot_3d_data' in _store")
            if issues:
                print(f"[DEBUG validate_batch_objects] Issues found:")
                for issue in issues:
                    print(f"  {issue}")
                print(f"  Object {i} _store keys: {list(obj._store.keys())}")
        return len(issues) == 0
    
    def __init__(self, *args, **kwargs):
        # 确保 subpocket_data 和 prot_3d_data 总是被设置，即使为 None
        subpocket_val = kwargs.pop('subpocket_data', None)
        prot_3d_val = kwargs.pop('prot_3d_data', None)
        
        # DEBUG: 检查调用来源
        if MultiGraphData._debug_enabled:
            import traceback
            caller_info = traceback.extract_stack(limit=3)[-2]
            print(f"[DEBUG MultiGraphData.__init__] Created from {caller_info.filename}:{caller_info.lineno}")
            print(f"  subpocket_data in kwargs: {'subpocket_data' in kwargs}")
            print(f"  prot_3d_data in kwargs: {'prot_3d_data' in kwargs}")
            print(f"  subpocket_val: {subpocket_val is not None}, prot_3d_val: {prot_3d_val is not None}")
        
        super(MultiGraphData, self).__init__(*args, **kwargs)
        
        # 检查 super().__init__ 后 _store 的状态
        store_keys_before = list(self._store.keys())
        
        # 强制将属性添加到 _store，即使值为 None
        # PyG 的批处理逻辑会检查 _store，所以我们必须确保这些属性存在
        # 注意：PyG 的 Data 类可能会过滤掉 None 值，所以我们需要特殊处理
        try:
            # 直接设置到 _store
            self._store['subpocket_data'] = subpocket_val
            self._store['prot_3d_data'] = prot_3d_val
            
            # 验证是否成功设置
            if 'subpocket_data' not in self._store:
                # 如果 subpocket_val 是 None，PyG 可能过滤掉了它
                # 创建一个占位符对象
                if subpocket_val is None:
                    # 创建一个最小的 Data 对象作为占位符
                    placeholder = Data(num_nodes=0)
                    self._store['subpocket_data'] = placeholder
                    # # 静默计数，不打印每个警告
                    # _placeholder_stats['subpocket_data'] += 1
                    # _placeholder_stats['total_created'] += 1
                else:
                    raise RuntimeError("Failed to set subpocket_data even though value is not None")
            
            if 'prot_3d_data' not in self._store:
                if prot_3d_val is None:
                    placeholder = Data(num_nodes=0)
                    self._store['prot_3d_data'] = placeholder
                    # # 静默计数，不打印每个警告
                    # _placeholder_stats['prot_3d_data'] += 1
                    # _placeholder_stats['total_created'] += 1
                else:
                    raise RuntimeError("Failed to set prot_3d_data even though value is not None")
                    
        except Exception as e:
            print(f"[ERROR MultiGraphData.__init__] Exception while setting attributes: {e}")
            print(f"  _store type: {type(self._store)}")
            print(f"  _store keys before setting: {store_keys_before}")
            print(f"  _store keys after setting: {list(self._store.keys())}")
            print(f"  subpocket_val: {subpocket_val}, prot_3d_val: {prot_3d_val}")
            import traceback
            traceback.print_exc()
            raise
        
        # 最终验证
        if 'subpocket_data' not in self._store or 'prot_3d_data' not in self._store:
            print(f"[ERROR MultiGraphData.__init__] Failed to set required attributes in _store!")
            print(f"  _store keys: {list(self._store.keys())}")
            print(f"  subpocket_data in _store: {'subpocket_data' in self._store}")
            print(f"  prot_3d_data in _store: {'prot_3d_data' in self._store}")
            print(f"  subpocket_val: {subpocket_val}, prot_3d_val: {prot_3d_val}")
            import traceback
            traceback.print_stack(limit=8)
            raise RuntimeError("Failed to set required attributes in _store")
    
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_3d_edge_index' and hasattr(self, 'prot_3d_data') and not self._is_placeholder(self.prot_3d_data):
            return self.prot_3d_data.num_nodes
        elif key == 'subpocket_edge_index' and hasattr(self, 'subpocket_data') and not self._is_placeholder(self.subpocket_data):
            return self.subpocket_data.num_nodes
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)
    
    def __cat_dim__(self, key, item, *args):
        # 对于 subpocket_data 和 prot_3d_data，如果是 None 或 Batch 对象，需要特殊处理
        if key in ['subpocket_data', 'prot_3d_data']:
            # DEBUG: 检查当前对象的 _store 状态（这是批处理时的关键检查点）
            if key not in self._store:
                print(f"[ERROR __cat_dim__] Key '{key}' not in self._store during batching!")
                print(f"  This indicates an object was created without the required attribute.")
                print(f"  self._store keys: {list(self._store.keys())}")
                print(f"  key: {key}, item type: {type(item)}, item value: {item}")
                print(f"  self attributes: {[k for k in dir(self) if not k.startswith('_')]}")
                import traceback
                print("  Call stack:")
                traceback.print_stack(limit=10)
                raise KeyError(f"Key '{key}' not in _store during batching. Object was likely created incorrectly.")
            
            # 如果值为 None 或是占位符对象，返回 None 表示不进行拼接（PyG 会跳过 None 值的拼接）
            if self._is_placeholder(item):
                return None
            # 如果是 Batch 对象，返回 None 表示不进行拼接（Batch 对象已经是一个批次）
            from torch_geometric.data import Batch
            if isinstance(item, Batch):
                return None
            # 如果是 Data 对象，返回 None 表示不进行拼接（这些是嵌套的图数据）
            if isinstance(item, Data):
                return None
        return super(MultiGraphData, self).__cat_dim__(key, item, *args)

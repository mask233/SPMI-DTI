# -*- coding:utf-8 -*-
"""
MIF-DTI模型定义文件
实现多模态信息融合的药物-靶点相互作用预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from layers_subpocket import SubpocketEncoder, PocketTransformer
from utils.subpocket_batch import (
    combine_subpocket_graphs,
    prepare_subpocket_runtime_batch,
    split_subpocket_batch,
)
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool
                                )
from config import hyperparameter

class MIF_conv_block(nn.Module):
    """
    MIF卷积块：基于图注意力网络(GAT)的图卷积模块
    用于处理图结构数据，包含GAT卷积、层归一化和池化操作
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        """
        初始化MIF卷积块
        
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MIF_conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 图注意力卷积层
        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)
        # 层归一化
        self.norm = LayerNorm(self.in_channels)
        # 自注意力池化层
        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        前向传播
        
        Args:
            x: 节点特征矩阵
            edge_index: 边索引
            batch: 批次索引
            edge_attr: 边属性（可选）
            
        Returns:
            x: 池化后的节点特征
            global_graph_emb: 全局图表示
        """
        # 层归一化 + ELU激活
        x = F.elu(self.norm(x, batch))
        # 图注意力卷积
        x = self.conv(x, edge_index, edge_attr)
        # 自注意力池化
        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)
        # 全局池化得到图级别表示
        global_graph_emb = global_add_pool(x, x_batch)
        return x, global_graph_emb


class MIFBlock(nn.Module):
    """
    MIF核心块：实现药物分子和蛋白质之间的多模态信息融合
    通过内部图卷积和交互图卷积捕获药物-蛋白质相互作用
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.4):
        """
        初始化MIF块
        
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MIFBlock, self).__init__()
        
        # 隐藏层维度计算
        self.hidden_channels = out_channels // (num_heads*2)
        # 药物分子内部图卷积（处理分子内原子间连接）
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        # 蛋白质内部图卷积（处理蛋白质内氨基酸间连接）
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.3)
        # 药物-蛋白质交互图卷积（处理分子与蛋白质间的相互作用）
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)
        # 层归一化
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        # 池化层
        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)
        # self.prot_pool = GATConv(out_channels, out_channels//num_heads, num_heads)

    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):
        """
        前向传播：融合药物分子和蛋白质的多模态信息
        
        Args:
            atom_x: 原子节点特征
            atom_edge_index: 原子间边索引
            bond_x: 化学键特征
            atom_batch: 原子批次索引
            aa_x: 氨基酸节点特征
            aa_edge_index: 氨基酸间边索引
            aa_edge_attr: 氨基酸边属性
            aa_batch: 氨基酸批次索引
            m2p_edge_index: 分子到蛋白质的交互边索引
            
            Returns:
            atom_x: 更新后的原子特征
            aa_x: 更新后的氨基酸特征
            drug_global_repr: 药物全局表示
            prot_global_repr: 蛋白质全局表示
        """
        # 保存残差连接的输入
        atom_x_res = atom_x
        aa_x_res = aa_x
        aa_batch_res = aa_batch

        # 药物分子内部图卷积（分子内信息）
        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)
        # 药物-蛋白质交互图卷积（从蛋白质到分子的信息流）
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        # 融合内部和交互信息
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        # 蛋白质内部图卷积（蛋白质内信息）
        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        # 药物-蛋白质交互图卷积（从分子到蛋白质的信息流）
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        # 融合内部和交互信息
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        # --- 池化操作 ---
        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        # 第1次 SAGPooling：捕获返回的新 edge_index 和 edge_attr
        aa_x_pooled, aa_edge_index_pooled, aa_edge_attr_pooled, aa_batch_pooled, _, _ = self.prot_pool(
            aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        if aa_x_pooled.size(0) > 0:
            aa_x = aa_x_pooled
            aa_edge_index = aa_edge_index_pooled
            aa_edge_attr = aa_edge_attr_pooled
            aa_batch = aa_batch_pooled

        # 第2次 SAGPooling：使用更新后的 edge_index
        aa_x, aa_edge_index, aa_edge_attr, aa_batch, _, _ = self.prot_pool(
            aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        if aa_x.size(0) == 0:
            aa_x = aa_x_res
            aa_batch = aa_batch_res
            print(f"    [MIF] SAGPool-2: ALL nodes dropped, restored to {aa_x.shape}", flush=True)
        # 残差连接 + Dropout
        if aa_x.size(0) == aa_x_res.size(0):
            aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)
        else:
            aa_x = F.dropout(F.elu(aa_x), 0.1, self.training)
        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        # 全局池化得到图级别表示
        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class MIFBlock_1D(nn.Module):
    """
    MIF一维序列块：处理药物和蛋白质的序列信息
    使用CNN提取序列特征，并通过交叉注意力机制融合药物-蛋白质序列信息
    """
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12]):
        """
        初始化MIF一维序列块
        
        Args:
            input_dim: 输入嵌入维度
            conv: 卷积输出维度基数
            drug_kernel: 药物序列CNN的卷积核大小列表
            prot_kernel: 蛋白质序列CNN的卷积核大小列表
        """
        super(MIFBlock_1D, self).__init__()
        self.attention_dim = conv * 4
        self.mix_attention_head = 5

        # 药物和蛋白质的CNN特征提取器
        self.Drug_CNNs = get_CNNs(input_dim, conv, drug_kernel)
        self.Protein_CNNs = get_CNNs(input_dim, conv, prot_kernel)

        # 多头交叉注意力层
        self.mix_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head, batch_first=True, dropout=0.3)

    def forward(self, drugembed, proteinembed):
        """
        前向传播：处理序列信息并融合
        
        Args:
            drugembed: 药物序列嵌入 [batch_size, seq_len, embed_dim]
            proteinembed: 蛋白质序列嵌入 [batch_size, seq_len, embed_dim]
            
        Returns:
            drugConv: 药物卷积特征
            proteinConv: 蛋白质卷积特征
            drugPool: 药物池化后的全局表示
            proteinPool: 蛋白质池化后的全局表示
        """
        # 转换为CNN输入格式：[batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # CNN特征提取
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # 转换回序列格式：[batch_size, embed_dim, seq_len] -> [batch_size, seq_len, embed_dim]
        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)

        # 交叉注意力机制：药物关注蛋白质，蛋白质关注药物
        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)

        # 融合原始特征和注意力特征
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        # 最大池化得到全局表示
        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool


class MIFDTI(nn.Module):
    """
    MIF-DTI主模型：多模态信息融合的药物-靶点相互作用预测模型
    融合图结构信息（分子图和蛋白质图）和序列信息（SMILES和氨基酸序列）
    """
    def __init__(self, depth=3, device='cuda:1', use_subpocket=False, pocket_num=30,
                 debug_data_validation=False, runtime_optimization=True):
        """
        初始化MIF-DTI模型
        
        Args:
            depth: 模型深度（MIF块的层数）
            device: 计算设备
        """
        super(MIFDTI, self).__init__()

        # 输入特征维度设置
        self.drug_in_channels = 43  # 药物原子特征维度
        self.prot_in_channels = 33  # 蛋白质氨基酸特征维度
        self.prot_evo_in_channels = 1280  # 蛋白质进化特征维度（ESM等预训练模型）
        self.hidden_channels = 200  # 隐藏层维度
        self.depth = depth
        self.device = device
        self.use_subpocket = use_subpocket
        self.pocket_num = pocket_num
        self.debug_data_validation = bool(debug_data_validation)
        self.runtime_optimization = bool(runtime_optimization)

        # 药物分子特征编码器
        self.atom_type_encoder = Embedding(20, self.hidden_channels)  # 原子类型嵌入
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=False)  # 原子特征MLP
        self.bond_encoder = Embedding(10, self.hidden_channels)  # 化学键类型嵌入

        # 蛋白质特征编码器
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=False)  # 进化特征MLP
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=False)  # 氨基酸特征MLP

        # 图结构编码器（多层MIF块）
        self.blocks = nn.ModuleList([MIFBlock() for _ in range(depth)])

        # 序列信息编码器
        self.drug_seq_emb = nn.Embedding(65, self.hidden_channels, padding_idx=0)  # 药物序列嵌入
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)  # 蛋白质序列嵌入
        self.blocks_1D = nn.ModuleList([MIFBlock_1D() for _ in range(depth)])  # 序列处理块

        # 最终注意力融合层（RESCAL用于多模态表示融合）
        if self.use_subpocket:
            # 使用带子口袋的融合层
            self.attn = RESCALWithSubpocket(self.hidden_channels, self.depth*2)
        else:
            # 使用标准RESCAL融合层
            self.attn = RESCAL(self.hidden_channels, self.depth*2)
        # self.attn = PoolAttention(self.hidden_channels)

        # 子口袋分支（参考SP-DTI的Subpocket-Transformer）
        if self.use_subpocket:
            self.subpocket_encoder = SubpocketEncoder(
                input_dim=31,
                hidden_dim=128,
                output_dim=self.hidden_channels,
                num_layers=3,
                pocket_num=self.pocket_num,
                pooling=False
            )
            # 注意：PocketTransformer不再需要，因为融合在RESCALWithSubpocket中完成
            # 但保留以保持向后兼容（如果其他地方还在使用）
            # self.pocket_transformer = PocketTransformer(
            #     input_dim=self.hidden_channels,
            #     emb_size=self.hidden_channels,
            #     emb_max_pos_size=self.pocket_num + 2,
            #     n_layer=2,
            #     num_attention_heads=4,
            #     device=device
            # )

        self.to(device)

    def forward(self,data):
        """
        前向传播：处理多模态输入并预测药物-靶点相互作用
        
        Args:
            data: 包含药物和蛋白质多模态信息的图数据对象
            
        Returns:
            scores: 相互作用预测分数 [batch_size, 2] (二分类：有相互作用/无相互作用)
        """
        # 提取药物分子数据
        atom_x, atom_x_feat, smiles_x, atom_edge_index, bond_x = \
            data.mol_x, data.mol_x_feat, data.mol_smiles_x, data.mol_edge_index, data.mol_edge_attr
        # 提取蛋白质数据
        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight, \
        # 批次索引
        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch
        # 原子层级（用于 m2p 边过滤）
        atom_node_level = getattr(data, 'mol_node_levels', None)

        # 药物分子特征编码
        atom_type_emb = self.atom_type_encoder(atom_x.squeeze())
        atom_feat_out = self.atom_feat_encoder(atom_x_feat)
        atom_x = atom_type_emb + atom_feat_out
        bond_x = self.bond_encoder(bond_x)
                
        # 蛋白质特征编码
        prot_aa_out = self.prot_aa(aa_x)
        prot_evo_out = self.prot_evo(aa_evo_x)
        aa_x = prot_aa_out + prot_evo_out
        # 将边权重转换为RBF嵌入
        aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)

        # 图结构编码：通过多层MIF块提取图表示
        drug_repr = []
        prot_repr = []
        for i in range(self.depth):
            edge_builder = (
                get_m2p_edge_from_batch
                if self.runtime_optimization else get_m2p_edge_from_batch_dense
            )
            m2p_edge_index = edge_builder(
                atom_batch, aa_batch, node_level=atom_node_level
            )
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            if self.debug_data_validation and (
                    not torch.isfinite(drug_global_repr).all().item()
                    or not torch.isfinite(prot_global_repr).all().item()):
                raise FloatingPointError(
                    f"Non-finite graph representation after MIF block {i}."
                )
            drug_repr.append(drug_global_repr)
            prot_repr.append(prot_global_repr)

        # 序列信息编码：处理SMILES和氨基酸序列
        atom_x_seq = self.drug_seq_emb(smiles_x)
        aa_x_seq = self.prot_seq_emb(seq_x)
        for i in range(self.depth):
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq, aa_x_seq, drug_seq_pool, prot_seq_pool = out_seq
            drug_repr.append(drug_seq_pool)
            prot_repr.append(prot_seq_pool)

        # 堆叠所有层的表示
        drug_repr = torch.stack(drug_repr, dim=-2)  # [batch_size, depth*2, hidden_dim]
        prot_repr = torch.stack(prot_repr, dim=-2)  # [batch_size, depth*2, hidden_dim]

        # 子口袋分支：如果使用子口袋，在融合层中处理
        pocket_embed = None
        if self.use_subpocket:
            batch_size = data.cls_y.size(0) if hasattr(data, 'cls_y') else (atom_batch.max().item() + 1)
            subpocket_batch = self._prepare_subpocket_batch(
                getattr(data, 'subpocket_data', None),
                device=atom_x.device,
                batch_size=batch_size,
            )
            if subpocket_batch is not None:
                try:
                    pocket_embed, pos_index = self.subpocket_encoder(subpocket_batch, batch_size=batch_size)
                    # 检查子口袋编码结果是否与 drug_repr 的 batch_size 一致
                    if pocket_embed.size(0) != drug_repr.size(0):
                        raise RuntimeError("Subpocket encoder batch size does not match DTI batch size.")
                    pocket_padding_mask = pos_index.eq(-1)
                except Exception as e:
                    raise RuntimeError("Subpocket encoding failed; refusing a silent batch fallback.") from e
            else:
                raise RuntimeError("Subpocket batch preparation returned None while use_subpocket=True.")

        # 协同注意力融合多模态表示并预测（包含子口袋特征）
        if self.use_subpocket:
            scores = self.attn(drug_repr, prot_repr, pocket_embed, pocket_padding_mask)
        else:
            scores = self.attn(drug_repr, prot_repr)

        return scores


    def _prepare_subpocket_batch(self, raw_data, device, batch_size):
        if batch_size is None or int(batch_size) < 1:
            raise ValueError("A positive DTI batch_size is required for subpocket preparation.")
        if getattr(self, 'runtime_optimization', True) and (
                isinstance(raw_data, Batch)
                or (
                    isinstance(raw_data, (list, tuple))
                    and all(isinstance(item, Batch) for item in raw_data)
                )
        ):
            return prepare_subpocket_runtime_batch(
                raw_data,
                batch_size=int(batch_size),
                graphs_per_sample=self.pocket_num,
                device=device,
                feature_width=31,
                validate_values=getattr(self, 'debug_data_validation', False),
            )
        return self._prepare_subpocket_batch_legacy(raw_data, device, batch_size)

    def _prepare_subpocket_batch_legacy(self, raw_data, device, batch_size):
        """
        将DataLoader提供的子口袋字段整理为单个Batch，便于编码器处理
        保留优化前的逐槽位拆分/校验路径，用于严格等价性对照。
        """
        if batch_size is None or int(batch_size) < 1:
            raise ValueError("A positive DTI batch_size is required for subpocket preparation.")
        batch_size = int(batch_size)
        expected_slots = batch_size * self.pocket_num

        def _virtual_slot():
            return Data(
                x=torch.zeros((1, 31), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pocket_number=torch.tensor([-1.0], dtype=torch.float32),
            )

        def _as_graphs(item):
            if item is None:
                return []
            if isinstance(item, Batch):
                return split_subpocket_batch(item)
            if isinstance(item, Data):
                return [item]
            if isinstance(item, (list, tuple)):
                graphs = []
                for child in item:
                    graphs.extend(_as_graphs(child))
                return graphs
            raise TypeError(f"Unsupported subpocket container: {type(item)!r}")

        def _validate_slot(graph, slot_index):
            if not isinstance(graph, Data):
                raise TypeError(f"Subpocket slot {slot_index} is not a PyG Data object.")
            x = getattr(graph, 'x', None)
            edge_index = getattr(graph, 'edge_index', None)
            pocket_number = getattr(graph, 'pocket_number', None)
            if x is None or x.dim() != 2 or x.size(0) < 1 or x.size(1) != 31:
                shape = None if x is None else tuple(x.shape)
                raise ValueError(f"Invalid x in subpocket slot {slot_index}: expected [N, 31], got {shape}.")
            if not torch.isfinite(x).all():
                raise ValueError(f"Non-finite subpocket features in slot {slot_index}.")
            if edge_index is None or edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(f"Invalid edge_index in subpocket slot {slot_index}.")
            if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.size(0)):
                raise ValueError(f"Out-of-range edge_index in subpocket slot {slot_index}.")
            if pocket_number is None or pocket_number.numel() != x.size(0):
                raise ValueError(f"Invalid pocket_number in subpocket slot {slot_index}.")
            if not torch.all(pocket_number == pocket_number[0]):
                raise ValueError(f"Subpocket slot {slot_index} has inconsistent pocket_number values.")

        if raw_data is None:
            slots = [_virtual_slot() for _ in range(expected_slots)]
            self._subpocket_all_virtual_batches = getattr(self, '_subpocket_all_virtual_batches', 0) + 1
        elif isinstance(raw_data, (list, tuple)) and len(raw_data) == batch_size:
            slots = []
            for sample_index, item in enumerate(raw_data):
                sample_slots = _as_graphs(item)
                if not sample_slots:
                    sample_slots = [_virtual_slot() for _ in range(self.pocket_num)]
                if len(sample_slots) != self.pocket_num:
                    raise RuntimeError(
                        f"Sample {sample_index} has {len(sample_slots)} subpocket slots; "
                        f"expected exactly {self.pocket_num}."
                    )
                slots.extend(sample_slots)
        else:
            slots = _as_graphs(raw_data)
            if len(slots) != expected_slots:
                raise RuntimeError(
                    f"Subpocket batch has {len(slots)} slots; expected {expected_slots} "
                    f"({batch_size} samples x {self.pocket_num})."
                )

        for slot_index, slot in enumerate(slots):
            _validate_slot(slot, slot_index)
        return combine_subpocket_graphs(slots, device=device)

    @staticmethod
    def _has_valid_subpocket(subpocket_batch):
        """
        判断子口袋Batch是否包含有效节点特征
        """
        if subpocket_batch is None:
            return False
        node_feat = getattr(subpocket_batch, 'x', None)
        return node_feat is not None and node_feat.numel() > 0

def get_m2p_edge_from_batch_dense(atom_batch, aa_batch, node_level=None):
    """Reference implementation retained for bitwise-equivalence checks."""
    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)
    if node_level is not None:
        mask = mask * (node_level == 1).reshape(-1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    return torch.stack([a_idx, b_idx], dim=0)


def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):
    """
    根据批次信息构建药物-蛋白质交互图的边索引
    
    Args:
        atom_batch: 原子节点的批次索引
        aa_batch: 氨基酸节点的批次索引
        node_level: 节点层级（可选，用于过滤特定层级的节点）
        
    Returns:
        edge_list: 边索引矩阵 [2, num_edges]
    """
    if atom_batch.dim() != 1 or aa_batch.dim() != 1:
        raise ValueError("atom_batch and aa_batch must be one-dimensional.")
    atom_indices = torch.arange(
        atom_batch.numel(), dtype=torch.long, device=atom_batch.device
    )
    if node_level is not None:
        if node_level.numel() != atom_batch.numel():
            raise ValueError("node_level must have one value per atom node.")
        atom_indices = atom_indices[node_level.reshape(-1).eq(1)]
    if atom_indices.numel() == 0 or aa_batch.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=atom_batch.device)

    # PyG batches store every sample's nodes contiguously. Build the same
    # atom-major Cartesian products returned by nonzero(dense_mask), without
    # allocating total_atoms x total_residues temporary storage.
    # ``minlength`` also covers a pathological sample whose protein pooling
    # produced no nodes; the dense reference would simply emit no edges for it.
    aa_counts = torch.bincount(aa_batch, minlength=atom_batch.numel())
    aa_order = torch.argsort(aa_batch, stable=True)
    atom_group = atom_batch[atom_indices]
    edge_counts = aa_counts[atom_group]
    a_idx = torch.repeat_interleave(atom_indices, edge_counts)
    if a_idx.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=atom_batch.device)

    aa_offsets = torch.cumsum(aa_counts, dim=0) - aa_counts
    segment_offsets = torch.cumsum(edge_counts, dim=0) - edge_counts
    local_aa_index = (
        torch.arange(a_idx.numel(), dtype=torch.long, device=atom_batch.device)
        - torch.repeat_interleave(segment_offsets, edge_counts)
    )
    grouped_aa_index = (
        torch.repeat_interleave(aa_offsets[atom_group], edge_counts) + local_aa_index
    )
    b_idx = aa_order[grouped_aa_index]
    return torch.stack([a_idx, b_idx], dim=0)

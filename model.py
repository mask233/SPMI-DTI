# -*- coding:utf-8 -*-
"""
SPIM-DTI模型定义文件
实现多模态信息融合的药物-靶点相互作用预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from layers_subpocket import SubpocketEncoder, PocketTransformer
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

        # 池化操作
        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        aa_x_pooled, _, _, aa_batch_pooled, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        # SAGPooling 可能在极端情况下丢弃所有节点，此时保留原始特征
        if aa_x_pooled.size(0) > 0:
            aa_x = aa_x_pooled
            aa_batch = aa_batch_pooled

        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        if aa_x.size(0) == 0:
            aa_x = aa_x_res
            aa_batch = aa_batch_res
        # 残差连接 + Dropout
        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)
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


class SPIMDTI(nn.Module):
    """
    SPIM-DTI主模型：多模态信息融合的药物-靶点相互作用预测模型
    融合图结构信息（分子图和蛋白质图）和序列信息（SMILES和氨基酸序列）
    """
    def __init__(self, depth=3, device='cuda:1', use_subpocket=False, pocket_num=30):
        """
        初始化SPIM-DTI模型
        
        Args:
            depth: 模型深度（SPIM块的层数）
            device: 计算设备
        """
        super(SPIMDTI, self).__init__()

        # 输入特征维度设置
        self.drug_in_channels = 43  # 药物原子特征维度
        self.prot_in_channels = 33  # 蛋白质氨基酸特征维度
        self.prot_evo_in_channels = 1280  # 蛋白质进化特征维度（ESM等预训练模型）
        self.hidden_channels = 200  # 隐藏层维度
        self.depth = depth
        self.device = device
        self.use_subpocket = use_subpocket
        self.pocket_num = pocket_num

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
        # 药物-蛋白质交互图边索引
        m2p_edge_index = data.m2p_edge_index

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
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
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
                device=atom_x.device
            )
            if self._has_valid_subpocket(subpocket_batch):
                pocket_embed, pos_index = self.subpocket_encoder(subpocket_batch, batch_size=batch_size)

        # 协同注意力融合多模态表示并预测（包含子口袋特征）
        if self.use_subpocket:
            scores = self.attn(drug_repr, prot_repr, pocket_embed)
        else:
            scores = self.attn(drug_repr, prot_repr)

        return scores


    def _prepare_subpocket_batch(self, raw_data, device):
        """
        将DataLoader提供的子口袋字段整理为单个Batch，便于编码器处理
        """
        if raw_data is None:
            return None

        def _is_valid_graph(graph):
            """子口袋占位符在 PyG 中表现为 Data(num_nodes=0) 且无 x 属性，需要过滤。"""
            if graph is None:
                return False
            if isinstance(graph, Data):
                node_feat = getattr(graph, 'x', None)
                return node_feat is not None and node_feat.numel() > 0
            return False

        def _batch_from_list(data_list):
            if not data_list:
                return None
            valid_list = [g for g in data_list if _is_valid_graph(g)]
            if not valid_list:
                return None
            batch = Batch.from_data_list(valid_list)
            return batch.to(device)

        if isinstance(raw_data, Batch):
            data_list = [g for g in raw_data.to_data_list() if _is_valid_graph(g)]
            return _batch_from_list(data_list)

        if isinstance(raw_data, Data):
            if not _is_valid_graph(raw_data):
                return None
            return Batch.from_data_list([raw_data]).to(device)

        if isinstance(raw_data, (list, tuple)):
            collected = []
            for item in raw_data:
                if item is None:
                    continue
                if isinstance(item, Batch):
                    collected.extend(item.to_data_list())
                elif isinstance(item, Data):
                    collected.append(item)
            return _batch_from_list(collected)

        return None

    @staticmethod
    def _has_valid_subpocket(subpocket_batch):
        """
        判断子口袋Batch是否包含有效节点特征
        """
        if subpocket_batch is None:
            return False
        node_feat = getattr(subpocket_batch, 'x', None)
        return node_feat is not None and node_feat.numel() > 0

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
    # 创建批次匹配矩阵：同一批次内的节点可以连接
    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)  # (num_a_nodes, num_b_nodes) 的bool矩阵
    # 如果指定了节点层级，只连接特定层级的节点
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    # 提取非零元素的位置作为边
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list
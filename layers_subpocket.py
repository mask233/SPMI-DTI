# -*- coding:utf-8 -*-
"""
子口袋相关神经网络层
包含子口袋编码层和子口袋注意力交互层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_max_pool, global_mean_pool
from torch_geometric.data import Batch


class SubpocketEncoder(nn.Module):
    """
    子口袋编码器：使用图神经网络处理子口袋图
    基于SP-DTI的TAG编码器，但使用PyTorch Geometric实现

    修复：原始版本5层TAGConv无残差连接和归一化，导致激活值逐层衰减至零，
    所有子口袋产生完全相同的embedding（仅来自transform的bias项）。
    现增加BatchNorm + 残差连接，默认3层，防止过度平滑。
    """
    def __init__(self, input_dim=31, hidden_dim=64, output_dim=128, num_layers=2, pocket_num=30, pooling=False):
        """
        初始化子口袋编码器

        Args:
            input_dim: 输入特征维度（原子特征，默认31）
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            num_layers: 图卷积层数（建议2-3层，过深会过度平滑）
            pocket_num: 子口袋数量
            pooling: 是否使用池化（True: 返回全局表示，False: 返回每个子口袋的表示）
        """
        super(SubpocketEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pocket_num = pocket_num
        self.pooling = pooling

        # 输入投影层：将 input_dim 映射到 hidden_dim（用于残差连接）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 图卷积层 + BatchNorm
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(TAGConv(hidden_dim, hidden_dim, K=2))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # 池化层（用于生成图级别表示）
        self.pooling_ligand = nn.Linear(hidden_dim, 1)
        self.pooling_protein = nn.Linear(hidden_dim, 1)

        # 输出投影层
        self.transform = nn.Linear(hidden_dim * 2, output_dim)  # max_pool + mean_pool
        
    def forward(self, batch_data, batch_size=None):
        """
        前向传播
        
        Args:
            batch_data: Batch对象，包含所有子口袋图
            batch_size: 可选的批次大小（如果提供，将使用此值而不是从数据中推断）
            
        Returns:
            如果pooling=True: 返回全局池化后的表示 [batch_size, output_dim]
            如果pooling=False: 返回每个子口袋的表示 [batch_size, pocket_num, output_dim] 和位置嵌入 [batch_size, pocket_num]
        """
        x = batch_data.x  # [total_nodes, input_dim]
        edge_index = batch_data.edge_index  # [2, total_edges]
        batch = batch_data.batch  # [total_nodes]
        pocket_number = batch_data.pocket_number  # [total_nodes]

        # 输入投影到 hidden_dim（使残差连接维度匹配）
        x = self.input_proj(x)

        # 图卷积：BatchNorm + ReLU + 残差连接
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + residual  # 残差连接：防止激活值逐层衰减至零
        
        # 全局池化（每个子口袋图）
        max_pooled = global_max_pool(x, batch)  # [num_graphs, hidden_dim]
        mean_pooled = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        
        # 拼接最大池化和平均池化
        protein_rep = torch.cat([max_pooled, mean_pooled], dim=1)  # [num_graphs, hidden_dim * 2]
        protein_rep = self.transform(protein_rep)  # [num_graphs, output_dim]
        
        # 获取批次大小（每个样本有pocket_num个子口袋）
        # 注意：batch_data包含所有子口袋图，每个样本有pocket_num个子口袋
        # 如果提供了batch_size参数，直接使用它
        if batch_size is not None:
            B = batch_size
        elif protein_rep.size(0) == 0:
            # 空批次处理
            B = 0
        elif hasattr(batch_data, 'ptr') and batch_data.ptr is not None:
            # 使用ptr来确定每个样本的子口袋图
            num_graphs = len(batch_data.ptr) - 1
            if num_graphs > 0:
                # 尝试根据期望的pocket_num计算批次大小
                B = num_graphs // self.pocket_num
                # 如果无法整除，说明实际的子口袋数量与期望的不匹配
                # 在这种情况下，我们需要推断实际的批次大小
                if B == 0:
                    # 如果实际的子口袋数量少于期望的，我们需要推断批次大小
                    # 假设每个样本有相同数量的子口袋（实际数量）
                    # 为了安全起见，我们假设有一个批次，但后续需要根据实际数据调整
                    # 实际上，我们应该从主数据中获取批次大小
                    # 但在这里我们只能根据子口袋数据推断
                    # 如果子口袋数量少于期望的，可能是每个样本的子口袋数量不同
                    # 最安全的方法是假设有一个批次，但我们需要知道实际的批次大小
                    # 让我们尝试从batch属性中推断
                    if hasattr(batch_data, 'batch') and batch_data.batch is not None:
                        # 从batch属性中获取唯一的批次ID数量
                        unique_batches = torch.unique(batch_data.batch)
                        B = len(unique_batches)
                    else:
                        # 如果无法推断，假设有一个批次
                        B = 1
            else:
                B = 0
        else:
            # 如果没有ptr，尝试从batch属性中获取批次大小
            if hasattr(batch_data, 'batch') and batch_data.batch is not None:
                unique_batches = torch.unique(batch_data.batch)
                B = len(unique_batches)
            else:
                # 假设所有图属于一个批次
                B = max(1, protein_rep.size(0) // self.pocket_num) if protein_rep.size(0) > 0 else 0
        
        # 重塑为 [batch_size, pocket_num, output_dim]
        if B == 0:
            # 空批次：返回空张量
            protein_rep_reshaped = torch.empty(0, self.pocket_num, protein_rep.size(1) if protein_rep.size(0) > 0 else self.output_dim, 
                                               device=protein_rep.device, dtype=protein_rep.dtype)
        elif protein_rep.size(0) == B * self.pocket_num:
            protein_rep_reshaped = protein_rep.view(B, self.pocket_num, -1)
        else:
            # 如果数量不匹配，填充或截断
            target_size = B * self.pocket_num
            if protein_rep.size(0) < target_size:
                # 填充
                padding = torch.zeros(target_size - protein_rep.size(0), protein_rep.size(1), 
                                     device=protein_rep.device, dtype=protein_rep.dtype)
                protein_rep = torch.cat([protein_rep, padding], dim=0)
            else:
                # 截断
                protein_rep = protein_rep[:target_size]
            protein_rep_reshaped = protein_rep.view(B, self.pocket_num, -1)
        
        if self.pooling:
            # 返回全局池化后的表示
            return torch.mean(protein_rep_reshaped, dim=1)  # [batch_size, output_dim]
        else:
            # 返回每个子口袋的表示和位置嵌入
            # 提取位置嵌入（每个子口袋图的第一个节点的pocket_number）
            if hasattr(batch_data, 'ptr') and batch_data.ptr is not None:
                num_nodes_per_graph = batch_data.ptr[1:] - batch_data.ptr[:-1]  # 每个图的节点数
            else:
                # 如果没有ptr，假设每个图至少有一个节点
                num_nodes_per_graph = torch.ones(protein_rep.size(0), dtype=torch.long, device=batch_data.x.device)
            
            positional_embeddings = []
            start_idx = 0
            
            # 如果num_nodes_per_graph为空，根据protein_rep的大小创建
            if len(num_nodes_per_graph) == 0:
                # 如果没有图信息，为每个子口袋创建默认位置嵌入
                num_nodes_per_graph = torch.ones(B * self.pocket_num, dtype=torch.long, device=batch_data.x.device)
            
            for num_nodes in num_nodes_per_graph:
                if num_nodes > 0 and start_idx < pocket_number.size(0):
                    pos_embed = pocket_number[start_idx]
                else:
                    pos_embed = torch.tensor(-1.0, device=pocket_number.device)
                positional_embeddings.append(pos_embed)
                start_idx += num_nodes.item() if isinstance(num_nodes, torch.Tensor) else num_nodes
            
            # 确保位置嵌入数量匹配
            if len(positional_embeddings) < B * self.pocket_num:
                # 填充
                padding = [torch.tensor(-1.0, device=pocket_number.device) 
                          for _ in range(B * self.pocket_num - len(positional_embeddings))]
                positional_embeddings.extend(padding)
            elif len(positional_embeddings) > B * self.pocket_num:
                # 截断
                positional_embeddings = positional_embeddings[:B * self.pocket_num]
            
            # 确保positional_embeddings不为空
            if len(positional_embeddings) == 0:
                # 如果仍然为空，创建默认位置嵌入
                positional_embeddings = [torch.tensor(-1.0, device=pocket_number.device) 
                                       for _ in range(B * self.pocket_num)]
            
            # 重塑位置嵌入
            positional_embeddings = torch.stack(positional_embeddings).view(B, -1)  # [batch_size, pocket_num]
            return protein_rep_reshaped, positional_embeddings


class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """位置嵌入层"""
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        # 使用max_position_size来设置位置嵌入的大小，但至少为32
        self.max_position_size = max(max_position_size, 32)
        self.position_embeddings = nn.Embedding(self.max_position_size + 1, hidden_size)  # +1 因为位置从1开始
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_input_ids, position_ids):
        """
        embedded_input_ids: [batch, seq_len, hidden_size] (已经嵌入)
        position_ids: [batch, seq_len]
        """
        position_ids = position_ids + 1
        position_ids = position_ids.long()
        # 限制位置索引在有效范围内（0到max_position_size）
        position_ids = torch.clamp(position_ids, 0, self.max_position_size)
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embedded_input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    """自注意力输出层"""
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    """注意力层"""
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    """中间层"""
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    """输出层"""
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    """Transformer编码器层"""
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    """多层Transformer编码器"""
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([layer for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class PocketTransformer(nn.Module):
    """
    子口袋Transformer：用于药物-蛋白质-子口袋的交互建模
    基于SP-DTI的PocketTransformer，但适配PyTorch Geometric
    """
    def __init__(self, input_dim=128, emb_size=128, emb_max_pos_size=545, dropout_rate=0.1, n_layer=2, 
                 intermediate_size=512, num_attention_heads=4, attention_probs_dropout=0.1, 
                 hidden_dropout_rate=0.1, device="cpu"):
        """
        初始化子口袋Transformer
        
        Args:
            input_dim: 输入特征维度
            emb_size: 嵌入维度
            emb_max_pos_size: 最大位置嵌入大小
            dropout_rate: Dropout比率
            n_layer: Transformer层数
            intermediate_size: 中间层维度
            num_attention_heads: 注意力头数
            attention_probs_dropout: 注意力Dropout比率
            hidden_dropout_rate: 隐藏层Dropout比率
            device: 计算设备
        """
        super(PocketTransformer, self).__init__()
        self.emb = Embeddings(input_dim, emb_size, emb_max_pos_size, dropout_rate)
        self.encoder = Encoder_MultipleLayers(n_layer, emb_size, intermediate_size, num_attention_heads,
                                              attention_probs_dropout, hidden_dropout_rate)
        self.output_shape = emb_size
        self.norm = nn.LayerNorm(emb_size)
        self.device = device
        # decoder的输入维度应该是 emb_size * 2（药物+蛋白质）
        decoder_input_dim = emb_size * 2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )

    def forward(self, pocket_data, drug_protein_embed):
        """
        前向传播
        
        Args:
            pocket_data: 元组 (pocket_embed, pos_index)
                - pocket_embed: 子口袋嵌入 [batch_size, pocket_num, emb_size]
                - pos_index: 位置索引 [batch_size, pocket_num]
            drug_protein_embed: 药物-蛋白质融合嵌入 [batch_size, embed_dim * 2]
                - 前一半是药物嵌入，后一半是蛋白质嵌入
                
        Returns:
            output: 预测分数 [batch_size, 2]
        """
        pocket, pos_index = pocket_data

        # 分离药物和蛋白质嵌入
        drug_embed = drug_protein_embed[:, :drug_protein_embed.shape[1]//2]  # [batch_size, embed_dim]
        protein_embed = drug_protein_embed[:, drug_protein_embed.shape[1]//2:]  # [batch_size, embed_dim]

        # 创建药物和蛋白质的位置嵌入（固定为0）
        pos_drug_protein = torch.zeros(drug_embed.shape[0], 2, device=pocket.device)
        pos_index = torch.cat([pos_drug_protein, pos_index], dim=1)  # [batch_size, 2 + pocket_num]

        # 拼接输入：[batch_size, 2 + pocket_num, emb_size]
        # 第一个位置是药物，第二个位置是蛋白质，后面是子口袋
        stacked_input = torch.cat([
            drug_embed.unsqueeze(1),  # [batch_size, 1, embed_dim]
            protein_embed.unsqueeze(1),  # [batch_size, 1, embed_dim]
            pocket  # [batch_size, pocket_num, emb_size]
        ], dim=1)  # [batch_size, 2 + pocket_num, emb_size]

        stacked_input = self.norm(stacked_input)

        # 创建注意力掩码（-1表示无效位置）
        attention_mask = (pos_index != -1).float()  # [batch_size, 2 + pocket_num]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, 2 + pocket_num]
        ex_e_mask = (1.0 - attention_mask) * -10000.0

        # 位置嵌入
        emb = self.emb(stacked_input, pos_index)  # [batch_size, 2 + pocket_num, emb_size]
        
        # Transformer编码
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())  # [batch_size, 2 + pocket_num, emb_size]

        # 残差连接
        encoded_layers += stacked_input

        # 提取药物和蛋白质嵌入
        drug_embed = encoded_layers[:, 0, :]  # [batch_size, emb_size]
        protein_embed = encoded_layers[:, 1, :]  # [batch_size, emb_size]

        # 拼接并解码
        concatenated_output = torch.cat([drug_embed, protein_embed], dim=1)  # [batch_size, emb_size * 2]
        concatenated_output = self.decoder(concatenated_output)  # [batch_size, 2]

        return concatenated_output


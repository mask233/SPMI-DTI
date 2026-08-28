"""
神经网络层定义文件
包含MLP、注意力机制、RESCAL等基础层组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    多层感知机（MLP）
    支持可选的输入/输出层归一化
    """
    def __init__(self, dims, out_norm=False, in_norm=False, bias=True):
        """
        初始化MLP
        
        Args:
            dims: 各层维度列表，如[input_dim, hidden_dim1, hidden_dim2, output_dim]
            out_norm: 是否在输出层使用层归一化
            in_norm: 是否在输入层使用层归一化
            bias: 是否使用偏置
        """
        super().__init__()
        # 创建全连接层列表
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        # 可选的层归一化
        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        """重置所有参数"""
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            y: 输出特征
        """
        y = x
        # 输入层归一化
        if self.in_norm:
            y = self.in_ln(y)

        # 隐藏层：线性变换 + ReLU激活
        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        # 输出层：仅线性变换
        y = self.FC_layers[-1](y)

        # 输出层归一化
        if self.out_norm:
            y = self.out_ln(y)

        return y


class CoAttentionLayer(nn.Module):
    """
    协同注意力层：计算两个序列之间的注意力分数
    用于捕获药物和蛋白质表示之间的交互关系
    """
    def __init__(self, n_features):
        """
        初始化协同注意力层
        
        Args:
            n_features: 特征维度
        """
        super().__init__()
        self.n_features = n_features
        # Query和Key的投影矩阵
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        # 偏置和注意力权重参数
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        # Xavier初始化
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        """
        计算协同注意力分数
        
        Args:
            receiver: 接收方序列 [batch_size, seq_len, n_features]
            attendant: 关注方序列 [batch_size, seq_len, n_features]
            
        Returns:
            attentions: 注意力分数矩阵 [batch_size, seq_len_receiver, seq_len_attendant]
        """
        # 计算Key和Query
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q

        # 计算注意力激活值
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        # 通过tanh和线性变换得到注意力分数
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions
    

class RESCAL(nn.Module):
    """
    RESCAL注意力融合层：用于融合多模态表示
    结合协同注意力和双线性交互进行最终预测
    """
    def __init__(self, n_features, depth):
        """
        初始化RESCAL层
        
        Args:
            n_features: 特征维度
            depth: 表示深度（模态数量）
        """
        super().__init__()
        self.n_features = n_features
        # 协同注意力层
        self.co_attn = CoAttentionLayer(n_features)
        # 最终预测MLP
        self.mlp = nn.Sequential(
            nn.Linear(depth*depth, 2)  # 输出二分类分数
        )

    def forward(self, heads, tails):
        """
        前向传播：融合多模态表示并预测
        
        Args:
            heads: 药物表示 [batch_size, depth, n_features]
            tails: 蛋白质表示 [batch_size, depth, n_features]
            
        Returns:
            scores: 预测分数 [batch_size, 2]
        """
        # 计算协同注意力分数
        alpha_scores = self.co_attn(heads, tails)
        # L2归一化
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        # 双线性交互（矩阵乘法）
        scores = (heads @ tails.transpose(-2, -1))
        # 应用注意力权重
        scores *= alpha_scores
        # 展平并通过MLP得到最终预测
        scores = self.mlp(scores.reshape(scores.shape[0], -1))
        return scores
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class RESCALWithSubpocket(nn.Module):
    """
    带子口袋的RESCAL融合层：实现三阶段协同注意力
    1. 蛋白质-子口袋协同注意力
    2. 药物-子口袋协同注意力
    3. 增强特征融合
    """
    def __init__(self, n_features, depth, num_heads=4, dropout=0.3):
        """
        初始化带子口袋的RESCAL层
        
        Args:
            n_features: 特征维度
            depth: 表示深度（模态数量）
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        
        # 阶段1：蛋白质-子口袋协同注意力
        self.prot_pocket_attn = nn.MultiheadAttention(
            n_features, num_heads, batch_first=True, dropout=dropout
        )
        self.prot_norm1 = nn.LayerNorm(n_features)
        
        # 阶段2：药物-子口袋协同注意力
        self.drug_pocket_attn = nn.MultiheadAttention(
            n_features, num_heads, batch_first=True, dropout=dropout
        )
        self.drug_norm1 = nn.LayerNorm(n_features)
        
        # 阶段3：增强特征融合（使用RESCAL机制）
        self.co_attn = CoAttentionLayer(n_features)
        self.prot_norm2 = nn.LayerNorm(n_features)
        self.drug_norm2 = nn.LayerNorm(n_features)
        
        # 最终预测MLP
        self.mlp = nn.Sequential(
            nn.Linear(depth*depth, 2)  # 输出二分类分数
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _base_scores(self, drug_repr, prot_repr):
        alpha_scores = self.co_attn(drug_repr, prot_repr)
        drug_repr = F.normalize(drug_repr, dim=-1)
        prot_repr = F.normalize(prot_repr, dim=-1)
        scores = drug_repr @ prot_repr.transpose(-2, -1)
        scores = scores * alpha_scores
        return self.mlp(scores.reshape(scores.shape[0], -1))

    def forward(self, drug_repr, prot_repr, pocket_embed=None, pocket_padding_mask=None):
        """
        前向传播：三阶段协同注意力融合
        
        Args:
            drug_repr: 药物表示 [batch_size, depth, n_features]
            prot_repr: 蛋白质表示 [batch_size, depth, n_features]
            pocket_embed: 子口袋表示 [batch_size, pocket_num, n_features] (可选)
            
        Returns:
            scores: 预测分数 [batch_size, 2]
        """
        # 如果没有子口袋特征，退回到标准RESCAL
        if pocket_embed is None:
            return self._base_scores(drug_repr, prot_repr)


        # 阶段1：蛋白质-子口袋协同注意力
        # 蛋白质作为query，子口袋作为key和value
        if pocket_padding_mask is None:
            pocket_padding_mask = torch.zeros(
                pocket_embed.shape[:2], dtype=torch.bool, device=pocket_embed.device
            )
        if pocket_padding_mask.shape != pocket_embed.shape[:2]:
            raise ValueError("pocket_padding_mask must have shape [batch_size, pocket_num].")
        if pocket_embed.size(0) != drug_repr.size(0):
            raise ValueError("Subpocket batch size does not match drug/protein representations.")

        # All-masked key/value rows make MultiheadAttention produce NaN.  They
        # deliberately use the original no-subpocket path, per sample.
        valid_rows = ~pocket_padding_mask.all(dim=1)
        base_scores = self._base_scores(drug_repr, prot_repr)
        if not valid_rows.any():
            return base_scores
        row_index = valid_rows.nonzero(as_tuple=True)[0]
        drug_repr = drug_repr[row_index]
        prot_repr = prot_repr[row_index]
        pocket_padding_mask = pocket_padding_mask[row_index]
        pocket_embed = pocket_embed[row_index].masked_fill(
            pocket_padding_mask.unsqueeze(-1), 0.0
        )

        prot_enhanced, _ = self.prot_pocket_attn(
            prot_repr, pocket_embed, pocket_embed,
            key_padding_mask=pocket_padding_mask,
        )
        prot_enhanced = self.prot_norm1(prot_repr + self.dropout(prot_enhanced))

        # 阶段2：药物-子口袋协同注意力
        # 药物作为query，子口袋作为key和value
        drug_enhanced, _ = self.drug_pocket_attn(
            drug_repr, pocket_embed, pocket_embed,
            key_padding_mask=pocket_padding_mask,
        )
        drug_enhanced = self.drug_norm1(drug_repr + self.dropout(drug_enhanced))
        
        # 阶段3：增强特征融合（使用RESCAL机制）
        # 计算协同注意力分数
        alpha_scores = self.co_attn(drug_enhanced, prot_enhanced)
        
        # L2归一化
        drug_enhanced = F.normalize(self.drug_norm2(drug_enhanced), dim=-1)
        prot_enhanced = F.normalize(self.prot_norm2(prot_enhanced), dim=-1)
        
        # 双线性交互（矩阵乘法）
        scores = (drug_enhanced @ prot_enhanced.transpose(-2, -1))
        
        # 应用注意力权重
        scores *= alpha_scores
        
        # 展平并通过MLP得到最终预测
        scores = self.mlp(scores.reshape(scores.shape[0], -1))
        
        return base_scores.index_copy(0, row_index, scores)
    
class PoolAttention(nn.Module):
    """
    池化注意力层：利用多头注意力进行多模态融合
    用于`with-attn`变体的关键组件
    """
    def __init__(self, n_features, num_neads=4):
        """
        初始化池化注意力层
        
        Args:
            n_features: 特征维度
            num_neads: 注意力头数
        """
        super().__init__()
        # 多头注意力层
        self.attn = nn.MultiheadAttention(n_features, num_heads=num_neads, batch_first=True)
        # 层归一化
        self.drug_norm = nn.LayerNorm(n_features)
        self.prot_norm = nn.LayerNorm(n_features)
        # 预测MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_features*2, n_features*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*2, n_features*1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*1, 2),  # 二分类输出
        )

    def forward(self, drug, prot):
        """
        前向传播
        
        Args:
            drug: 药物表示 [batch_size, seq_len, n_features]
            prot: 蛋白质表示 [batch_size, seq_len, n_features]
            
        Returns:
            scores: 预测分数 [batch_size, 2]
        """
        # 层归一化
        drug = self.drug_norm(drug)
        prot = self.prot_norm(prot)
        # 交叉注意力：药物关注蛋白质，蛋白质关注药物
        drug_attn = self.attn(drug, prot, prot)[0]
        prot_attn = self.attn(prot, drug, drug)[0]
        # 融合原始特征和注意力特征，然后最大池化
        drug_pool = torch.max((drug+drug_attn)/2, dim=1)[0]
        prot_pool = torch.max((prot+prot_attn)/2, dim=1)[0]
        # 拼接并通过MLP预测
        scores = self.mlp(torch.cat([drug_pool, prot_pool], dim=-1))
        return scores

class AttentionLayer(nn.Module):
    """
    注意力层：使用多头注意力机制融合药物和蛋白质表示
    """
    def __init__(self, n_features, heads=4):
        """
        初始化注意力层
        
        Args:
            n_features: 特征维度
            heads: 注意力头数
        """
        super().__init__()
        self.n_features = n_features
        self.heads = heads
        # 多头注意力层
        self.attn = nn.MultiheadAttention(self.n_features, self.heads, batch_first=True, dropout=0.3)
        # 预测MLP
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.n_features*2, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, 2)  # 二分类输出
        )

    def forward(self, drug_repr, prot_repr):
        """
        前向传播
        
        Args:
            drug_repr: 药物表示 [batch_size, seq_len, n_features]
            prot_repr: 蛋白质表示 [batch_size, seq_len, n_features]
            
        Returns:
            result: 预测分数 [batch_size, 2]
        """
        # 交叉注意力
        drug_output, _ = self.attn(drug_repr, prot_repr, prot_repr)
        prot_output, _ = self.attn(prot_repr, drug_repr, drug_repr)

        # 残差连接：融合原始特征和注意力特征
        drug_output = drug_output * 0.5 + drug_repr * 0.5
        prot_output = prot_output * 0.5 + prot_repr * 0.5

        # 最大池化得到全局表示
        drug_pool, _ = torch.max(drug_output, dim=1)
        prot_pool, _ = torch.max(prot_output, dim=1)
        # 拼接并通过MLP预测
        concat_repr = torch.cat([drug_pool, prot_pool], -1)
        result = self.mlp(concat_repr)
        return result
    

def rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    """
    径向基函数（RBF）嵌入
    将距离值转换为高维嵌入表示
    
    参考: https://github.com/jingraham/neurips19-graph-protein-design
    
    Args:
        D: 距离张量，形状为 [...dims]
        D_min: 最小距离值
        D_max: 最大距离值
        D_count: RBF中心数量
        device: 计算设备
        
    Returns:
        RBF: RBF嵌入，形状为 [...dims, D_count]
    """
    # 将超出最大值的距离截断
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    # 创建RBF中心点
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    # 计算RBF标准差
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    # 计算RBF嵌入：exp(-((D - mu) / sigma)^2)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_CNNs(input_dim, conv_dim, kernel):
    """
    构建一维CNN序列特征提取器
    使用多个不同大小的卷积核提取序列特征
    
    Args:
        input_dim: 输入特征维度
        conv_dim: 卷积输出维度基数
        kernel: 卷积核大小列表，如[4, 6, 8]
        
    Returns:
        CNN序列：包含多个卷积层和ReLU激活的Sequential模块
    """
    return nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=kernel[2]),
            nn.ReLU(),
        )

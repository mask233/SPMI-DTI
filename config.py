# -*- coding:utf-8 -*-
"""
超参数配置文件
定义模型训练和模型结构相关的超参数
"""


class hyperparameter():
    """
    超参数类：包含所有模型训练和结构相关的超参数
    """
    def __init__(self):
        # 设备配置
        self.device_id = 0  # CUDA设备ID，设置为1表示使用cuda:1
        
        # 训练相关超参数
        self.Learning_rate = 6e-5  
        self.Epoch = 200  # 训练轮数
        self.Batch_size = 64  # 批次大小
        self.Patience = 50  # 早停耐心值（验证集性能不提升的轮数）
        self.decay_interval = 10  # 学习率衰减间隔
        self.lr_decay = 0.5  # 学习率衰减系数
        self.weight_decay = 1e-4  # 权重衰减（L2正则化）
        
        # 模型结构相关超参数
        self.embed_dim = 64  # 嵌入维度
        self.protein_kernel = [4, 8, 12]  # 蛋白质序列CNN的卷积核大小列表
        self.drug_kernel = [4, 6, 8]  # 药物序列CNN的卷积核大小列表
        self.conv = 50  # 卷积输出维度基数
        self.char_dim = 64  # 字符嵌入维度
        
        # 3D特征相关超参数
        self.use_3d = True  # 是否使用3D结构特征
        # PDB文件目录路径（如果使用3D特征，需要指定）
        # 选项1: 使用完整PDB文件
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/BIOSNAP_pdb
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_pdb
        self.pdb_dir = "/home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_pdb"
        self.prot_3d_cache_path = None  # 3D特征缓存路径（可选，用于加速加载）
        
        # 子口袋特征相关超参数
        self.use_subpocket = True  # 是否使用子口袋特征
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/BIOSNAP_subpocket/
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_subpocket/
        self.subpocket_dir = "/home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_subpocket"  # 子口袋文件目录路径
        self.pocket_num = 30  # 子口袋数量（每个蛋白质最多提取的子口袋数）
        # 子口袋特征缓存路径（可选，用于加速加载）
        # 如果为 None，会自动尝试从 "./DataSets/Preprocessed/{DATASET}-subpocket.pt" 加载
        # 如果指定路径，则使用指定路径
        self.subpocket_cache_path = None  # 例如: "./DataSets/Preprocessed/Davis-subpocket.pt"
        
        # 损失函数相关超参数
        self.loss_epsilon = 1  # PolyLoss的epsilon参数
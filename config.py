# -*- coding:utf-8 -*-
"""
超参数配置文件
定义模型训练和模型结构相关的超参数
"""

import os


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class hyperparameter():
    """
    超参数类：包含所有模型训练和结构相关的超参数
    """
    def __init__(self, dataset_name=None, data_dir=None, preprocess_dir=None):
        dataset_name = dataset_name or os.environ.get("DTI_DATASET")
        # Input layout. Explicit function/CLI values take precedence over
        # environment variables, while the historical ./DataSets layout stays
        # the default. Results and logs remain relative to the code directory.
        self.data_dir = os.path.normpath(
            data_dir or os.environ.get("DTI_DATA_DIR") or os.path.join(".", "DataSets")
        )
        self.preprocess_dir = os.path.normpath(
            preprocess_dir
            or os.environ.get("DTI_PREPROCESS_DIR")
            or os.path.join(self.data_dir, "Preprocessed")
        )
        # 设备配置
        self.device_id = 1  # CUDA设备ID，设置为1表示使用cuda:1
        
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
        # The current MIFDTI.forward does not consume prot_3d_data. Keep this
        # legacy branch disabled unless explicitly requested for diagnostics;
        # otherwise it spends substantial time/memory building unused graphs.
        self.use_3d = _env_flag("DTI_USE_LEGACY_3D", default=False)
        # PDB文件目录路径（如果使用3D特征，需要指定）
        # 选项1: 使用完整PDB文件
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/BIOSNAP_pdb
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_pdb
        pdb_root = os.environ.get("DTI_PDB_ROOT", os.path.join(self.data_dir, "pdb"))
        self.pdb_root = os.path.normpath(pdb_root)
        default_pdb_dir = os.path.join(self.pdb_root, f"{dataset_name}_pdb") if dataset_name else None
        self.pdb_dir = os.environ.get("DTI_PDB_DIR", default_pdb_dir)
        self.prot_3d_cache_path = os.environ.get("DTI_PROT_3D_CACHE")
        
        # 子口袋特征相关超参数
        self.use_subpocket = True  # 是否使用子口袋特征
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/BIOSNAP_subpocket/
        # /home/zjx/00_Code/MIF+SP/DataSets/pdb/Davis_subpocket/
        default_subpocket_dir = (
            os.path.join(self.pdb_root, f"{dataset_name}_subpocket") if dataset_name else None
        )
        self.subpocket_dir = os.environ.get("DTI_SUBPOCKET_DIR", default_subpocket_dir)
        self.pocket_num = 30  # 子口袋数量（每个蛋白质最多提取的子口袋数）
        # 子口袋特征缓存路径（可选，用于加速加载）
        # 如果为 None，训练代码会自动尝试从
        # "{preprocess_dir}/{DATASET}-subpocket.pt" 加载
        # 如果指定路径，则使用指定路径
        self.subpocket_cache_path = os.environ.get("DTI_SUBPOCKET_CACHE")
        
        # 损失函数相关超参数
        self.loss_epsilon = 1  # PolyLoss的epsilon参数

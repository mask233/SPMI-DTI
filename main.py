"""
SPIM-DTI主入口文件
用于运行多模态信息融合的药物-靶点相互作用预测模型
"""

import argparse
import torch
from RunModel import run_MIF_model,ensemble_run_MIF_model
from model import SPIMDTI
from config import hyperparameter
import setproctitle
setproctitle.setproctitle("final_BIO") 
 
# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    prog='SPIM-DTI',
    description='SPIM-DTI is model in paper: \"multimodal information fusion method for drug-target interaction prediction\"',
    epilog='Model config set by config.py')

# 添加命令行参数
parser.add_argument('dataSetName', choices=[
                    "DrugBank", "Davis", "BIOSNAP","BD2D"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['SPIM-DTI'],
                    default='SPIM-DTI', help='Which model to use, \"SPIM-DTI\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
parser.add_argument('-g', '--gpu', type=int, default=None,
                    help='cuda number (overrides config.py if specified)')

# 解析参数
args = parser.parse_args()

# 加载配置文件获取设备ID（如果命令行未指定）
hp = hyperparameter()
gpu_id = args.gpu if args.gpu is not None else hp.device_id

# 设置计算设备
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 根据模型类型运行相应的训练/测试流程
if args.model == 'SPIM-DTI':
    # 运行标准SPIM-DTI模型
    run_MIF_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=SPIMDTI, K_Fold=args.fold, LOSS='PolyLoss', device=device)
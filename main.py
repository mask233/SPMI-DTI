"""
MIF-DTI主入口文件
用于运行多模态信息融合的药物-靶点相互作用预测模型
支持多种子运行与 mean±std 统计汇总（审稿要求：多独立运行 + 显著性检验）
"""

import os
import faulthandler
import traceback
# Expensive synchronous CUDA debugging is opt-in so normal training and the
# efficiency experiments measure the real execution path.
if os.environ.get('DTI_CUDA_DEBUG', '0') == '1':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import torch
import RunModel as run_model_runtime
from RunModel import run_MIF_model, ensemble_run_MIF_model
from model import MIFDTI
from config import hyperparameter
import setproctitle
from utils.notify import NtfyNotifier

faulthandler.enable(all_threads=True)
setproctitle.setproctitle("final_BIO")

# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    prog='MIF-DTI',
    description='MIF-DTI is model in paper: \"multimodal information fusion method for drug-target interaction prediction\"',
    epilog='Model config set by config.py')

# 添加命令行参数
parser.add_argument('dataSetName', choices=[
                    "DrugBank", "Davis", "BIOSNAP", "BD2D"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['MIF-DTI', 'MIF-DTI-B'],
                    default='MIF-DTI', help='Which model to use, \"MIF-DTI\" is used by default')
# BUGFIX: seed 支持单个或逗号分隔多个值，多种子时自动汇总 mean±std
parser.add_argument('-s', '--seed', type=str, default='114514',
                    help='Set the random seed(s), comma-separated for multi-run, e.g. -s 1,2,3 (default: 114514)')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
parser.add_argument('-g', '--gpu', type=int, default=None,
                    help='cuda number (overrides config.py if specified)')
parser.add_argument('--data-dir', default=None,
                    help='directory containing <DATASET>.txt (default: DTI_DATA_DIR or ./DataSets)')
parser.add_argument('--preprocess-dir', default=None,
                    help='protein/ligand/subpocket cache directory (default: DTI_PREPROCESS_DIR or <data-dir>/Preprocessed)')
parser.add_argument('--resume', action='store_true',
                    help='resume each fold from its latest epoch state when available')
parser.add_argument('--cuda-stability', action='store_true',
                    help='synchronize CUDA batches and disable flash/memory-efficient SDP for native-crash diagnosis')
parser.add_argument('--debug-data-validation', action='store_true',
                    help='enable expensive per-batch tensor validation for diagnostics')
parser.add_argument('--legacy-runtime', action='store_true',
                    help='use the pre-optimization subpocket/m2p runtime path for rollback diagnostics')
parser.add_argument('--ntfy-topic', default=os.environ.get('NTFY_TOPIC'),
                    help='ntfy topic; omitted disables in-process notifications')
# BUGFIX: 损失函数可显式指定，默认 CE（与实际训练一致，论文公式按 CE 描述）
parser.add_argument('--loss', type=str, choices=['CE', 'PolyLoss'], default='CE',
                    help='Loss function: CE or PolyLoss (default: CE, matches paper equations)')

# 解析参数
args = parser.parse_args()


def parse_seeds(seed_str):
    """解析种子参数：支持单个整数或逗号分隔的多个整数"""
    return [int(x.strip()) for x in seed_str.split(',') if x.strip() != '']


def _metrics_text(metrics):
    return "  ".join(f"{key}: {value:.4f}" for key, value in metrics.items())


def main(notifier=None):
    if args.cuda_stability:
        run_model_runtime.enable_cuda_stability_mode()
    # 加载配置文件获取设备ID（如果命令行未指定）
    hp = hyperparameter(
        args.dataSetName, data_dir=args.data_dir, preprocess_dir=args.preprocess_dir
    )
    gpu_id = args.gpu if args.gpu is not None else hp.device_id

    # 设置计算设备
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    seeds = parse_seeds(args.seed)
    completed_metrics = {}

    if args.model == 'MIF-DTI':
        if len(seeds) == 1:
            # 单种子：保持原有行为
            metrics = run_MIF_model(
                SEED=seeds[0], DATASET=args.dataSetName,
                MODEL=MIFDTI, K_Fold=args.fold, LOSS=args.loss, device=device,
                DATA_DIR=hp.data_dir, PREPROCESS_DIR=hp.preprocess_dir,
                RESUME=args.resume, NOTIFIER=notifier,
                DEBUG_DATA_VALIDATION=args.debug_data_validation,
                RUNTIME_OPTIMIZATION=not args.legacy_runtime,
            )
            completed_metrics[seeds[0]] = metrics
        else:
            # 多种子：循环运行并汇总 mean±std
            print(f"\n[Multi-Seed] 使用 {len(seeds)} 个种子: {seeds}")
            all_metrics = {k: [] for k in ['AUC', 'PRC', 'Accuracy', 'Recall', 'Precision']}
            for s in seeds:
                print(f"\n{'='*60}\n[Multi-Seed] seed={s} 开始\n{'='*60}")
                m = run_MIF_model(SEED=s, DATASET=args.dataSetName,
                                  MODEL=MIFDTI, K_Fold=args.fold, LOSS=args.loss, device=device,
                                  DATA_DIR=hp.data_dir, PREPROCESS_DIR=hp.preprocess_dir,
                                  RESUME=args.resume, NOTIFIER=notifier,
                                  DEBUG_DATA_VALIDATION=args.debug_data_validation,
                                  RUNTIME_OPTIMIZATION=not args.legacy_runtime)
                completed_metrics[s] = m
                for k in all_metrics:
                    all_metrics[k].append(m[k])
                if notifier is not None:
                    notifier.send(
                        "DTI Seed Done",
                        f"Seed completed successfully.\ndataset: {args.dataSetName}\n"
                        f"seed: {s}\n{_metrics_text(m)}",
                    )

            # 汇总 mean±std 并写入文件
            lines = []
            lines.append(f"Multi-seed summary: {args.dataSetName}, seeds={seeds}, K_Fold={args.fold}, loss={args.loss}")
            lines.append("-" * 70)
            for k, v in all_metrics.items():
                # Seeds are independent experimental runs: report sample SD.
                mean = np.mean(v)
                std = np.std(v, ddof=1) if len(v) > 1 else 0.0
                lines.append(f"{k:>10}: {mean:.4f} ± {std:.4f}   (values: {[f'{x:.4f}' for x in v]})")
            summary = "\n".join(lines)
            print("\n" + "=" * 70)
            print(summary)
            print("=" * 70)

            summary_dir = f"./{args.dataSetName}"
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, "multi_seed_summary.txt")
            # One invocation produces one summary; do not mix stale seed runs.
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary + "\n\n")
            print(f"[Multi-Seed] 汇总已写入: {summary_path}")

    if args.model == 'MIF-DTI-B':
        # 运行集成模型（需要先运行K折训练）
        ensemble_run_MIF_model(SEED=seeds[0], DATASET=args.dataSetName,
                               K_Fold=args.fold, device=device, LOSS=args.loss,
                               DATA_DIR=hp.data_dir, PREPROCESS_DIR=hp.preprocess_dir)
    return {
        "dataset": args.dataSetName,
        "seeds": seeds,
        "model": args.model,
        "metrics": completed_metrics,
        "device": str(device),
    }


def cli():
    notifier = NtfyNotifier(args.ntfy_topic)
    if notifier.enabled:
        notifier.send(
            "DTI Training Started",
            "Training started.\n"
            f"dataset: {args.dataSetName}\n"
            f"model: {args.model}\n"
            f"seeds: {args.seed}\nfolds: {args.fold}\n"
            f"resume: {args.resume}",
        )
    try:
        result = main(notifier=notifier)
    except BaseException as exc:
        if notifier.enabled:
            notifier.send(
                "DTI Training Failed",
                "Python exception stopped training.\n"
                f"dataset: {args.dataSetName}\n"
                f"error: {type(exc).__name__}: {exc}\n"
                f"traceback:\n{traceback.format_exc()[-3500:]}",
                priority="high",
            )
        raise
    if notifier.enabled:
        metric_lines = []
        for seed, metrics in result["metrics"].items():
            metric_lines.append(f"seed {seed}: {_metrics_text(metrics)}")
        notifier.send(
            "DTI Training Done",
            "Training completed successfully.\n"
            f"dataset: {result['dataset']}\nmodel: {result['model']}\n"
            + ("\n".join(metric_lines) if metric_lines else "ensemble evaluation completed"),
        )
    return result


if __name__ == '__main__':
    cli()

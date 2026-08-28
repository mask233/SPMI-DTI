# -*- coding:utf-8 -*-
"""Evaluation helpers and reproducible, sample-level prediction export."""

import csv
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def _as_batch_list(value, expected_size, field_name):
    """Return PyG-collated string metadata as one value per sample."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        # String metadata is expected here. Failing loudly avoids silently
        # writing predictions that cannot be paired with their source sample.
        raise TypeError(f"Unsupported {field_name} metadata type: {type(value)!r}")
    if len(values) != expected_size:
        raise ValueError(
            f"{field_name} has {len(values)} entries, expected {expected_size}; "
            "prediction/sample alignment is unsafe."
        )
    return [str(item) for item in values]


def _model_pair_sha256(mol_key, prot_key):
    return hashlib.sha256(f"{mol_key}\0{prot_key}".encode("utf-8")).hexdigest()


def prediction_metrics(y_true, y_pred, y_score):
    """Compute binary-classification metrics from already aligned predictions."""
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC/AUPRC require both classes in the evaluated split.")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "PRC": float(auc(recall_curve, precision_curve)),
    }


def _probability_mean_logits(outputs):
    """Represent a probability-mean ensemble as log-probability logits."""
    probabilities = torch.stack(
        [F.softmax(output, dim=1) for output in outputs], dim=0
    ).mean(dim=0)
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


def _evaluate(MODEL, pbar, LOSS, DEVICE, mif):
    if isinstance(MODEL, list):
        if not MODEL:
            raise ValueError("Ensemble MODEL list cannot be empty.")
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()

    test_losses, y_true, y_pred, y_score, records = [], [], [], [], []
    sample_index = 0
    with torch.no_grad():
        for _, data in pbar:
            if mif:
                batch_size = data.cls_y.size(0)
                mol_keys = _as_batch_list(data.mol_key, batch_size, "mol_key")
                prot_keys = _as_batch_list(data.prot_key, batch_size, "prot_key")
                drug_ids = _as_batch_list(
                    getattr(data, "drug_id", data.mol_key), batch_size, "drug_id")
                protein_ids = _as_batch_list(
                    getattr(data, "protein_id", data.prot_key), batch_size, "protein_id")
                data = data.to(DEVICE)
                labels = data.cls_y
                if isinstance(MODEL, list):
                    logits = _probability_mean_logits([item(data) for item in MODEL])
                else:
                    logits = MODEL(data)
            else:
                compounds, proteins, labels = data
                compounds, proteins, labels = compounds.to(DEVICE), proteins.to(DEVICE), labels.to(DEVICE)
                mol_keys = [f"unavailable_mol_{sample_index + j}" for j in range(labels.size(0))]
                prot_keys = [f"unavailable_prot_{sample_index + j}" for j in range(labels.size(0))]
                drug_ids = list(mol_keys)
                protein_ids = list(prot_keys)
                if isinstance(MODEL, list):
                    logits = _probability_mean_logits(
                        [item(compounds, proteins) for item in MODEL])
                else:
                    logits = MODEL(compounds, proteins)

            loss = LOSS(logits, labels)
            labels_np = labels.detach().cpu().numpy().astype(int)
            scores_np = F.softmax(logits, dim=1).detach().cpu().numpy()
            predictions_np = np.argmax(scores_np, axis=1).astype(int)
            positive_scores = scores_np[:, 1]

            y_true.extend(labels_np.tolist())
            y_pred.extend(predictions_np.tolist())
            y_score.extend(positive_scores.tolist())
            test_losses.append(loss.item())
            for drug_id, protein_id, mol_key, prot_key, label, score, prediction in zip(
                    drug_ids, protein_ids, mol_keys, prot_keys,
                    labels_np, positive_scores, predictions_np):
                records.append({
                    "sample_index": sample_index,
                    "sample_id": _model_pair_sha256(mol_key, prot_key),
                    "external_pair_id": f"{drug_id}::{protein_id}",
                    "drug_id": drug_id,
                    "protein_id": protein_id,
                    "mol_key": mol_key,
                    "prot_key": prot_key,
                    "y_true": int(label),
                    "y_score": float(score),
                    "y_pred": int(prediction),
                })
                sample_index += 1

    metrics = prediction_metrics(y_true, y_pred, y_score)
    return y_true, y_pred, y_score, records, float(np.mean(test_losses)), metrics


# Retained for callers outside this repository that use the legacy helpers.
def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM=1):
    y_true, y_pred, _, _, loss, metrics = _evaluate(MODEL, pbar, LOSS, DEVICE, mif=False)
    return y_true, y_pred, loss, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["AUC"], metrics["PRC"]


def test_MIF_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM=1):
    y_true, y_pred, _, _, loss, metrics = _evaluate(MODEL, pbar, LOSS, DEVICE, mif=True)
    return y_true, y_pred, loss, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["AUC"], metrics["PRC"]


def save_prediction_records(records, filepath, seed=None, fold=None, split=None):
    """Overwrite a structured CSV so reruns never append stale predictions."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fields = ["sample_index", "sample_id", "external_pair_id", "drug_id", "protein_id",
              "mol_key", "prot_key", "y_true", "y_score", "y_pred", "seed", "fold", "split"]
    with open(filepath, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "seed": seed, "fold": fold, "split": split})


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE,
               dataset_class="Train", save=True, FOLD_NUM=1, MIF=False,
               seed=None, fold=None, split=None, return_predictions=False):
    """Evaluate a model and optionally return aligned sample-level predictions.

    ``FOLD_NUM`` is retained for API compatibility. List models are always
    averaged by their actual list length, which prevents a denominator mismatch.
    """
    # Keep evaluation in the main process/thread. Background prefetching around
    # PyG objects and native CUDA extensions makes fatal crashes harder to
    # localize and offers little benefit with num_workers=0.
    test_pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
    y_true, y_pred, y_score, records, loss_test, metrics = _evaluate(
        MODEL, test_pbar, LOSS, DEVICE, mif=MIF)
    split = split or dataset_class.lower()
    if save:
        fold_label = "ensemble" if isinstance(MODEL, list) else (fold if fold is not None else FOLD_NUM)
        filename = f"{DATASET}_{split}_predictions_seed_{seed}_fold_{fold_label}.csv"
        save_prediction_records(records, os.path.join(save_path, filename), seed=seed, fold=fold_label, split=split)

    results = ('{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.'
               .format(dataset_class, loss_test, metrics["Accuracy"], metrics["Precision"],
                       metrics["Recall"], metrics["AUC"], metrics["PRC"]))
    print(results)
    legacy_result = (results, metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
                     metrics["AUC"], metrics["PRC"])
    return (*legacy_result, records) if return_predictions else legacy_result

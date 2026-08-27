"""检查实体特征是否能够预测仿真标签中注入的扰动。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from afil.paths import DATA_DIR, FULL_SIMULATED_DATA_DIR


def _load_original(name: str) -> np.ndarray:
    frame = pd.read_table(DATA_DIR / f"{name}Matrix.txt", header=None)
    return frame.iloc[:, :-1].to_numpy(dtype=np.float64)


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    continuous = frame[["processing_capacity", "device_load"]].to_numpy(dtype=np.float64)
    continuous = (continuous - continuous.mean(axis=0)) / (continuous.std(axis=0) + 1e-12)
    categorical = pd.get_dummies(
        frame[["entity_type", "network_interface_type"]], dtype=float
    ).to_numpy(dtype=np.float64)
    return np.column_stack([np.ones(len(frame)), continuous, categorical])


def _service_log_ratio(original: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    valid = (original > 0) & (simulated > 0)
    ratio = np.zeros_like(original, dtype=np.float64)
    ratio[valid] = np.log(simulated[valid] / original[valid])
    counts = valid.sum(axis=0)
    return np.divide(ratio.sum(axis=0), counts, out=np.zeros(original.shape[1]), where=counts > 0)


def _fit_and_score(features: np.ndarray, target: np.ndarray, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    split = int(0.8 * len(order))
    train, test = order[:split], order[split:]
    weights = np.linalg.lstsq(features[train], target[train], rcond=None)[0]
    prediction = features[test] @ weights
    residual = target[test] - prediction
    baseline = target[test] - target[train].mean()
    return {
        "test_services": int(len(test)),
        "mae": float(np.mean(np.abs(residual))),
        "mean_baseline_mae": float(np.mean(np.abs(baseline))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum(baseline**2)),
    }


def main():
    entity = pd.read_csv(FULL_SIMULATED_DATA_DIR / "service_entity_features.csv").sort_values(
        "service_id"
    )
    matrices = np.load(FULL_SIMULATED_DATA_DIR / "qops_rt_tp_v1.npz")
    ws = pd.read_csv(DATA_DIR / "wslist.txt", skiprows=2, header=None, sep="\t")
    keep = np.where(ws.iloc[:, 1:].notna().all(axis=1).to_numpy())[0]
    features = _design_matrix(entity)
    report = {}
    for qos_type in ("rt", "tp"):
        original = _load_original(qos_type)[:, keep]
        target = _service_log_ratio(original, matrices[f"{qos_type}_ps"])
        report[qos_type] = _fit_and_score(features, target)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

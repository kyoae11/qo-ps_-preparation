"""养老医院 QoPS RT/TP 可复现仿真数据生成器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from afil.paths import DATA_DIR, FULL_SIMULATED_DATA_DIR, PILOT_DATA_DIR, PROJECT_ROOT


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_aligned_ws_dream():
    """按 AFIL 原预处理规则移除字段缺失服务，并同步 RT/TP 列。"""
    user_raw = pd.read_csv(DATA_DIR / "userlist.txt", skiprows=2, header=None, sep="\t")
    ws_raw = pd.read_csv(DATA_DIR / "wslist.txt", skiprows=2, header=None, sep="\t")
    keep = np.where(ws_raw.iloc[:, 1:].notna().all(axis=1).to_numpy())[0]
    rt = pd.read_table(DATA_DIR / "rtMatrix.txt", header=None).iloc[:, :-1].to_numpy(dtype=np.float64)[:, keep]
    tp = pd.read_table(DATA_DIR / "tpMatrix.txt", header=None).iloc[:, :-1].to_numpy(dtype=np.float64)[:, keep]
    return user_raw, ws_raw.iloc[keep].reset_index(drop=True), rt, tp, keep


def _choice(rng, probabilities: dict, size: int):
    names = list(probabilities)
    probs = np.asarray(list(probabilities.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return rng.choice(names, size=size, p=probs)


def _common_entity_columns(rng, count, prefix, entity_type, cfg, width, height):
    capacity = np.clip(
        rng.normal(cfg["processing_capacity_mean"], cfg["processing_capacity_std"], count),
        0.20,
        1.00,
    )
    return {
        "entity_id": [f"{prefix}{idx:04d}" for idx in range(count)],
        "entity_type": [entity_type] * count,
        "location_x_m": rng.uniform(0.0, width, count),
        "location_y_m": rng.uniform(0.0, height, count),
        "unit_physical_cost": rng.uniform(*cfg["unit_physical_cost"], count),
        "processing_capacity": capacity,
        "network_interface_type": _choice(rng, cfg["interface_probabilities"], count),
        "device_load": rng.beta(2.0, 3.0, count),
    }


def build_entities(config: dict):
    rng = np.random.default_rng(config["seed"])
    width = config["hospital"]["width_m"]
    height = config["hospital"]["height_m"]
    ecfg = config["entities"]

    scount = int(ecfg["static_count"])
    static_data = _common_entity_columns(
        rng, scount, "S", "static", ecfg["static"], width, height
    )
    static_data["service_radius_m"] = rng.uniform(
        *ecfg["static"]["service_radius_m"], scount
    )
    static = pd.DataFrame(static_data)

    dcount = int(ecfg["dynamic_count"])
    dynamic_data = _common_entity_columns(
        rng, dcount, "D", "dynamic", ecfg["dynamic"], width, height
    )
    dynamic_data["mobility_speed_mps"] = rng.uniform(
        *ecfg["dynamic"]["mobility_speed_mps"], dcount
    )
    beta_a, beta_b = ecfg["dynamic"]["battery_level_percent_beta"]
    dynamic_data["battery_level_percent"] = np.clip(
        100.0 * rng.beta(beta_a, beta_b, dcount), 20.0, 100.0
    )
    dynamic = pd.DataFrame(dynamic_data)

    user_locations = pd.DataFrame({
        "user_id": np.arange(339, dtype=np.int64),
        "location_x_m": rng.uniform(0.0, width, 339),
        "location_y_m": rng.uniform(0.0, height, 339),
    })
    return static, dynamic, user_locations


def build_service_mapping(n_services: int, static: pd.DataFrame, dynamic: pd.DataFrame, seed: int):
    """固定一服务一实体，并保证每个实体至少承载一个服务。"""
    rng = np.random.default_rng(seed + 17)
    entity_ids = np.concatenate([static["entity_id"].to_numpy(), dynamic["entity_id"].to_numpy()])
    if n_services < len(entity_ids):
        raise ValueError("服务数量少于实体数量，无法保证每个实体至少承载一个服务")
    services = rng.permutation(n_services)
    assignments = np.resize(rng.permutation(entity_ids), n_services)
    mapping = pd.DataFrame({"service_id": services, "entity_id": assignments})
    return mapping.sort_values("service_id").reset_index(drop=True)


def build_entity_feature_table(static, dynamic, mapping):
    entity = pd.concat([static, dynamic], ignore_index=True, sort=False)
    return mapping.merge(entity, on="entity_id", how="left", validate="many_to_one")


def perturb_matrices(rt, tp, service_features, config):
    """只注入实体结构化扰动；v1 明确不加入随机噪声。"""
    pcfg = config["perturbation"]
    interface = service_features["network_interface_type"].to_numpy()
    penalties = pcfg["interface_penalties"]
    interface_rt = np.asarray([penalties[name]["rt"] for name in interface])
    interface_tp = np.asarray([penalties[name]["tp"] for name in interface])
    capacity_penalty = 1.0 - service_features["processing_capacity"].to_numpy(dtype=np.float64)
    load = service_features["device_load"].to_numpy(dtype=np.float64)

    rw = pcfg["rt_weights"]
    tw = pcfg["tp_weights"]
    z_rt = (
        rw["capacity_penalty"] * capacity_penalty
        + rw["device_load"] * load
        + rw["network_interface"] * interface_rt
    )
    z_tp = (
        tw["capacity_penalty"] * capacity_penalty
        + tw["device_load"] * load
        + tw["network_interface"] * interface_tp
    )
    rt_multiplier = np.exp(float(pcfg["lambda_rt"]) * z_rt)
    tp_multiplier = np.exp(-float(pcfg["lambda_tp"]) * z_tp)

    rt_ps = np.where(rt > 0, rt * rt_multiplier[None, :], rt)
    tp_ps = np.where(tp > 0, tp * tp_multiplier[None, :], tp)
    factors = pd.DataFrame({
        "service_id": np.arange(len(service_features)),
        "rt_penalty": z_rt,
        "tp_penalty": z_tp,
        "rt_multiplier": rt_multiplier,
        "tp_multiplier": tp_multiplier,
    })
    return rt_ps, tp_ps, factors


def _write_tables(output_dir, static, dynamic, user_locations, mapping, service_features):
    output_dir.mkdir(parents=True, exist_ok=True)
    static.to_csv(output_dir / "static_entities.csv", index=False)
    dynamic.to_csv(output_dir / "dynamic_entities.csv", index=False)
    user_locations.to_csv(output_dir / "user_hospital_locations.csv", index=False)
    mapping.to_csv(output_dir / "service_entity_mapping.csv", index=False)
    service_features.to_csv(output_dir / "service_entity_features.csv", index=False)


def _summary(config, rt, tp, rt_ps, tp_ps, static, dynamic, mapping):
    rt_obs = rt > 0
    tp_obs = tp > 0
    return {
        "version": config["version"],
        "seed": config["seed"],
        "random_noise": config["perturbation"]["random_noise"],
        "users": int(rt.shape[0]),
        "services": int(rt.shape[1]),
        "static_entities": int(len(static)),
        "dynamic_entities": int(len(dynamic)),
        "entities_without_services": int(len(static) + len(dynamic) - mapping["entity_id"].nunique()),
        "rt_observations": int(rt_obs.sum()),
        "tp_observations": int(tp_obs.sum()),
        "rt_mean_original": float(rt[rt_obs].mean()),
        "rt_mean_simulated": float(rt_ps[rt_obs].mean()),
        "tp_mean_original": float(tp[tp_obs].mean()),
        "tp_mean_simulated": float(tp_ps[tp_obs].mean()),
        "rt_mean_relative_change": float(np.mean(rt_ps[rt_obs] / rt[rt_obs] - 1.0)),
        "tp_mean_relative_change": float(np.mean(tp_ps[tp_obs] / tp[tp_obs] - 1.0)),
    }


def generate(config_path: Path, mode: str = "all"):
    config = load_config(config_path)
    _, _, rt, tp, _ = load_aligned_ws_dream()
    static, dynamic, user_locations = build_entities(config)
    mapping = build_service_mapping(rt.shape[1], static, dynamic, config["seed"])
    service_features = build_entity_feature_table(static, dynamic, mapping)
    rt_ps, tp_ps, factors = perturb_matrices(rt, tp, service_features, config)

    if mode in {"pilot", "all"}:
        _write_tables(PILOT_DATA_DIR, static, dynamic, user_locations, mapping, service_features)
        p = config["pilot"]
        users = np.arange(int(p["user_count"]))
        services = np.arange(int(p["service_count"]))
        uu, ss = np.meshgrid(users, services, indexing="ij")
        pilot = pd.DataFrame({
            "user_id": uu.ravel(),
            "service_id": ss.ravel(),
            "rt_original": rt[np.ix_(users, services)].ravel(),
            "rt_ps": rt_ps[np.ix_(users, services)].ravel(),
            "tp_original": tp[np.ix_(users, services)].ravel(),
            "tp_ps": tp_ps[np.ix_(users, services)].ravel(),
        }).merge(service_features, on="service_id", how="left", validate="many_to_one")
        pilot.to_csv(PILOT_DATA_DIR / "pilot_calls.csv", index=False)
        # 标签生成中间量单独保存用于审计，不作为训练输入。
        pilot.merge(factors, on="service_id", how="left").to_csv(
            PILOT_DATA_DIR / "pilot_calls_audit.csv", index=False
        )

    if mode in {"full", "all"}:
        _write_tables(FULL_SIMULATED_DATA_DIR, static, dynamic, user_locations, mapping, service_features)
        np.savez_compressed(FULL_SIMULATED_DATA_DIR / "qops_rt_tp_v1.npz", rt_ps=rt_ps, tp_ps=tp_ps)

    summary = _summary(config, rt, tp, rt_ps, tp_ps, static, dynamic, mapping)
    PILOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (PILOT_DATA_DIR / "simulation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成养老医院 QoPS RT/TP 仿真数据")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "simulation_v1.json",
    )
    parser.add_argument("--mode", choices=["pilot", "full", "all"], default="all")
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

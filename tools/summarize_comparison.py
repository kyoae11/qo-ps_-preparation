"""汇总基础AFIL、直接拼接和门控融合的多种子结果。"""

from pathlib import Path

import pandas as pd

from afil.paths import RESULT_DIR


def main():
    result_dir = RESULT_DIR / "first_comparison"
    csv_path = result_dir / "result.csv"
    if not csv_path.exists():
        raise SystemExit(f"找不到结果文件：{csv_path}")
    data = pd.read_csv(csv_path)
    rows = []
    for qos_type in ("rt", "tp"):
        subset = data[(data["type"] == qos_type) & (data["density"] == 0.1)]
        latest = subset.drop_duplicates(["model", "seed"], keep="last")
        counts = latest.groupby("model")["seed"].nunique()
        if not {"baseline", "entity_concat"}.issubset(counts.index):
            continue
        baseline = latest[latest["model"] == "baseline"]
        enhanced = latest[latest["model"] == "entity_concat"]
        common_seeds = sorted(set(baseline.seed) & set(enhanced.seed))
        baseline = baseline[baseline.seed.isin(common_seeds)]
        enhanced = enhanced[enhanced.seed.isin(common_seeds)]
        if not common_seeds:
            continue
        baseline_mae = baseline.mae.mean()
        entity_mae = enhanced.mae.mean()
        baseline_rmse = baseline.rmse.mean()
        entity_rmse = enhanced.rmse.mean()
        row = {
            "type": qos_type,
            "runs": len(common_seeds),
            "baseline_mae": baseline_mae,
            "baseline_mae_std": baseline.mae.std(ddof=1),
            "entity_mae": entity_mae,
            "entity_mae_std": enhanced.mae.std(ddof=1),
            "mae_improvement_percent": 100.0 * (baseline_mae - entity_mae) / baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_rmse_std": baseline.rmse.std(ddof=1),
            "entity_rmse": entity_rmse,
            "entity_rmse_std": enhanced.rmse.std(ddof=1),
            "rmse_improvement_percent": 100.0 * (baseline_rmse - entity_rmse) / baseline_rmse,
        }
        gated = latest[latest["model"] == "entity_gated"]
        gated = gated[gated.seed.isin(common_seeds)]
        if len(gated) == len(common_seeds):
            row.update({
                "gated_mae": gated.mae.mean(),
                "gated_mae_std": gated.mae.std(ddof=1),
                "gated_mae_improvement_percent": 100.0 * (baseline_mae - gated.mae.mean()) / baseline_mae,
                "gated_rmse": gated.rmse.mean(),
                "gated_rmse_std": gated.rmse.std(ddof=1),
                "gated_rmse_improvement_percent": 100.0 * (baseline_rmse - gated.rmse.mean()) / baseline_rmse,
            })
        rows.append(row)
    if not rows:
        raise SystemExit("RT/TP对照结果尚不完整")
    summary = pd.DataFrame(rows)
    summary.to_csv(result_dir / "first_comparison_summary.csv", index=False)
    lines = [
        "# 首轮基础AFIL与实体增强AFIL对照", "",
        "| 类型 | 次数 | 基线MAE | 实体MAE | MAE改善 | 基线RMSE | 实体RMSE | RMSE改善 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['type'].upper()} | {row['runs']} | "
            f"{row['baseline_mae']:.6f}±{row['baseline_mae_std']:.6f} | "
            f"{row['entity_mae']:.6f}±{row['entity_mae_std']:.6f} | "
            f"{row['mae_improvement_percent']:+.2f}% | {row['baseline_rmse']:.6f} | "
            f"{row['entity_rmse']:.6f} | {row['rmse_improvement_percent']:+.2f}% |"
        )
    text = "\n".join(lines) + "\n"
    if all("gated_mae" in row for row in rows):
        lines = [
            "# AFIL实体融合多随机种子对照", "",
            "| 类型 | 次数 | 基线MAE | 拼接MAE | 门控MAE | 门控MAE改善 | 基线RMSE | 拼接RMSE | 门控RMSE | 门控RMSE改善 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['type'].upper()} | {row['runs']} | {row['baseline_mae']:.6f} | "
                f"{row['entity_mae']:.6f} | {row['gated_mae']:.6f}±{row['gated_mae_std']:.6f} | "
                f"{row['gated_mae_improvement_percent']:+.2f}% | {row['baseline_rmse']:.6f} | "
                f"{row['entity_rmse']:.6f} | {row['gated_rmse']:.6f}±{row['gated_rmse_std']:.6f} | "
                f"{row['gated_rmse_improvement_percent']:+.2f}% |"
            )
        text = "\n".join(lines) + "\n"
    (result_dir / "first_comparison_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

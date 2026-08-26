"""汇总首轮基础AFIL与实体增强AFIL结果。"""

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
        subset = data[(data["type"] == qos_type) & (data["density"] == 0.1) & (data["seed"] == 0)]
        latest = subset.drop_duplicates("model", keep="last").set_index("model")
        if not {"baseline", "entity_concat"}.issubset(latest.index):
            continue
        baseline = latest.loc["baseline"]
        enhanced = latest.loc["entity_concat"]
        rows.append({
            "type": qos_type,
            "baseline_mae": baseline.mae,
            "entity_mae": enhanced.mae,
            "mae_improvement_percent": 100.0 * (baseline.mae - enhanced.mae) / baseline.mae,
            "baseline_rmse": baseline.rmse,
            "entity_rmse": enhanced.rmse,
            "rmse_improvement_percent": 100.0 * (baseline.rmse - enhanced.rmse) / baseline.rmse,
        })
    if not rows:
        raise SystemExit("RT/TP对照结果尚不完整")
    summary = pd.DataFrame(rows)
    summary.to_csv(result_dir / "first_comparison_summary.csv", index=False)
    lines = [
        "# 首轮基础AFIL与实体增强AFIL对照", "",
        "| 类型 | 基线MAE | 实体MAE | MAE改善 | 基线RMSE | 实体RMSE | RMSE改善 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['type'].upper()} | {row['baseline_mae']:.6f} | {row['entity_mae']:.6f} | "
            f"{row['mae_improvement_percent']:+.2f}% | {row['baseline_rmse']:.6f} | "
            f"{row['entity_rmse']:.6f} | {row['rmse_improvement_percent']:+.2f}% |"
        )
    text = "\n".join(lines) + "\n"
    (result_dir / "first_comparison_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

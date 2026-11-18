"""Read-only terminal monitor. Closing this process does not stop training."""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def snapshot(root):
    lines = ["QoPS 三阶段实验 · 实时训练监控（每 3 秒刷新）",
             datetime.datetime.now().strftime("服务器时间：%Y-%m-%d %H:%M:%S"),
             "关闭窗口 / Ctrl+C 仅退出监控，不会停止训练。", "─" * 74]
    try:
        gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"], text=True, timeout=5).strip()
        lines.extend(["显卡编号、型号、利用率、已用/总显存、温度：", gpu])
    except Exception as exc:
        lines.append("显卡状态读取失败：" + type(exc).__name__)
    jobs = read_json(root / "search_jobs.json", [])
    results = list((root / "results").glob("*/result.json"))
    finished = [f for f in results if "_no_cap" not in str(f) and "_no_state" not in str(f)]
    lines.extend(["", f"候选训练已完成：{len(finished)} / {len(jobs)}（每个随机种子单独计数）"])
    logs = sorted((root / "results").glob("*/console.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        current = logs[0]
        rows = []
        for line in current.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
                if "epoch" in item:
                    rows.append(item)
            except ValueError:
                pass
        lines.extend(["最近训练任务：", current.parent.name])
        if rows:
            best = min(rows, key=lambda d: d["val_mae"])
            lines.extend([f"当前轮次：{rows[-1]['epoch']} / 最多 100；连续 12 轮未改善会提前停止。",
                          f"本任务最佳验证 MAE：{best['val_mae']:.6f} 秒（第 {best['epoch']} 轮）",
                          "", "最近轮次          训练 MAE（秒）       验证 MAE（秒）"])
            lines.extend(f"Epoch {d['epoch']:>3}          {d.get('train_mae', float('nan')):>10.6f}           {d['val_mae']:.6f}" for d in rows[-6:])
        else:
            lines.extend(["正在加载数据或拟合模型……"] + current.read_text(errors="replace").splitlines()[-2:])
    lines.extend(["", "最近完成的实验（验证集结果）："])
    for path in sorted(results, key=lambda p: p.stat().st_mtime, reverse=True)[:4]:
        z = read_json(path)
        if z:
            lines.append(f"{path.parent.name}\n  MAE={z['metrics']['mae']:.6f} 秒；最佳轮次={z['best_epoch']}")
    pipeline = root / "formal_pipeline.log"
    if pipeline.exists():
        tail = pipeline.read_text(errors="replace").splitlines()[-3:]
        lines.extend(["", "流程日志："] + [s[:180] for s in tail])
    lines.extend(["─" * 74, "MAE 是平均绝对误差。网络、就绪、执行的量级不同，请勿直接比较。",
                  "目前显示验证过程；最终测试评价在方法与参数冻结后进行。"])
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/qops_lxy_20261118")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    try:
        while True:
            display = snapshot(Path(args.root))
            print(("" if args.once else "\033[2J\033[H") + display, flush=True)
            if args.once:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print("监控已关闭，服务器训练继续运行。")

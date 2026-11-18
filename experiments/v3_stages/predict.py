"""Inference from prepared t0 snapshots only; never opens labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluate import infer

def predict(root, inputs, seed=11, device="cpu"):
    root = Path(root)
    selected = json.loads((root / "selection.json").read_text())
    x = pd.read_pickle(inputs)
    assert not {"network_time", "ready_time", "execution_time", "completion_time", "feasible"} & set(x)
    net = selected["network"]
    ready = selected["ready"]
    network = infer(root / "results" / net["folders"][str(seed)], x, device)
    if ready["kind"] == "mechanistic":
        readiness = x.mechanism_ready.to_numpy()
    else:
        ready_seed = 11 if ready["kind"] == "tree" else seed
        readiness = infer(root / "results" / ready["folders"][str(ready_seed)], x, device)
    execution = x.base_execution_time.to_numpy() / x.processing_capacity.to_numpy()
    assert np.isfinite(execution).all() and (execution >= 0).all()
    output = x[["run_id", "invocation_id", "entity_id", "entity_type"]].copy()
    output["seed"] = seed
    output["pred_network_seconds"] = network
    output["pred_ready_seconds"] = readiness
    output["pred_execution_seconds"] = execution
    output["pred_completion_seconds"] = network + readiness + execution
    assert np.isfinite(output.iloc[:, 5:].to_numpy()).all()
    return output

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inputs", required=True, help="Prepared t0 input pickle, including known-queue mechanism summaries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, choices=[11,22,33], default=11)
    args = ap.parse_args()
    torch.set_num_threads(4)
    result = predict(args.root, args.inputs, args.seed)
    result.to_csv(args.out, index=False)
    print(f"Predicted {len(result)} calls on CPU without loading labels: {args.out}")

"""基础 AFIL、实体直接融合与残差门控融合的公平对照实验。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from afil.load_data import get_list
from afil.paths import FULL_SIMULATED_DATA_DIR, PROJECT_ROOT, RESULT_DIR
from afil.preprocessing import get_input
from qops_preparation.simulation import generate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SharedProjectionAttention(nn.Module):
    """保持基础 AFIL 实际使用共享 Q/K/V 投影的行为。"""

    def __init__(self, embed_dim: int, d_k: int, n_head: int):
        super().__init__()
        self.d_k = d_k
        self.n_head = n_head
        self.projection = nn.Linear(embed_dim, d_k * n_head, bias=False)
        nn.init.normal_(self.projection.weight, 0.0, 0.1)
        self.norm = nn.LayerNorm(d_k)

    def forward(self, fields):
        batch = fields.shape[0]
        projected = self.projection(fields).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        scores = torch.matmul(projected, projected.transpose(-1, -2)) / np.sqrt(self.d_k)
        context = torch.matmul(torch.softmax(scores, dim=-1), projected)
        interaction = 0.5 * (context.sum(dim=1).pow(2) - context.pow(2).sum(dim=1))
        return self.norm(interaction)


class AFILModel(nn.Module):
    def __init__(
        self,
        user_features,
        service_features,
        cardinalities,
        entity_features,
        model_name,
        embed_dim=128,
        d_k=128,
        n_head=6,
        hidden=(1024, 1024, 1024),
    ):
        super().__init__()
        self.model_name = model_name
        self.register_buffer("user_features", user_features)
        self.register_buffer("service_features", service_features)
        self.user_field_count = user_features.shape[1]
        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(size, embed_dim) for size in cardinalities[: self.user_field_count]]
        )
        self.service_embeddings = nn.ModuleList(
            [nn.Embedding(size, embed_dim) for size in cardinalities[self.user_field_count :]]
        )
        base_fields = len(cardinalities)

        if model_name in {"entity_concat", "entity_gated"}:
            self.register_buffer("entity_features", entity_features)
            self.entity_type_embedding = nn.Embedding(2, embed_dim)
            self.interface_embedding = nn.Embedding(4, embed_dim)
            self.capacity_projection = nn.Linear(1, embed_dim)
            self.load_projection = nn.Linear(1, embed_dim)

        if model_name == "entity_concat":
            base_fields += 4

        self.attention = SharedProjectionAttention(embed_dim, d_k, n_head)
        layers = []
        previous = base_fields * d_k
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.dnn = nn.Sequential(*layers)

        if model_name == "entity_gated":
            self.entity_encoder = nn.Sequential(
                nn.Linear(4 * embed_dim, previous),
                nn.ReLU(),
                nn.LayerNorm(previous),
            )
            self.entity_gate = nn.Linear(previous * 2, previous)
            # 初始时主要依赖已验证的 AFIL 主干，再逐步学习实体信息的注入强度。
            nn.init.zeros_(self.entity_gate.weight)
            nn.init.constant_(self.entity_gate.bias, -2.0)
        self.output = nn.Linear(previous, 1)

    def encode_entity(self, service_id):
        ef = self.entity_features[service_id]
        return torch.cat([
            self.entity_type_embedding(ef[:, 0].long()),
            self.interface_embedding(ef[:, 1].long()),
            self.capacity_projection(ef[:, 2:3]),
            self.load_projection(ef[:, 3:4]),
        ], dim=1)

    def forward(self, user_id, service_id):
        uf = self.user_features[user_id]
        sf = self.service_features[service_id]
        user_tokens = [embedding(uf[:, idx]) for idx, embedding in enumerate(self.user_embeddings)]
        service_tokens = [embedding(sf[:, idx]) for idx, embedding in enumerate(self.service_embeddings)]
        fields = user_tokens + service_tokens
        if self.model_name == "entity_concat":
            entity_tokens = self.encode_entity(service_id).view(service_id.shape[0], 4, -1)
            fields.extend(entity_tokens.unbind(dim=1))
        encoded = self.attention(torch.stack(fields, dim=1)).flatten(start_dim=1)
        base_representation = self.dnn(encoded)
        if self.model_name == "entity_gated":
            entity_representation = self.entity_encoder(self.encode_entity(service_id))
            gate = torch.sigmoid(
                self.entity_gate(torch.cat([base_representation, entity_representation], dim=1))
            )
            base_representation = base_representation + gate * entity_representation
        return self.output(base_representation)


def load_entity_tensor(path: Path):
    frame = pd.read_csv(path).sort_values("service_id")
    type_map = {"static": 0, "dynamic": 1}
    interface_map = {"ethernet": 0, "wifi6": 1, "wifi5": 2, "cellular": 3}
    capacity = frame["processing_capacity"].to_numpy(dtype=np.float32)
    load = frame["device_load"].to_numpy(dtype=np.float32)
    capacity = (capacity - capacity.mean()) / (capacity.std() + 1e-8)
    load = (load - load.mean()) / (load.std() + 1e-8)
    values = np.column_stack([
        frame["entity_type"].map(type_map).to_numpy(),
        frame["network_interface_type"].map(interface_map).to_numpy(),
        capacity,
        load,
    ]).astype(np.float32)
    if np.isnan(values).any():
        raise ValueError("实体特征存在无法编码的类别或缺失值")
    return torch.tensor(values)


def evaluate(model, loader, device):
    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    with torch.no_grad():
        for user_id, service_id, target in loader:
            user_id, service_id = user_id.to(device), service_id.to(device)
            target = target.to(device).reshape(-1, 1)
            prediction = model(user_id, service_id)
            absolute_error += torch.abs(prediction - target).sum().item()
            squared_error += torch.square(prediction - target).sum().item()
    count = len(loader.dataset)
    return absolute_error / count, np.sqrt(squared_error / count)


def run(args):
    if args.model not in {"baseline", "entity_concat", "entity_gated"}:
        raise ValueError("model 必须是 baseline、entity_concat 或 entity_gated")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda)
        device = torch.device(f"cuda:{args.cuda}")
    print(f"device={device} model={args.model} type={args.type}")

    npz_path = FULL_SIMULATED_DATA_DIR / "qops_rt_tp_v1.npz"
    if not npz_path.exists():
        generate(PROJECT_ROOT / "config" / "simulation_v1.json", mode="full")
    matrices = np.load(npz_path)
    matrix = matrices[f"{args.type}_ps"]

    user_path = PROJECT_ROOT / "data" / "ws_dream" / "dataset1" / "userlist.txt"
    service_path = PROJECT_ROOT / "data" / "ws_dream" / "dataset1" / "wslist.txt"
    original_matrix_path = PROJECT_ROOT / "data" / "ws_dream" / "dataset1" / f"{args.type}Matrix.txt"
    user, service, _, user_cards, service_cards = get_input(
        user_path, service_path, original_matrix_path
    )
    entity = load_entity_tensor(FULL_SIMULATED_DATA_DIR / "service_entity_features.csv")

    train, test = get_list(matrix, args.md, args.seed)
    print(f"train={len(train)} test={len(test)}")
    train_ds = TensorDataset(
        torch.tensor(train[:, 0], dtype=torch.long),
        torch.tensor(train[:, 1], dtype=torch.long),
        torch.tensor(train[:, 2], dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(test[:, 0], dtype=torch.long),
        torch.tensor(test[:, 1], dtype=torch.long),
        torch.tensor(test[:, 2], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=args.num_workers)

    model = AFILModel(
        torch.tensor(user.astype(np.int64)),
        torch.tensor(service.astype(np.int64)),
        user_cards + service_cards,
        entity,
        args.model,
        args.embed_dim,
        args.d_k,
        args.n_head,
        tuple(int(value) for value in args.hidden_units.split(",")),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for user_id, service_id, target in train_loader:
            user_id, service_id = user_id.to(device), service_id.to(device)
            target = target.to(device).reshape(-1, 1)
            prediction = model(user_id, service_id)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item() * len(target)
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mae, rmse = evaluate(model, test_loader, device)
            best = {"epoch": epoch, "mae": mae, "rmse": rmse}
            print(
                f"epoch={epoch}/{args.epochs} train_mae={total_loss/len(train_ds):.6f} "
                f"test_mae={mae:.6f} test_rmse={rmse:.6f}"
            )

    result_dir = RESULT_DIR / "first_comparison"
    result_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "model": args.model,
        "type": args.type,
        "density": args.md,
        "seed": args.seed,
        "epochs": args.epochs,
        **best,
    }
    result_file = result_dir / f"{args.model}_{args.type}_md{args.md:g}_seed{args.seed}.json"
    result_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = result_dir / "result.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="运行基础AFIL与实体增强AFIL对照实验")
    parser.add_argument(
        "--model", choices=["baseline", "entity_concat", "entity_gated"], required=True
    )
    parser.add_argument("--type", choices=["rt", "tp"], default="rt")
    parser.add_argument("--md", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--hidden_units", default="1024,1024,1024")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.md <= 1 or args.epochs < 1 or args.eval_every < 1:
        parser.error("md必须位于(0,1]，epochs和eval_every必须为正数")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

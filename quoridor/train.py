import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from quoridor.agent import mirror_features, mirror_perm
from quoridor.model import QuoridorNet, masked_logits, save


def load_data(paths: list[str]) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in paths]
    boards = {tuple(int(x) for x in p["board"]) for p in parts}
    if len(boards) != 1:
        raise ValueError(f"mixed boards: {boards}")
    keys = ["features", "mask", "action", "value", "exact"]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    flat = np.ascontiguousarray(data["features"].reshape(len(data["features"]), -1))
    _, first = np.unique(
        flat.view(np.dtype((np.void, flat.shape[1] * flat.itemsize))), return_index=True
    )
    first.sort()
    if len(first) < len(flat):
        print(f"dropped {len(flat) - len(first)} duplicate positions across files")
    data = {k: v[first] for k, v in data.items()}
    data["board"] = parts[0]["board"]
    return data


def value_target(data: dict[str, np.ndarray], kind: str) -> np.ndarray:
    if kind == "exact":
        return np.where(np.isnan(data["exact"]), data["value"], data["exact"])
    return data["value"]


def run_epoch(
    net: QuoridorNet,
    tensors: dict[str, torch.Tensor],
    idx: np.ndarray,
    args: argparse.Namespace,
    optim: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler] | None,
    perm: torch.Tensor | None,
) -> dict[str, float]:
    train = optim is not None
    net.train(train)
    totals = {"policy": 0.0, "value": 0.0, "acc": 0.0, "sign": 0.0, "exact_n": 0.0}
    n = 0
    for start in range(0, len(idx), args.batch):
        b = torch.from_numpy(idx[start : start + args.batch])
        x = tensors["features"][b].to(args.device).float()
        mask = tensors["mask"][b].to(args.device)
        action = tensors["action"][b].to(args.device)
        target = tensors["target"][b].to(args.device)
        exact = tensors["exact"][b].to(args.device)
        if train and perm is not None:
            flip = torch.rand(len(b), device=args.device) < 0.5
            x[flip] = mirror_features(x[flip])
            mask[flip] = mask[flip][:, perm]
            action[flip] = perm[action[flip]]
        with torch.set_grad_enabled(train):
            logits, value, _ = net(x)
            logits = masked_logits(logits, mask)
            lp = F.cross_entropy(logits, action)
            lv = F.mse_loss(value, target)
            if optim is not None:
                opt, sched = optim
                opt.zero_grad(set_to_none=True)
                (lp + args.value_weight * lv).backward()
                opt.step()
                sched.step()
        k = len(b)
        n += k
        totals["policy"] += lp.item() * k
        totals["value"] += lv.item() * k
        totals["acc"] += (logits.argmax(1) == action).float().sum().item()
        solved = ~torch.isnan(exact)
        totals["sign"] += (
            (torch.sign(value[solved]) == exact[solved]).float().sum().item()
        )
        totals["exact_n"] += solved.sum().item()
    out = {k: totals[k] / n for k in ("policy", "value", "acc")}
    out["sign"] = totals["sign"] / totals["exact_n"] if totals["exact_n"] else math.nan
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="supervised training on alpha-beta self-play data"
    )
    p.add_argument("data", nargs="+")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--value-target", choices=["search", "exact"], default="search")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--run",
        required=True,
        help="run name, saved to runs/RUN/net.pt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.path.exists(os.path.join("runs", args.run)):
        raise SystemExit(f"runs/{args.run} exists")
    torch.manual_seed(args.seed)
    data = load_data(args.data)
    width, height = int(data["board"][0]), int(data["board"][1])
    tensors = {
        "features": torch.from_numpy(data["features"]),
        "mask": torch.from_numpy(data["mask"]),
        "action": torch.from_numpy(data["action"].astype(np.int64)),
        "target": torch.from_numpy(
            value_target(data, args.value_target).astype(np.float32)
        ),
        "exact": torch.from_numpy(data["exact"]),
    }
    n = len(data["action"])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = order[:n_val], order[n_val:]
    solved = ~np.isnan(data["exact"][val_idx])
    if solved.any():
        base = np.mean(
            np.sign(data["value"][val_idx][solved]) == data["exact"][val_idx][solved]
        )
        print(
            f"baseline: alpha-beta value sign matches exact on {base:.1%} of solved val positions"
        )

    perm = (
        torch.from_numpy(mirror_perm(width, height)).to(args.device)
        if width % 2 == 1
        else None
    )
    net = QuoridorNet(args.blocks, args.channels).to(args.device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps = args.epochs * math.ceil(len(train_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1
    )
    print(
        f"{n - n_val} train / {n_val} val positions, {width}x{height}, "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params, device {args.device}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            net, tensors, rng.permutation(train_idx), args, (opt, sched), perm
        )
        va = run_epoch(net, tensors, val_idx, args, None, None)
        print(
            f"epoch {epoch:3d}  train p {tr['policy']:.3f} v {tr['value']:.3f} acc {tr['acc']:.1%}  |  "
            f"val p {va['policy']:.3f} v {va['value']:.3f} acc {va['acc']:.1%} sign {va['sign']:.1%}  "
            f"{time.time() - t0:.1f}s"
        )

    out = os.path.join("runs", args.run, "net.pt")
    os.makedirs(os.path.dirname(out))
    save(net, out, {"board": data["board"].tolist()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from quoridor import quoridor_engine as qe
from quoridor.agent import (
    MAX_CONSIDERED,
    NetEvaluator,
    mirror_features,
    mirror_perm,
    positive,
    run_search,
)
from quoridor.arena import (
    Opener,
    elo,
    net_vs_net,
    opening,
    play_match,
    policy_opener,
    uniform_move,
)
from quoridor.model import QuoridorNet, masked_logits, save

FIELDS = ("features", "mask", "policy", "value", "q", "moves_left")


@dataclass
class Slot:
    game: qe.Game
    features: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    sides: list[int] = field(default_factory=list)
    qs: list[float] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)
    check: bool = False
    would_resign: int = -1
    resigned: int = -1


@dataclass
class Batch:
    features: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    q: np.ndarray
    moves_left: np.ndarray


def finish(slot: Slot) -> Batch:
    g = slot.game
    w = g.winner() if slot.resigned < 0 else 1 - slot.resigned
    z = [0.0 if w < 0 else 1.0 if w == s else -1.0 for s in slot.sides]
    if slot.resigned < 0:
        left = [(g.ply - p) / g.max_plies for p in slot.plies]
    else:
        left = [-1.0] * len(slot.plies)
    return Batch(
        np.stack(slot.features).astype(np.float16),
        np.stack(slot.masks),
        np.stack(slot.policies),
        np.array(z, dtype=np.float32),
        np.array(slot.qs, dtype=np.float32),
        np.array(left, dtype=np.float32),
    )


def play_games(
    evaluate: NetEvaluator,
    mcts: qe.MCTS,
    args: argparse.Namespace,
    rng: random.Random,
    opener: Opener = uniform_move,
) -> tuple[list[Batch], int, dict[str, int]]:
    width, height, walls = evaluate.board
    started = 0
    slots: list[Slot] = []
    done: list[Batch] = []
    plies = 0
    stats = {"resigned": 0, "checked": 0, "false": 0}
    while started < args.games or slots:
        while started < args.games and len(slots) < args.parallel:
            slots.append(
                Slot(
                    opening(
                        width,
                        height,
                        walls,
                        rng.randint(0, args.open_plies),
                        rng,
                        opener,
                    ),
                    check=args.resign > 0 and rng.random() < args.resign_check,
                )
            )
            started += 1
        run_search(mcts, [s.game for s in slots], evaluate, noise=True)
        actions = mcts.actions()
        policies = mcts.policies()
        root_values = mcts.root_values()
        keep: list[Slot] = []
        for i, s in enumerate(slots):
            s.features.append(s.game.features())
            s.masks.append(s.game.legal_mask())
            s.policies.append(policies[i])
            s.sides.append(s.game.side)
            s.qs.append(root_values[i])
            s.plies.append(s.game.ply)
            if args.resign > 0 and root_values[i] < -args.resign:
                if not s.check:
                    s.resigned = s.game.side
                    stats["resigned"] += 1
                    done.append(finish(s))
                    continue
                if s.would_resign < 0:
                    s.would_resign = s.game.side
            s.game.apply(actions[i])
            plies += 1
            if not s.game.is_terminal():
                keep.append(s)
            else:
                if s.would_resign >= 0:
                    stats["checked"] += 1
                    stats["false"] += s.game.winner() != 1 - s.would_resign
                done.append(finish(s))
        slots = keep
    return done, plies, stats


class Buffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.store: Batch | None = None
        self.size = 0
        self.pos = 0

    def add(self, batches: list[Batch]) -> None:
        new = {
            k: np.concatenate([getattr(b, k) for b in batches])[-self.capacity :]
            for k in FIELDS
        }
        n = len(new["value"])
        if n == 0:
            return
        if self.store is None:
            self.store = Batch(
                *(
                    np.zeros((self.capacity, *new[k].shape[1:]), dtype=new[k].dtype)
                    for k in FIELDS
                )
            )
        idx = (self.pos + np.arange(n)) % self.capacity
        for k in FIELDS:
            getattr(self.store, k)[idx] = new[k]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def arrays(self) -> Batch:
        assert self.store is not None
        return Batch(*(getattr(self.store, k)[: self.size] for k in FIELDS))


def train_steps(
    evaluate: NetEvaluator,
    opt: torch.optim.Optimizer,
    data: Batch,
    steps: int,
    args: argparse.Namespace,
    perm: torch.Tensor | None,
) -> tuple[float, float, float]:
    net, dev = evaluate.net, evaluate.device
    net.train()
    t = {k: torch.from_numpy(getattr(data, k)) for k in FIELDS}
    w = args.search_value_weight
    target_all = (1.0 - w) * t["value"] + w * t["q"]
    sums = [0.0, 0.0, 0.0]
    for _ in range(steps):
        b = torch.randint(0, len(target_all), (args.batch,))
        x, mask, pi = (
            t["features"][b].to(dev).float(),
            t["mask"][b].to(dev),
            t["policy"][b].to(dev),
        )
        target, left = target_all[b].to(dev), t["moves_left"][b].to(dev)
        if perm is not None:
            flip = torch.rand(len(b), device=dev) < 0.5
            x[flip] = mirror_features(x[flip])
            mask[flip] = mask[flip][:, perm]
            pi[flip] = pi[flip][:, perm]
        logits, value, moves_left = net(x)
        lp = -(pi * F.log_softmax(masked_logits(logits, mask), dim=1)).sum(1).mean()
        lv = F.mse_loss(value, target)
        known = left >= 0
        lm = (
            F.mse_loss(moves_left[known], left[known])
            if bool(known.any())
            else moves_left.sum() * 0.0
        )
        opt.zero_grad(set_to_none=True)
        (lp + args.value_weight * lv + args.moves_left_weight * lm).backward()
        opt.step()
        for i, v in enumerate((lp, lv, lm)):
            sums[i] += v.item()
    net.eval()
    n = max(steps, 1)
    return sums[0] / n, sums[1] / n, sums[2] / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gumbel AlphaZero self-play training loop")
    p.add_argument(
        "--init",
        required=True,
        help="checkpoint to start from, or 'scratch' (needs --board)",
    )
    p.add_argument("--blocks", type=int, default=6, help="net size for --init scratch")
    p.add_argument(
        "--channels", type=int, default=64, help="net size for --init scratch"
    )
    p.add_argument(
        "--run",
        required=True,
        help="run name, saved to runs/RUN",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--games", type=int, default=1000, help="games per iteration")
    p.add_argument("--parallel", type=int, default=256)
    p.add_argument("--sims", type=int, default=28, help="sims per move")
    p.add_argument("--max-considered", type=int, default=MAX_CONSIDERED)
    p.add_argument(
        "--open-plies",
        type=int,
        default=2,
        help="up to this many random plies at the start",
    )
    p.add_argument(
        "--resign",
        type=float,
        default=0.0,
        help="resign below -RESIGN (0: never)",
    )
    p.add_argument(
        "--resign-check",
        type=float,
        default=0.1,
        help="fraction of games that never resign",
    )
    p.add_argument(
        "--open-temp",
        type=positive,
        help="temperature for opening plies (default: random)",
    )
    p.add_argument(
        "--buffer", type=int, default=100_000, help="replay buffer size in positions"
    )
    p.add_argument(
        "--reuse",
        type=float,
        default=4.0,
        help="samples per new position",
    )
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument(
        "--search-value-weight",
        type=float,
        default=0.0,
        help="weight of search value in the value target",
    )
    p.add_argument(
        "--moves-left-weight",
        type=float,
        default=0.0,
        help="loss weight of the moves-left head",
    )
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--eval-sims", type=int, default=64)
    p.add_argument(
        "--eval-net", help="evaluate against this net instead of depth-4 alpha-beta"
    )
    p.add_argument(
        "--board",
        type=int,
        nargs=3,
        metavar=("W", "H", "WALLS"),
        help="play on this board instead of the net's",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.join("runs", args.run)
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} exists")
    os.makedirs(run_dir)
    log_path = os.path.join(run_dir, "log.txt")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    init = args.init
    if init == "scratch":
        if not args.board:
            raise SystemExit("--init scratch needs --board")
        init = os.path.join(run_dir, "iter000.pt")
        save(
            QuoridorNet(args.blocks, args.channels),
            init,
            {"board": args.board, "iter": 0},
        )
    evaluate = NetEvaluator(init)
    if args.board:
        evaluate.board = args.board
    width, height, _ = evaluate.board
    perm = (
        torch.from_numpy(mirror_perm(width, height)).to(evaluate.device)
        if width % 2 == 1
        else None
    )
    opt = torch.optim.AdamW(
        evaluate.net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    mcts = qe.MCTS(sims=args.sims, max_considered=args.max_considered, seed=args.seed)
    buffer = Buffer(args.buffer)
    reference = NetEvaluator(args.eval_net) if args.eval_net else None

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}  {vars(args)}")
    for it in range(1, args.iters + 1):
        t0 = time.time()
        opener = (
            policy_opener(evaluate, args.open_temp)
            if args.open_temp is not None
            else uniform_move
        )
        batches, plies, stats = play_games(evaluate, mcts, args, rng, opener)
        t_play = time.time() - t0
        buffer.add(batches)
        new = sum(len(b.value) for b in batches)
        data = buffer.arrays()
        steps = math.ceil(new * args.reuse / args.batch)
        t1 = time.time()
        lp, lv, lm = train_steps(evaluate, opt, data, steps, args, perm)
        t_train = time.time() - t1
        extra = {"board": evaluate.board, "iter": it}
        save(evaluate.net, os.path.join(run_dir, f"iter{it:03d}.pt"), extra)
        log(
            f"iter {it:3d}  games {args.games} plies {plies} samples {new} buffer {len(data.value)}  "
            f"play {t_play:.0f}s ({args.games / t_play:.1f} games/s) train {t_train:.0f}s ({steps} steps)  "
            f"loss p {lp:.3f} v {lv:.3f} ml {lm:.4f}"
            + (
                f"  resigned {stats['resigned']}/{args.games}, wrong in checks {stats['false']}/{stats['checked']}"
                if args.resign > 0
                else ""
            )
        )
        if args.eval_every and it % args.eval_every == 0:
            t2 = time.time()
            if reference is not None:
                w, d, l = net_vs_net(
                    evaluate,
                    reference,
                    args.eval_games,
                    args.eval_sims,
                    args.max_considered,
                    args.open_plies,
                    args.seed,
                    opener,
                )
                label = f"vs {args.eval_net}"
            else:
                w, d, l = play_match(
                    evaluate,
                    args.eval_games,
                    args.eval_sims,
                    args.max_considered,
                    seed=args.seed,
                    open_plies=args.open_plies,
                    opener=opener,
                )
                label = "vs alpha-beta depth 4"
            score = (w + 0.5 * d) / (w + d + l)
            log(
                f"eval {it:3d}  {label}, {args.eval_sims} sims: +{w} ={d} -{l}  "
                f"score {score:.1%}  elo {elo(score):+.0f}  ({time.time() - t2:.0f}s)"
            )


if __name__ == "__main__":
    main()

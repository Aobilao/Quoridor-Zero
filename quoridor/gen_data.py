import argparse
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from quoridor import quoridor_engine as qe


@dataclass
class Sample:
    features: np.ndarray
    mask: np.ndarray
    action: int
    score: int
    side: int
    exact: float


@dataclass
class Shared:
    seen: set[int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    games_done: int = 0


def score_to_value(score: int, k: float) -> float:
    if score > qe.MATE_BOUND:
        return 1.0
    if score < -qe.MATE_BOUND:
        return -1.0
    return math.tanh(score / k)


def random_opening(
    game: qe.Game, plies: int, rng: random.Random, wall_prob: float
) -> None:
    for _ in range(plies):
        if game.is_terminal():
            return
        actions = game.legal_actions()
        pawn = [a for a in actions if a < qe.N_PAWN_ACTIONS]
        walls = [a for a in actions if a >= qe.N_PAWN_ACTIONS]
        pool = walls if walls and rng.random() < wall_prob else pawn
        game.apply(rng.choice(pool))


def solve(game: qe.Game, solver: qe.Searcher, time_ms: int) -> float:
    r = solver.search(game, max_depth=200, time_ms=time_ms)
    if r.score > qe.MATE_BOUND:
        return 1.0
    if r.score < -qe.MATE_BOUND:
        return -1.0
    return math.nan


def play_game(
    args: argparse.Namespace,
    seed: int,
    searcher: qe.Searcher,
    solver: qe.Searcher | None,
    shared: Shared,
) -> tuple[list[Sample], int]:
    rng = random.Random(seed)
    game = qe.Game(*args.board, max_plies=1000 if solver else 0)
    random_opening(game, rng.randint(args.open_min, args.open_max), rng, args.wall_prob)
    searcher.clear()
    samples: list[Sample] = []
    while not game.is_terminal():
        r = searcher.search(game, max_depth=args.depth, time_ms=0)
        key = game.hash()
        with shared.lock:
            fresh = key not in shared.seen
            shared.seen.add(key)
        if fresh:
            exact = solve(game, solver, args.solve_ms) if solver else math.nan
            samples.append(
                Sample(
                    game.features(),
                    game.legal_mask(),
                    r.action,
                    r.score,
                    game.side,
                    exact,
                )
            )
        game.apply(r.action)
    with shared.lock:
        shared.games_done += 1
    return samples, game.winner()


def worker(
    args: argparse.Namespace, seeds: list[int], shared: Shared
) -> list[tuple[list[Sample], int]]:
    searcher = qe.Searcher(tt_mb=args.tt_mb)
    solver = qe.Searcher(tt_mb=args.tt_mb * 4) if args.solve_ms > 0 else None
    return [play_game(args, s, searcher, solver, shared) for s in seeds]


def collect(args: argparse.Namespace) -> dict[str, np.ndarray]:
    shared = Shared()
    seeds = [args.seed * 1_000_003 + i for i in range(args.games)]
    chunks = [seeds[i :: args.threads] for i in range(args.threads)]
    with ThreadPoolExecutor(args.threads) as pool:
        futures = [pool.submit(worker, args, c, shared) for c in chunks]
        while not all(f.done() for f in futures):
            time.sleep(1.0)
            print(
                f"\rgames {shared.games_done}/{args.games}  positions {len(shared.seen)}",
                end="",
                flush=True,
            )
        results = [g for f in futures for g in f.result()]
    print()
    feats, masks, actions, values, outcomes, scores, exact = [], [], [], [], [], [], []
    for samples, winner in results:
        for s in samples:
            feats.append(s.features)
            masks.append(s.mask)
            actions.append(s.action)
            scores.append(s.score)
            exact.append(s.exact)
            values.append(score_to_value(s.score, args.value_scale))
            outcomes.append(0.0 if winner < 0 else 1.0 if winner == s.side else -1.0)
    return {
        "features": np.stack(feats).astype(np.float16),
        "mask": np.stack(masks).astype(np.uint8),
        "action": np.array(actions, dtype=np.int16),
        "value": np.array(values, dtype=np.float32),
        "outcome": np.array(outcomes, dtype=np.float32),
        "score": np.array(scores, dtype=np.int32),
        "exact": np.array(exact, dtype=np.float32),
        "board": np.array(args.board, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="alpha-beta self-play data generation")
    p.add_argument(
        "--board",
        type=int,
        nargs=3,
        default=[5, 5, 3],
        metavar=("W", "H", "WALLS"),
    )
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--open-min", type=int, default=4)
    p.add_argument("--open-max", type=int, default=8)
    p.add_argument(
        "--wall-prob",
        type=float,
        default=0.3,
        help="chance a random opening ply is a wall",
    )
    p.add_argument(
        "--value-scale",
        type=float,
        default=300.0,
        help="value = tanh(score / scale)",
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tt-mb", type=int, default=32)
    p.add_argument(
        "--solve-ms",
        type=int,
        default=0,
        help="if > 0, also store exact win/loss labels",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/data.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    data = collect(args)
    np.savez_compressed(args.out, allow_pickle=False, **data)
    n = len(data["action"])
    agree = float(np.mean(np.sign(data["value"]) == np.sign(data["outcome"])))
    print(f"{n} positions in {time.time() - t0:.1f}s -> {args.out}")
    print(f"value sign agrees with game outcome on {agree:.1%} of positions")
    solved = ~np.isnan(data["exact"])
    if solved.any():
        e = data["exact"][solved]
        print(
            f"solved {solved.mean():.1%}, search value sign matches exact on {np.mean(np.sign(data['value'][solved]) == e):.1%}"
        )


if __name__ == "__main__":
    main()

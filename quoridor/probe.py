import argparse
import os
import random
import time

import numpy as np

from quoridor import quoridor_engine as qe
from quoridor.agent import MAX_CONSIDERED, NetEvaluator, run_search

MAX_PLIES = 1000


def replay(width: int, height: int, walls: int, moves: list[int]) -> qe.Game:
    g = qe.Game(width, height, walls, max_plies=MAX_PLIES)
    for a in moves:
        g.apply(a)
    return g


def proven(score: int) -> int:
    return 1 if score > qe.MATE_BOUND else -1 if score < -qe.MATE_BOUND else 0


def build(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = random.Random(args.seed)
    solver = qe.Searcher(tt_mb=256)
    player = qe.Searcher(tt_mb=32)
    seqs: list[list[int]] = []
    results: list[int] = []
    good: list[np.ndarray] = []
    t0 = time.time()
    while len(seqs) < args.positions:
        g = qe.Game(*args.board, max_plies=MAX_PLIES)
        moves: list[int] = []
        for _ in range(rng.randint(args.min_plies, args.max_plies)):
            if g.is_terminal():
                break
            if rng.random() < 0.5:
                a = rng.choice(g.legal_actions())
            else:
                a = player.search(g, max_depth=2, time_ms=0).action
            g.apply(a)
            moves.append(a)
        if g.is_terminal() or len(g.legal_actions()) < 2:
            continue
        result = proven(solver.search(g, max_depth=200, time_ms=args.solve_ms).score)
        if result == 0:
            continue
        mask = np.zeros(g.num_actions, dtype=np.uint8)
        if result > 0:
            for a in g.legal_actions():
                c = g.clone()
                c.apply(a)
                if (
                    c.is_terminal()
                    or proven(
                        solver.search(c, max_depth=200, time_ms=args.solve_ms).score
                    )
                    < 0
                ):
                    mask[a] = 1
        seqs.append(moves)
        results.append(result)
        good.append(mask)
        print(f"\rbuilt {len(seqs)}/{args.positions}", end="", flush=True)
    print(f"  ({time.time() - t0:.0f}s)")
    width = max(len(s) for s in seqs)
    padded = np.full((len(seqs), width), -1, dtype=np.int16)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return {
        "moves": padded,
        "result": np.array(results, dtype=np.int8),
        "good": np.stack(good),
        "board": np.array(args.board, dtype=np.int32),
    }


def load_games(data: dict[str, np.ndarray]) -> list[qe.Game]:
    w, h, walls = (int(x) for x in data["board"])
    return [
        replay(w, h, walls, [int(a) for a in row if a >= 0]) for row in data["moves"]
    ]


def score(
    path: str,
    data: dict[str, np.ndarray],
    games: list[qe.Game],
    sims: list[int],
) -> str:
    evaluate = NetEvaluator(path)
    x = qe.batch_features(games)
    _, values = evaluate(x)
    result = data["result"]
    sign = float(np.mean(np.sign(values) == result))
    win = result > 0
    parts = [f"value sign {sign:.1%}"]
    for s in sims:
        mcts = qe.MCTS(sims=s, max_considered=MAX_CONSIDERED, seed=0)
        run_search(mcts, games, evaluate)
        actions = np.array(mcts.actions())
        ok = data["good"][np.arange(len(games)), actions].astype(bool)
        parts.append(f"{s} sims keeps win {ok[win].mean():.1%}")
    return (
        f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path):12s} "
        + "  ".join(parts)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="score nets against exact solver labels")
    p.add_argument("nets", nargs="*")
    p.add_argument("--data", default="data/probe5.npz")
    p.add_argument("--positions", type=int, default=400)
    p.add_argument(
        "--board",
        type=int,
        nargs=3,
        default=[5, 5, 3],
        metavar=("W", "H", "WALLS"),
    )
    p.add_argument("--solve-ms", type=int, default=5000)
    p.add_argument("--sims", type=int, nargs="+", default=[1, 64])
    p.add_argument("--min-plies", type=int, default=4)
    p.add_argument("--max-plies", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.data):
        np.savez_compressed(args.data, allow_pickle=False, **build(args))
    data = dict(np.load(args.data))
    games = load_games(data)
    win = data["result"] > 0
    good = data["good"][win].sum(1)
    legal = np.array([len(g.legal_actions()) for g in games])[win]
    print(
        f"{len(games)} positions, {win.sum()} won for the mover. Winning moves per won position: "
        f"mean {good.mean():.1f} of {legal.mean():.1f} legal (random move keeps win {np.mean(good / legal):.1%})"
    )
    for path in args.nets:
        print(score(path, data, games, args.sims), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time
from quoridor import quoridor_engine as qe
from quoridor.agent import MAX_CONSIDERED, Mover, NetEvaluator, run_search

Book = dict[int, tuple[int, float]]


def load_book(path: str, board: list[int]) -> Book:
    with open(path) as f:
        data = json.load(f)
    if data["board"] != list(board):
        raise ValueError(f"book {path} is for board {data['board']}, not {board}")
    return {int(h): (int(e[0]), float(e[1])) for h, e in data["positions"].items()}


def with_book(book: Book, mover: Mover) -> Mover:
    def move(games: list[qe.Game]) -> list[int]:
        actions = [book[g.hash()][0] if g.hash() in book else -1 for g in games]
        rest = [i for i, a in enumerate(actions) if a < 0]
        if rest:
            for i, a in zip(rest, mover([games[i] for i in rest])):
                actions[i] = a
        return actions

    return move


def analyse(
    games: list[qe.Game],
    evaluate: NetEvaluator,
    sims: int,
    max_considered: int,
    batch: int,
) -> list[tuple[int, float, list[tuple[int, int]]]]:
    out = []
    mcts = qe.MCTS(sims=sims, max_considered=max_considered)
    for start in range(0, len(games), batch):
        chunk = games[start : start + batch]
        run_search(mcts, chunk, evaluate)
        actions, values = mcts.actions(), mcts.root_values()
        for i in range(len(chunk)):
            visits = sorted(mcts.visits(i), key=lambda av: -av[1])
            out.append((actions[i], values[i], visits))
    return out


def build(args: argparse.Namespace) -> tuple[list[int], dict[int, list[float]]]:
    evaluate = NetEvaluator(args.net)
    width, height, walls = evaluate.board
    positions: dict[int, list[float]] = {}
    replies: dict[int, list[int]] = {}
    for our_side in (0, 1):
        frontier = [qe.Game(width, height, walls)]
        for ply in range(args.plies):
            todo = {g.hash(): g for g in frontier if g.hash() not in positions}
            t0 = time.time()
            for (h, g), (action, value, visits) in zip(
                todo.items(),
                analyse(
                    list(todo.values()),
                    evaluate,
                    args.sims,
                    args.max_considered,
                    args.batch,
                ),
            ):
                positions[h] = [action, round(value, 4), ply]
                replies[h] = [a for a, n in visits[: args.replies] if n > 0]
            nxt: dict[int, qe.Game] = {}
            for g in frontier:
                h = g.hash()
                moves = [int(positions[h][0])] if g.side == our_side else replies[h]
                for a in moves:
                    child = g.clone()
                    child.apply(a)
                    if not child.is_terminal():
                        nxt[child.hash()] = child
            frontier = list(nxt.values())
            print(
                f"side {our_side} ply {ply}: searched {len(todo)} new positions in {time.time() - t0:.0f}s, "
                f"book {len(positions)}, next frontier {len(frontier)}",
                flush=True,
            )
    return list(evaluate.board), positions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="build an opening book"
    )
    p.add_argument("net")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--plies", type=int, default=12, help="book depth in plies"
    )
    p.add_argument(
        "--replies",
        type=int,
        default=3,
        help="opponent replies to follow",
    )
    p.add_argument("--sims", type=int, default=4096, help="sims per book position")
    p.add_argument("--max-considered", type=int, default=MAX_CONSIDERED)
    p.add_argument("--batch", type=int, default=64, help="positions searched at once")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    board, positions = build(args)
    data = {
        "board": board,
        "net": args.net,
        "sims": args.sims,
        "plies": args.plies,
        "replies": args.replies,
        "positions": {str(h): v for h, v in positions.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"wrote {args.out}: {len(positions)} positions in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

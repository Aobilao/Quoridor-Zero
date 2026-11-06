import argparse
import math
import random
import time
from collections.abc import Callable

import numpy as np

from quoridor import quoridor_engine as qe
from quoridor.agent import MAX_CONSIDERED, Mover, NetEvaluator, positive, run_search
from quoridor.book import load_book, with_book

Opener = Callable[[qe.Game, random.Random], int]


def uniform_move(g: qe.Game, rng: random.Random) -> int:
    return rng.choice(g.legal_actions())


def policy_opener(evaluate: NetEvaluator, temp: float) -> Opener:
    def move(g: qe.Game, rng: random.Random) -> int:
        logits, _ = evaluate(g.features()[None])
        legal = g.legal_actions()
        z = logits[0][legal].astype(np.float64) / temp
        weights = np.exp(z - z.max())
        return rng.choices(legal, weights=weights.tolist())[0]

    return move


def opening(
    width: int,
    height: int,
    walls: int,
    plies: int,
    rng: random.Random,
    opener: Opener = uniform_move,
) -> qe.Game:
    while True:
        g = qe.Game(width, height, walls)
        for _ in range(plies):
            if g.is_terminal():
                break
            g.apply(opener(g, rng))
        if not g.is_terminal():
            return g


def elo(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def net_mover(
    evaluate: NetEvaluator,
    sims: int,
    max_considered: int = MAX_CONSIDERED,
    seed: int = 0,
) -> Mover:
    mcts = qe.MCTS(sims=sims, max_considered=max_considered, seed=seed)

    def move(games: list[qe.Game]) -> list[int]:
        run_search(mcts, games, evaluate)
        return mcts.actions()

    return move


def alphabeta_mover(depth: int, time_ms: int = 0) -> Mover:
    searcher = qe.Searcher(tt_mb=16)

    def move(games: list[qe.Game]) -> list[int]:
        actions = []
        for g in games:
            searcher.clear()
            actions.append(searcher.search(g, max_depth=depth, time_ms=time_ms).action)
        return actions

    return move


def play_pairs(
    board: list[int],
    a: Mover,
    b: Mover,
    n_games: int,
    open_plies: int = 2,
    seed: int = 0,
    opener: Opener = uniform_move,
) -> tuple[int, int, int]:
    width, height, walls = board
    rng = random.Random(seed)
    games: list[qe.Game] = []
    a_side: list[int] = []
    for _ in range(n_games // 2):
        g = opening(width, height, walls, open_plies, rng, opener)
        for side in (0, 1):
            games.append(g.clone())
            a_side.append(side)
    while any(not g.is_terminal() for g in games):
        for mover, is_a in ((a, True), (b, False)):
            turn = [
                i
                for i, g in enumerate(games)
                if not g.is_terminal() and (g.side == a_side[i]) == is_a
            ]
            if turn:
                for i, act in zip(turn, mover([games[i] for i in turn])):
                    games[i].apply(act)
    wins = sum(g.winner() == s for g, s in zip(games, a_side))
    draws = sum(g.winner() < 0 for g in games)
    return wins, draws, len(games) - wins - draws


def net_vs_net(
    a: NetEvaluator,
    b: NetEvaluator,
    n_games: int,
    sims: int,
    max_considered: int = MAX_CONSIDERED,
    open_plies: int = 2,
    seed: int = 0,
    opener: Opener = uniform_move,
) -> tuple[int, int, int]:
    return play_pairs(
        a.board,
        net_mover(a, sims, max_considered, seed),
        net_mover(b, sims, max_considered, seed),
        n_games,
        open_plies,
        seed,
        opener,
    )


def play_match(
    evaluate: NetEvaluator,
    n_games: int,
    sims: int,
    max_considered: int = MAX_CONSIDERED,
    ab_depth: int = 4,
    seed: int = 0,
    open_plies: int = 2,
    opener: Opener = uniform_move,
) -> tuple[int, int, int]:
    return play_pairs(
        evaluate.board,
        net_mover(evaluate, sims, max_considered, seed),
        alphabeta_mover(ab_depth),
        n_games,
        open_plies,
        seed,
        opener,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="net + MCTS against the alpha-beta bot, or against another net"
    )
    p.add_argument("net")
    p.add_argument("--opponent", help="play this net instead of alpha-beta")
    p.add_argument(
        "--mirror", action="store_true", help="mirror averaging for the first net only"
    )
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--max-considered", type=int, default=MAX_CONSIDERED)
    p.add_argument("--ab-depth", type=int, default=4)
    p.add_argument(
        "--ab-time",
        type=int,
        default=0,
        help="alpha-beta ms per move, 0 for fixed depth",
    )
    p.add_argument(
        "--open-plies",
        type=int,
        default=2,
        help="random opening plies",
    )
    p.add_argument(
        "--open-temp",
        type=positive,
        help="temperature for opening plies (default: random)",
    )
    p.add_argument(
        "--board",
        type=int,
        nargs=3,
        metavar=("W", "H", "WALLS"),
        help="play on this board instead of the net's",
    )
    p.add_argument("--book", help="opening book for the first net")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate = NetEvaluator(args.net, mirror=args.mirror)
    if args.board:
        evaluate.board = args.board
    opener = (
        policy_opener(evaluate, args.open_temp)
        if args.open_temp is not None
        else uniform_move
    )
    a = net_mover(evaluate, args.sims, args.max_considered, args.seed)
    if args.book:
        a = with_book(load_book(args.book, evaluate.board), a)
    if args.opponent:
        b = net_mover(
            NetEvaluator(args.opponent), args.sims, args.max_considered, args.seed
        )
        label = f"{args.net} vs {args.opponent}, {args.sims} sims"
    else:
        b = alphabeta_mover(args.ab_depth, args.ab_time)
        label = f"net {args.sims} sims vs alpha-beta depth {args.ab_depth}"
    if args.book:
        label += ", with book"
    t0 = time.time()
    w, d, l = play_pairs(
        evaluate.board, a, b, args.games, args.open_plies, args.seed, opener
    )
    score = (w + 0.5 * d) / (w + d + l)
    print(
        f"{label}: +{w} ={d} -{l}  score {score:.1%}  elo {elo(score):+.0f}  ({time.time() - t0:.1f}s)"
    )


if __name__ == "__main__":
    main()

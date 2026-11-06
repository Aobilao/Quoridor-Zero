import random
from collections.abc import Callable

import numpy as np

from quoridor import quoridor_engine as qe

Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def uniform(num_actions: int) -> Evaluator:
    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(x), num_actions), np.float32), np.zeros(len(x), np.float32)

    return evaluate


def run(mcts: qe.MCTS, games: list[qe.Game], evaluate: Evaluator, noise: bool) -> int:
    mcts.start(games, noise)
    rounds = 0
    while True:
        x = mcts.leaf_features()
        if len(x) == 0:
            return rounds
        mcts.feed(*evaluate(x))
        rounds += 1


def random_position(rng: random.Random, plies: int) -> qe.Game:
    while True:
        g = qe.Game(5, 5, 3)
        for _ in range(plies):
            if g.is_terminal():
                break
            g.apply(rng.choice(g.legal_actions()))
        if not g.is_terminal():
            return g


def check_invariants() -> None:
    rng = random.Random(0)
    games = [random_position(rng, rng.randint(0, 8)) for _ in range(64)]
    sims = 48
    mcts = qe.MCTS(sims=sims, max_considered=16, seed=1)
    rounds = run(mcts, games, uniform(games[0].num_actions), noise=True)
    assert rounds <= sims + 1
    pol = mcts.policies()
    for i, g in enumerate(games):
        mask = g.legal_mask().astype(bool)
        assert abs(pol[i].sum() - 1.0) < 1e-5
        assert np.all(pol[i][~mask] == 0)
        assert mask[mcts.actions()[i]]
        visits = mcts.visits(i)
        assert len(visits) == mask.sum()
        if len(visits) > 1:
            assert sum(n for _, n in visits) == sims


def check_finds_win() -> None:
    g = qe.Game(5, 5, 0)
    for a in (0, 2, 0, 3, 0, 2):
        g.apply(a)
    assert g.pawns[0][0] == 3 and g.side == 0
    mcts = qe.MCTS(sims=16, max_considered=16)
    run(mcts, [g], uniform(g.num_actions), noise=False)
    assert mcts.actions()[0] == 0
    assert mcts.policies()[0][0] > 0.5
    assert mcts.root_values()[0] == 1.0


def check_wins_in_one() -> None:
    rng = random.Random(2)
    searcher = qe.Searcher()
    found = 0
    while found < 30:
        g = random_position(rng, rng.randint(4, 16))
        r = searcher.search(g, max_depth=1, time_ms=0)
        if r.score <= qe.MATE_BOUND:
            continue
        found += 1
        sims = 2 * len(g.legal_actions())
        mcts = qe.MCTS(sims=sims, max_considered=qe.Game(5, 5).num_actions, seed=found)
        run(mcts, [g], uniform(g.num_actions), noise=True)
        child = g.clone()
        child.apply(mcts.actions()[0])
        assert child.winner() == g.side
        assert mcts.root_values()[0] == 1.0


def main() -> None:
    check_invariants()
    check_finds_win()
    check_wins_in_one()
    print("mcts: all checks passed")


if __name__ == "__main__":
    main()

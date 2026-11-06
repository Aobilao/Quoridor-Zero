import random

import numpy as np

from quoridor import quoridor_engine as qe


def check_random_games(n_games: int, width: int, height: int, walls: int) -> None:
    rng = random.Random(0)
    for _ in range(n_games):
        g = qe.Game(width, height, walls)
        while not g.is_terminal():
            actions = g.legal_actions()
            mask = g.legal_mask()
            assert mask.sum() == len(actions)
            assert all(mask[a] for a in actions)
            feats = g.features()
            assert feats.shape == (qe.N_PLANES, height, width)
            assert feats[0].sum() == 1 and feats[1].sum() == 1
            g.apply(rng.choice(actions))


def check_canonical_view() -> None:
    g = qe.Game(5, 5, 3)
    f0 = g.features()
    g.apply(1 if 1 in g.legal_actions() else g.legal_actions()[0])
    g2 = qe.Game(5, 5, 3)
    assert np.array_equal(f0[0], g2.features()[0])
    me_row = int(np.argwhere(g.features()[0])[0][0])
    assert me_row == 0


def check_search() -> None:
    g = qe.Game(5, 3, 3, max_plies=1000)
    r = qe.Searcher(tt_mb=64).search(g, max_depth=100, time_ms=10000)
    assert r.score > qe.MATE_BOUND
    g.apply(r.action)


def check_illegal_rejected() -> None:
    g = qe.Game()
    try:
        g.apply(g.num_actions)
    except ValueError:
        return
    raise AssertionError("illegal action accepted")


def main() -> None:
    check_random_games(50, 5, 5, 4)
    check_random_games(20, 9, 9, 10)
    check_canonical_view()
    check_search()
    check_illegal_rejected()
    g = qe.Game()
    batch = qe.batch_features([g, g.clone()])
    assert batch.shape == (2, qe.N_PLANES, 9, 9)
    print(g)
    print(qe.Searcher().search(g, max_depth=64, time_ms=1000))
    print("python bindings: all checks passed")


if __name__ == "__main__":
    main()

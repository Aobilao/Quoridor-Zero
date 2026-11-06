import argparse
from collections.abc import Callable

import numpy as np
import torch

from quoridor import quoridor_engine as qe
from quoridor.model import load

Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
Mover = Callable[[list[qe.Game]], list[int]]
MAX_CONSIDERED = 16


def positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def mirror_perm(width: int, height: int) -> np.ndarray:
    g = qe.Game(width, height, 0)
    return np.array([g.mirror_action(a) for a in range(g.num_actions)], dtype=np.int64)


def mirror_features(x: torch.Tensor) -> torch.Tensor:
    x = x.flip(3)
    x[:, 2:4, :, :-1] = x[:, 2:4, :, 1:].clone()
    x[:, 2:4, :, -1] = 0
    return x


class NetEvaluator:
    def __init__(
        self,
        path: str,
        device: str | None = None,
        mirror: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, ckpt = load(path, self.device)
        self.board: list[int] = ckpt["board"]
        self.mirror = mirror
        self.perms: dict[tuple[int, int], torch.Tensor] = {}

    def perm(self, width: int, height: int) -> torch.Tensor:
        if (width, height) not in self.perms:
            self.perms[width, height] = torch.from_numpy(mirror_perm(width, height)).to(
                self.device
            )
        return self.perms[width, height]

    @torch.inference_mode()
    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(x).to(self.device)
        logits, value, _ = self.net(t)
        if self.mirror and x.shape[3] % 2 == 1:
            logits_m, value_m, _ = self.net(mirror_features(t))
            logits = (logits + logits_m[:, self.perm(x.shape[3], x.shape[2])]) / 2
            value = (value + value_m) / 2
        return logits.float().cpu().numpy(), value.float().cpu().numpy()


def run_search(
    mcts: qe.MCTS,
    games: list[qe.Game],
    evaluate: Evaluator,
    noise: bool = False,
) -> None:
    mcts.start(games, noise)
    while True:
        x = mcts.leaf_features()
        if len(x) == 0:
            return
        mcts.feed(*evaluate(x))

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass, field, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from matplotlib.patches import Circle

from quoridor import quoridor_engine as qe
from quoridor.agent import MAX_CONSIDERED, NetEvaluator, run_search
from quoridor.arena import opening
from quoridor.play import notation, parse

COLORS = ("#2a78d6", "#e34948")
PLAY_SIMS = 64
NAMES = ("Blue", "Red")


@dataclass
class Position:
    width: int
    height: int
    start_walls: int
    side: int
    pawns: list[tuple[int, int]]
    walls_left: list[int]
    hw: set[tuple[int, int]] = field(default_factory=set)
    vw: set[tuple[int, int]] = field(default_factory=set)

    @staticmethod
    def of(g: qe.Game, start_walls: int) -> "Position":
        return Position(
            g.width,
            g.height,
            start_walls,
            g.side,
            [(r, c) for r, c in g.pawns],
            list(g.walls_left),
            {(r, c) for k, r, c in g.walls if k == "h"},
            {(r, c) for k, r, c in g.walls if k == "v"},
        )

    def goal_row(self, player: int) -> int:
        return self.height - 1 if player == 0 else 0

    def open_between(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        (r1, c1), (r2, _) = sorted((a, b))
        if r1 != r2:
            return (r1, c1) not in self.hw and (r1, c1 - 1) not in self.hw
        return (r1, c1) not in self.vw and (r1 - 1, c1) not in self.vw

    def distances(self, player: int) -> np.ndarray:
        dist = np.full((self.height, self.width), -1, dtype=np.int64)
        queue: deque[tuple[int, int]] = deque()
        for c in range(self.width):
            dist[self.goal_row(player), c] = 0
            queue.append((self.goal_row(player), c))
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (
                    0 <= nr < self.height
                    and 0 <= nc < self.width
                    and dist[nr, nc] < 0
                    and self.open_between((r, c), (nr, nc))
                ):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        return dist

    def flip(self, r: int, rows: int) -> int:
        return r if self.side == 0 else rows - 1 - r

    def features(self) -> np.ndarray:
        h, w, n = self.height, self.width, self.width * self.height
        me, op = self.side, 1 - self.side
        x = np.zeros((8, h, w), dtype=np.float32)
        x[0, self.flip(self.pawns[me][0], h), self.pawns[me][1]] = 1
        x[1, self.flip(self.pawns[op][0], h), self.pawns[op][1]] = 1
        for r, c in self.hw:
            x[2, self.flip(r, h - 1), c] = 1
        for r, c in self.vw:
            x[3, self.flip(r, h - 1), c] = 1
        x[4] = self.walls_left[me] / self.start_walls if self.start_walls else 0
        x[5] = self.walls_left[op] / self.start_walls if self.start_walls else 0
        for plane, player in ((6, me), (7, op)):
            d = self.distances(player)
            m = np.where(d < 0, 1.0, d / n).astype(np.float32)
            x[plane] = m if self.side == 0 else m[::-1]
        return x


def build_game(
    board: list[int],
    moves: list[str],
    plies: int,
    open_plies: int,
    seed: int,
    evaluate: NetEvaluator,
    sims: int,
) -> qe.Game:
    if moves:
        g = qe.Game(board[0], board[1], board[2])
        for text in moves:
            a = parse(g, text)
            if a is None:
                raise SystemExit(
                    f"illegal move {text}. Legal moves: {' '.join(sorted(notation(g, x) for x in g.legal_actions()))}"
                )
            g.apply(a)
        return g
    g = opening(board[0], board[1], board[2], open_plies, random.Random(seed))
    mcts = qe.MCTS(sims=sims, max_considered=MAX_CONSIDERED, seed=seed)
    for _ in range(plies):
        run_search(mcts, [g], evaluate)
        child = g.clone()
        child.apply(mcts.actions()[0])
        if child.is_terminal():
            break
        g = child
    return g


def values(evaluate: NetEvaluator, xs: list[np.ndarray]) -> np.ndarray:
    return evaluate(np.stack(xs))[1]


def to_board(planes: np.ndarray, side: int) -> np.ndarray:
    if side == 1:
        flipped = planes[:, ::-1].copy()
        flipped[2:4, :-1] = planes[2:4, :-1][:, ::-1]
        flipped[2:4, -1] = 0
        planes = flipped
    return planes.sum(0)


def saliency(evaluate: NetEvaluator, pos: Position) -> np.ndarray:
    x = torch.from_numpy(pos.features()[None]).to(evaluate.device).requires_grad_(True)
    _, value, _ = evaluate.net(x)
    value.sum().backward()
    assert x.grad is not None
    return to_board(x.grad[0].abs().cpu().numpy(), pos.side)


def draw_board(ax: Axes, pos: Position, title: str) -> None:
    for r in range(pos.height + 1):
        ax.plot([0, pos.width], [r, r], color="#d5d8d4", linewidth=0.8, zorder=1)
    for c in range(pos.width + 1):
        ax.plot([c, c], [0, pos.height], color="#d5d8d4", linewidth=0.8, zorder=1)
    for p, (r, c) in enumerate(pos.pawns):
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.32, color=COLORS[p], zorder=5))
    ax.set_xlim(-0.2, pos.width + 0.2)
    ax.set_ylim(-0.2, pos.height + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks(
        [c + 0.5 for c in range(pos.width)],
        [chr(ord("a") + c) for c in range(pos.width)],
    )
    ax.set_yticks(
        [r + 0.5 for r in range(pos.height)], [str(r + 1) for r in range(pos.height)]
    )
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title)


def wall_segment(kind: str, r: int, c: int) -> tuple[list[float], list[float]]:
    if kind == "h":
        return [c + 0.05, c + 1.95], [r + 1, r + 1]
    return [c + 1, c + 1], [r + 0.05, r + 1.95]


def draw_walls(
    ax: Axes, pos: Position, colors: dict[tuple[str, int, int], str] | None = None
) -> None:
    for kind, slots in (("h", pos.hw), ("v", pos.vw)):
        for r, c in slots:
            xs, ys = wall_segment(kind, r, c)
            color = colors.get((kind, r, c), "#3a3a3a") if colors else "#3a3a3a"
            ax.plot(xs, ys, color=color, linewidth=6, solid_capstyle="butt", zorder=4)


def explain(
    g: qe.Game, evaluate: NetEvaluator, label: str, sims: int, seed: int = 0
) -> Figure:
    pos = Position.of(g, evaluate.board[2])
    if not np.array_equal(pos.features(), g.features()):
        raise ValueError("rebuilt input planes differ from the engine's")
    me = pos.side
    base = float(values(evaluate, [pos.features()])[0])

    walls = [("h", r, c) for r, c in pos.hw] + [("v", r, c) for r, c in pos.vw]
    removed = []
    for kind, r, c in walls:
        q = replace(pos, hw=set(pos.hw), vw=set(pos.vw))
        (q.hw if kind == "h" else q.vw).discard((r, c))
        removed.append(q.features())
    deltas = values(evaluate, removed) - base if removed else np.zeros(0)

    cells = [
        (r, c)
        for r in range(pos.height)
        for c in range(pos.width)
        if r != pos.goal_row(me) and (r, c) != pos.pawns[1 - me]
    ]
    moved = []
    for r, c in cells:
        q = replace(pos, pawns=list(pos.pawns))
        q.pawns[me] = (r, c)
        moved.append(q.features())
    pawn_map = np.full((pos.height, pos.width), np.nan)
    for (r, c), v in zip(cells, values(evaluate, moved)):
        pawn_map[r, c] = 50 * (v + 1)
    pawn_map[pos.goal_row(me), :] = 100

    sal = saliency(evaluate, pos)

    mcts = qe.MCTS(sims=sims, max_considered=MAX_CONSIDERED, seed=seed)
    run_search(mcts, [g], evaluate)
    visits = {a: n for a, n in mcts.visits(0) if n > 0}
    best = mcts.actions()[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12.5))
    (a1, a2), (a3, a4) = axes

    lim = max(0.05, float(np.abs(deltas).max())) if len(deltas) else 0.05
    norm = TwoSlopeNorm(0, -lim, lim)
    cmap = plt.get_cmap("PRGn_r")
    draw_board(
        a1,
        pos,
        f"Walls: change in {NAMES[me]}'s value if removed\n(green: the wall helps {NAMES[me]}, purple: it hurts)",
    )
    draw_walls(a1, pos, {w: to_hex(cmap(norm(d))) for w, d in zip(walls, deltas)})
    for (kind, r, c), d in zip(walls, deltas):
        xs, ys = wall_segment(kind, r, c)
        a1.text(
            sum(xs) / 2,
            sum(ys) / 2 + (0.28 if kind == "h" else 0),
            f"{d:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
            color="#16181a",
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=a1,
        fraction=0.046,
        pad=0.04,
        label="value change",
    )

    draw_board(a2, pos, f"{NAMES[me]}'s win chance with its pawn on each square")
    im = a2.imshow(
        pawn_map,
        origin="lower",
        extent=(0, pos.width, 0, pos.height),
        cmap="viridis",
        vmin=0,
        vmax=100,
        zorder=0,
    )
    draw_walls(a2, pos)
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04, label="win chance (%)")

    draw_board(a3, pos, "Gradient saliency: |d value / d input|, summed over planes")
    im = a3.imshow(
        sal,
        origin="lower",
        extent=(0, pos.width, 0, pos.height),
        cmap="magma_r",
        zorder=0,
    )
    draw_walls(a3, pos)
    fig.colorbar(im, ax=a3, fraction=0.046, pad=0.04, label="sensitivity")

    draw_board(a4, pos, f"Search: visits per move ({sims} sims)")
    draw_walls(a4, pos)
    top = max(visits.values()) if visits else 1
    vnorm = Normalize(0, top)
    vmap = plt.get_cmap("Oranges")
    for a, n in sorted(visits.items(), key=lambda t: t[1]):
        kind, r, c = g.describe_action(a)
        color = vmap(0.25 + 0.75 * vnorm(n))
        if kind == "p":
            a4.add_patch(
                Circle(
                    (c + 0.5, r + 0.5),
                    0.12 + 0.25 * n / top,
                    color=color,
                    zorder=6,
                    alpha=0.9,
                )
            )
        else:
            xs, ys = wall_segment(kind, r, c)
            a4.plot(
                xs,
                ys,
                color=color,
                linewidth=3 + 5 * n / top,
                solid_capstyle="butt",
                zorder=6,
                alpha=0.9,
            )
        if n >= 0.15 * top:
            xs, ys = ([c + 0.5], [r + 0.5]) if kind == "p" else wall_segment(kind, r, c)
            a4.text(
                sum(xs) / len(xs),
                sum(ys) / len(ys) - 0.32,
                str(n),
                ha="center",
                fontsize=8,
                zorder=7,
            )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=vnorm, cmap=vmap),
        ax=a4,
        fraction=0.046,
        pad=0.04,
        label="visits",
    )

    fig.suptitle(
        f"{label}, ply {g.ply}, {NAMES[me]} to move, walls left {NAMES[0]} {pos.walls_left[0]} / {NAMES[1]} {pos.walls_left[1]}\n"
        f"net win chance for {NAMES[me]}: {50 * (base + 1):.0f}%, search value {50 * (mcts.root_values()[0] + 1):.0f}%, chosen move {notation(g, best)}"
    )
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        description="show what the net reacts to in one position"
    )
    p.add_argument("--net", default="runs/sp9h/iter040.pt")
    p.add_argument(
        "--moves",
        nargs="*",
        default=[],
        help="moves to play first, e.g. e2 e8 d3h",
    )
    p.add_argument(
        "--plies",
        type=int,
        default=24,
        help="plies the net plays (without --moves)",
    )
    p.add_argument("--open-plies", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--out", default="plots/explain.png")
    args = p.parse_args()

    evaluate = NetEvaluator(args.net)
    board = evaluate.board
    g = build_game(
        board, args.moves, args.plies, args.open_plies, args.seed, evaluate, PLAY_SIMS
    )
    fig = explain(g, evaluate, args.net, args.sims, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

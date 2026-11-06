import argparse
import sys
from typing import Literal

from quoridor import quoridor_engine as qe

NAMES = ("Blue", "Red")

HELP = """commands:
  c2        move your pawn to square c2
  b3h b3v   place a horizontal / vertical wall at the top-right corner of b3
  moves     list legal moves
  hint      ask the bot for a move
  undo      take back your last move
  help      show this text
  quit      exit"""


def square(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{r + 1}"


def notation(game: qe.Game, action: int) -> str:
    kind, r, c = game.describe_action(action)
    return square(r, c) if kind == "p" else square(r, c) + kind


def render(game: qe.Game) -> str:
    w, h = game.width, game.height
    rows, cols = 2 * h - 1, 4 * w - 3
    canvas = [[" "] * cols for _ in range(rows)]
    for r in range(h):
        for c in range(w):
            canvas[2 * (h - 1 - r)][4 * c] = "·"
    for player, (r, c) in enumerate(game.pawns):
        canvas[2 * (h - 1 - r)][4 * c] = NAMES[player][0]
    for kind, r, c in game.walls:
        if kind == "h":
            y = 2 * (h - 1 - r) - 1
            for x in range(4 * c, 4 * c + 5):
                canvas[y][x] = "━"
        else:
            x = 4 * c + 2
            for y in range(2 * (h - 1 - r) - 2, 2 * (h - 1 - r) + 1):
                canvas[y][x] = "┃"
    lines = []
    for y, row in enumerate(canvas):
        label = f"{h - y // 2:>2} " if y % 2 == 0 else "   "
        lines.append(label + "".join(row))
    lines.append("   " + "".join(chr(ord("a") + c).ljust(4) for c in range(w)).rstrip())
    w1, w2 = game.walls_left
    lines.append(f"walls left  Blue (B): {w1}  Red (R): {w2}")
    return "\n".join(lines)


def describe_score(score: int) -> str:
    if score > qe.MATE_BOUND or score < -qe.MATE_BOUND:
        n = qe.WIN - abs(score)
        verb = "wins" if score > 0 else "loses"
        return f"{verb} in {n} {'ply' if n == 1 else 'plies'}"
    return f"eval {score:+d}"


def parse(game: qe.Game, text: str) -> int | None:
    for a in game.legal_actions():
        if notation(game, a) == text:
            return a
    return None


def bot_move(
    game: qe.Game, searcher: qe.Searcher, depth: int, time_ms: int
) -> tuple[int, str]:
    r = searcher.search(game, max_depth=depth, time_ms=time_ms)
    info = f"depth {r.depth}, {describe_score(r.score)}, {r.nodes} nodes"
    return r.action, info


def announce_end(game: qe.Game) -> None:
    print(render(game))
    winner = game.winner()
    print("draw by move limit" if winner < 0 else f"{NAMES[winner]} wins")


def human_turn(
    game: qe.Game, searcher: qe.Searcher, args: argparse.Namespace
) -> Literal["quit", "undo"] | int:
    while True:
        text = input(f"{NAMES[game.side]}> ").strip().lower()
        if text in ("quit", "exit", "q"):
            return "quit"
        if text == "undo":
            return "undo"
        if text == "help":
            print(HELP)
        elif text == "moves":
            print(" ".join(sorted(notation(game, a) for a in game.legal_actions())))
        elif text == "hint":
            action, info = bot_move(game, searcher, args.ab_depth, args.ab_time)
            print(f"hint: {notation(game, action)} ({info})")
        else:
            action = parse(game, text)
            if action is not None:
                return action
            print("not a legal move (type 'moves' to list them)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="play Quoridor in the terminal")
    p.add_argument(
        "--board",
        type=int,
        nargs=3,
        default=[9, 9, 10],
        metavar=("W", "H", "WALLS"),
    )
    who = p.add_mutually_exclusive_group()
    who.add_argument(
        "--you",
        choices=["blue", "red"],
        default="blue",
        help="your side (blue moves first)",
    )
    who.add_argument("--watch", action="store_true", help="bot vs bot")
    who.add_argument("--hotseat", action="store_true", help="two humans, no bot")
    p.add_argument(
        "--ab-time", type=int, default=2000, help="alpha-beta bot time per move in ms"
    )
    p.add_argument(
        "--ab-depth", type=int, default=64, help="alpha-beta bot max search depth"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    game = qe.Game(*args.board)
    human = NAMES.index(args.you.capitalize())
    searcher = qe.Searcher(tt_mb=128)
    history: list[qe.Game] = []
    print(HELP)
    while not game.is_terminal():
        print(render(game))
        if args.hotseat or (not args.watch and game.side == human):
            choice = human_turn(game, searcher, args)
            if choice == "quit":
                return
            if choice == "undo":
                steps = 1 if args.hotseat else 2
                if len(history) < steps:
                    print("nothing to undo")
                    continue
                for _ in range(steps):
                    game = history.pop()
                continue
            action = choice
        else:
            action, info = bot_move(game, searcher, args.ab_depth, args.ab_time)
            print(f"bot {NAMES[game.side]} plays {notation(game, action)} ({info})")
        history.append(game.clone())
        game.apply(action)
    announce_end(game)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

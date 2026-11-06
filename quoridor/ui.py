import argparse
import base64
import glob
import io
import json
import os
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Protocol

import matplotlib.pyplot as plt
import torch

from quoridor import quoridor_engine as qe
from quoridor.agent import MAX_CONSIDERED, NetEvaluator, run_search
from quoridor.book import Book, load_book
from quoridor.explain import explain
from quoridor.play import describe_score, notation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BEST = "runs/sp9h/iter040.pt"
SIMS = 512
BOOK = "books/book_9x9w10.json"


class Bot(Protocol):
    def choose(self, game: qe.Game) -> tuple[int, str, float | None]: ...


class AlphaBetaBot:
    def __init__(self, time_ms: int) -> None:
        self.time_ms = time_ms
        self.searcher = qe.Searcher(tt_mb=128)

    def choose(self, game: qe.Game) -> tuple[int, str, float | None]:
        r = self.searcher.search(game, max_depth=64, time_ms=self.time_ms)
        proven = (
            1.0
            if r.score > qe.MATE_BOUND
            else 0.0
            if r.score < -qe.MATE_BOUND
            else None
        )
        return (
            r.action,
            f"alpha-beta depth {r.depth}, {describe_score(r.score)}",
            proven,
        )


class NetBot:
    def __init__(
        self, path: str, evaluate: NetEvaluator, sims: int, book: Book
    ) -> None:
        self.path = path
        self.evaluate = evaluate
        self.sims = sims
        self.book = book
        self.mcts = qe.MCTS(sims=sims, max_considered=MAX_CONSIDERED)

    def choose(self, game: qe.Game) -> tuple[int, str, float | None]:
        if game.hash() in self.book:
            action, value = self.book[game.hash()]
            chance = (value + 1) / 2
            return action, f"opening book, win chance {100 * chance:.0f}%", chance
        run_search(self.mcts, [game], self.evaluate)
        chance = (self.mcts.root_values()[0] + 1) / 2
        return (
            self.mcts.actions()[0],
            f"net {self.sims} sims, win chance {100 * chance:.0f}%",
            chance,
        )


@dataclass
class Session:
    game: qe.Game
    board: list[int]
    humans: list[int]
    bots: dict[int, Bot]
    history: list[qe.Game] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    lasts: list[tuple[str, int, int] | None] = field(default_factory=list)
    evals: list[tuple[float, int] | None] = field(default_factory=list)
    info: str = ""
    resigned: int | None = None

    def over(self) -> bool:
        return self.game.is_terminal() or self.resigned is not None

    def apply(self, action: int, evaluation: tuple[float, int] | None = None) -> None:
        self.evals.append(evaluation)
        self.history.append(self.game.clone())
        self.moves.append(notation(self.game, action))
        self.lasts.append(self.game.describe_action(action))
        self.game.apply(action)

    def undo(self) -> None:
        if self.resigned is not None:
            self.resigned = None
            self.info = ""
            return
        while self.history:
            self.game = self.history.pop()
            self.moves.pop()
            self.lasts.pop()
            self.evals.pop()
            if not self.humans or self.game.side in self.humans:
                break
        self.info = ""

    def position(self, ply: int) -> qe.Game:
        if not 0 <= ply <= len(self.moves):
            raise ValueError(f"no position at ply {ply}")
        return self.game if ply == len(self.moves) else self.history[ply]

    def win(self, ply: int) -> dict[str, Any] | None:
        views: list[float | None] = [None, None]
        for e in reversed(self.evals[:ply]):
            if e is not None and views[e[1]] is None:
                views[e[1]] = e[0]
        if not self.position(ply).is_terminal() and views == [None, None]:
            return None
        return {"views": views}

    def state(self, view: int | None = None) -> dict[str, Any]:
        ply = len(self.moves) if view is None else view
        g = self.position(ply)
        live = ply == len(self.moves)
        resigned = self.resigned if live else None
        over = g.is_terminal() or resigned is not None
        return {
            "width": g.width,
            "height": g.height,
            "side": g.side,
            "ply": g.ply,
            "max_plies": g.max_plies,
            "pawns": g.pawns,
            "walls": g.walls,
            "walls_left": g.walls_left,
            "legal": []
            if over or not live
            else [[a, *g.describe_action(a)] for a in g.legal_actions()],
            "terminal": over,
            "winner": g.winner() if resigned is None else 1 - resigned,
            "resigned": resigned,
            "humans": self.humans,
            "moves": self.moves,
            "last": self.lasts[ply - 1] if ply > 0 else None,
            "info": self.info if live else "",
            "win": self.win(ply),
            "can_undo": bool(self.history),
            "view": ply,
            "live": live,
        }


class App:
    def __init__(self) -> None:
        self.nets: dict[str, list[int]] = {}
        self.evaluators: dict[str, NetEvaluator] = {}
        paths = sorted(glob.glob("runs/*/*.pt", root_dir=ROOT))
        for path in paths:
            ckpt = torch.load(os.path.join(ROOT, path), map_location="cpu")
            self.nets[path] = ckpt["board"]
        if BEST in self.nets:
            board = self.nets[BEST]
            bot: Bot = NetBot(BEST, self.evaluator(BEST), SIMS, self.book_for(BEST))
        else:
            board, bot = [5, 5, 3], AlphaBetaBot(1000)
        self.session = Session(qe.Game(*board), board, [0], {1: bot})

    def book_for(self, path: str) -> Book:
        full = os.path.join(ROOT, BOOK)
        if not os.path.exists(full):
            return {}
        try:
            return load_book(full, self.nets[path])
        except ValueError:
            return {}

    def evaluator(self, path: str) -> NetEvaluator:
        if path not in self.evaluators:
            self.evaluators[path] = NetEvaluator(os.path.join(ROOT, path))
        return self.evaluators[path]

    def config(self) -> dict[str, Any]:
        return {
            "nets": [{"path": p, "board": b} for p, b in self.nets.items()],
            "best": BEST,
            "sims": SIMS,
        }

    def make_bot(self, spec: dict[str, Any]) -> Bot:
        if spec["kind"] == "net":
            return NetBot(
                spec["path"],
                self.evaluator(spec["path"]),
                int(spec["sims"]),
                self.book_for(spec["path"]),
            )
        return AlphaBetaBot(int(spec["time_ms"]))

    def new(self, body: dict[str, Any]) -> dict[str, Any]:
        specs: list[dict[str, Any] | None] = body["bots"]
        if len(specs) != 2:
            raise ValueError("bots must list one entry per side")
        width, height, walls = (
            int(body["width"]),
            int(body["height"]),
            int(body["walls"]),
        )
        boards = set()
        for spec in specs:
            if spec and spec["kind"] == "net":
                if spec["path"] not in self.nets:
                    raise ValueError("unknown net")
                boards.add(tuple(self.nets[spec["path"]]))
        if len(boards) > 1:
            raise ValueError("the two nets were trained on different boards")
        if boards:
            width, height, walls = boards.pop()
        if not (3 <= width <= 9 and 3 <= height <= 9 and 0 <= walls <= 20):
            raise ValueError("board must be 3 to 9 squares per side, walls 0 to 20")
        bots = {side: self.make_bot(spec) for side, spec in enumerate(specs) if spec}
        humans = [side for side, spec in enumerate(specs) if not spec]
        self.session = Session(
            qe.Game(width, height, walls), [width, height, walls], humans, bots
        )
        return self.session.state()

    def move(self, body: dict[str, Any]) -> dict[str, Any]:
        s = self.session
        if s.over() or s.game.side not in s.humans:
            raise ValueError("not your turn")
        s.apply(int(body["action"]))
        return s.state()

    def bot_move(self) -> dict[str, Any]:
        s = self.session
        if s.over() or s.game.side not in s.bots:
            raise ValueError("not the bot's turn")
        side = s.game.side
        bot = s.bots[side]
        action, info, chance = bot.choose(s.game)
        s.info = f"{('Blue', 'Red')[side]}: {info}"
        evaluation = (
            None if chance is None else (chance if side == 0 else 1.0 - chance, side)
        )
        s.apply(action, evaluation)
        return s.state()

    def undo(self) -> dict[str, Any]:
        self.session.undo()
        return self.session.state()

    def resign(self) -> dict[str, Any]:
        s = self.session
        if s.over():
            raise ValueError("the game is over")
        if not s.humans:
            raise ValueError("no human player to resign")
        side = s.humans[0] if len(s.humans) == 1 else s.game.side
        s.resigned = side
        s.info = f"{('Blue', 'Red')[side]} resigns"
        return s.state()

    def view(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.session.state(int(body["ply"]))

    def inspect_net(self, game: qe.Game) -> tuple[str, int, str]:
        s = self.session
        me, names = game.side, ("Blue", "Red")
        for side in (me, 1 - me):
            bot = s.bots.get(side)
            if isinstance(bot, NetBot):
                who = (
                    "the bot to move"
                    if side == me
                    else f"{names[side]}'s bot, from {names[me]}'s side"
                )
                return (
                    bot.path,
                    bot.sims,
                    f"{names[side]}'s net with its own {bot.sims} sims ({who})",
                )
        matching = [p for p, b in self.nets.items() if b == s.board]
        if not matching:
            raise ValueError(
                f"no net for {s.board[0]}x{s.board[1]} with {s.board[2]} walls"
            )
        path = BEST if BEST in matching else matching[-1]
        return (
            path,
            SIMS,
            f"no net bot in this game, so the default net with {SIMS} sims",
        )

    def explain(self, body: dict[str, Any]) -> dict[str, Any]:
        s = self.session
        ply = body.get("ply")
        game = s.game if ply is None else s.position(int(ply))
        if game.is_terminal():
            raise ValueError("the game is over")
        path, sims, note = self.inspect_net(game)
        fig = explain(game, self.evaluator(path), path, sims)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return {
            "net": path,
            "sims": sims,
            "note": note,
            "image": base64.b64encode(buf.getvalue()).decode(),
        }


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def reply(self, payload: dict[str, Any], code: int = 200) -> None:
            self.send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "ui.html"), "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/config":
                self.reply(app.config())
            elif self.path == "/api/state":
                self.reply(app.session.state())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            routes = {
                "/api/new": lambda: app.new(body),
                "/api/move": lambda: app.move(body),
                "/api/bot": app.bot_move,
                "/api/undo": app.undo,
                "/api/resign": app.resign,
                "/api/view": lambda: app.view(body),
                "/api/explain": lambda: app.explain(body),
            }
            if self.path not in routes:
                self.send(404, b"not found", "text/plain")
                return
            try:
                self.reply(routes[self.path]())
            except (ValueError, KeyError, TypeError) as e:
                self.reply({"error": str(e)}, 400)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="play Quoridor against the bots in a browser"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = App()
    server = HTTPServer((args.host, args.port), make_handler(app))
    print(
        f"open http://localhost:{args.port}  ({len(app.nets)} nets found, ctrl-c to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

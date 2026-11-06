#include "engine.hpp"
#include "mcts.hpp"
#include "search.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using namespace qd;

static std::vector<int> legal_actions(const Game& g) {
    MoveList ml;
    g.legal_moves(ml);
    std::vector<int> out(ml.n);
    for (int i = 0; i < ml.n; i++)
        out[i] = g.action_of(ml.m[i]);
    return out;
}

static py::array_t<uint8_t> legal_mask(const Game& g) {
    py::array_t<uint8_t> arr(g.num_actions());
    uint8_t* p = arr.mutable_data();
    std::fill(p, p + g.num_actions(), 0);
    MoveList ml;
    g.legal_moves(ml);
    for (int i = 0; i < ml.n; i++)
        p[g.action_of(ml.m[i])] = 1;
    return arr;
}

static void apply_action(Game& g, int a) {
    if (g.is_terminal())
        throw std::invalid_argument("game is over");
    if (a < 0 || a >= g.num_actions())
        throw std::invalid_argument("action out of range");
    int m = g.move_of(a);
    MoveList ml;
    g.legal_moves(ml);
    for (int i = 0; i < ml.n; i++)
        if (ml.m[i] == m && g.action_of(m) == a) {
            g.apply(m);
            return;
        }
    throw std::invalid_argument("illegal action");
}

using Placement = std::tuple<std::string, int, int>;

static Placement describe_action(const Game& g, int a) {
    if (a < 0 || a >= g.num_actions())
        throw std::invalid_argument("action out of range");
    int m = g.move_of(a);
    if (m < H_BASE)
        return {"p", m / g.W, m % g.W};
    bool horiz = m < V_BASE;
    int s = m - (horiz ? H_BASE : V_BASE);
    return {horiz ? "h" : "v", s / g.SW, s % g.SW};
}

static std::vector<Placement> walls(const Game& g) {
    std::vector<Placement> out;
    for (int s = 0; s < g.S; s++) {
        if ((g.hw >> s) & 1)
            out.emplace_back("h", s / g.SW, s % g.SW);
        if ((g.vw >> s) & 1)
            out.emplace_back("v", s / g.SW, s % g.SW);
    }
    return out;
}

static py::array_t<float> features(const Game& g) {
    py::array_t<float> arr(std::vector<py::ssize_t>{N_PLANES, g.H, g.W});
    g.features(arr.mutable_data());
    return arr;
}

static py::array_t<float> batch_features(const std::vector<Game>& games) {
    if (games.empty())
        throw std::invalid_argument("empty batch");
    const int h = games[0].H, w = games[0].W;
    const py::ssize_t n = py::ssize_t(games.size());
    py::array_t<float> arr(std::vector<py::ssize_t>{n, N_PLANES, h, w});
    float* p = arr.mutable_data();
    for (py::ssize_t i = 0; i < n; i++) {
        if (games[i].H != h || games[i].W != w)
            throw std::invalid_argument("mixed board sizes");
        games[i].features(p + i * N_PLANES * h * w);
    }
    return arr;
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static const Game& mcts_root(const BatchMCTS& b) {
    if (b.size() == 0)
        throw std::invalid_argument("call start() first");
    return b.tree(0).root_game();
}

static void mcts_start(BatchMCTS& b, const std::vector<Game>& games, bool noise) {
    if (games.empty())
        throw std::invalid_argument("empty batch");
    for (const Game& g : games) {
        if (g.W != games[0].W || g.H != games[0].H)
            throw std::invalid_argument("mixed board sizes");
        if (g.is_terminal())
            throw std::invalid_argument("terminal root");
    }
    b.start(games, noise);
}

static py::array_t<float> mcts_leaf_features(BatchMCTS& b) {
    const int h = mcts_root(b).H, w = mcts_root(b).W;
    const int k = b.prepare();
    py::array_t<float> arr(std::vector<py::ssize_t>{k, N_PLANES, h, w});
    float* p = arr.mutable_data();
    for (int i = 0; i < k; i++)
        b.leaf_game(i).features(p + i * N_PLANES * h * w);
    return arr;
}

static void mcts_feed(BatchMCTS& b, FloatArray logits, FloatArray values) {
    const int a = mcts_root(b).num_actions();
    const py::ssize_t k = py::ssize_t(b.pending().size());
    if (logits.ndim() != 2 || logits.shape(0) != k || logits.shape(1) != a)
        throw std::invalid_argument("logits must have shape (pending, num_actions)");
    if (values.ndim() != 1 || values.shape(0) != k)
        throw std::invalid_argument("values must have shape (pending,)");
    b.feed(logits.data(), values.data(), a);
}

static py::array_t<float> mcts_policies(const BatchMCTS& b) {
    const int a = mcts_root(b).num_actions();
    py::array_t<float> arr(std::vector<py::ssize_t>{b.size(), a});
    float* p = arr.mutable_data();
    for (int i = 0; i < b.size(); i++)
        b.tree(i).policy(p + i * a, a);
    return arr;
}

PYBIND11_MODULE(quoridor_engine, m) {
    m.attr("N_PLANES") = N_PLANES;
    m.attr("N_PAWN_ACTIONS") = N_PAWN_ACTIONS;
    m.attr("WIN") = WIN;
    m.attr("MATE_BOUND") = MATE_BOUND;

    py::class_<Game>(m, "Game")
        .def(py::init<int, int, int, int>(), py::arg("width") = 9, py::arg("height") = 9,
             py::arg("walls") = 10, py::arg("max_plies") = 0)
        .def_readonly("width", &Game::W)
        .def_readonly("height", &Game::H)
        .def_readonly("side", &Game::side)
        .def_readonly("ply", &Game::ply)
        .def_readonly("max_plies", &Game::max_plies)
        .def_property_readonly("num_actions", &Game::num_actions)
        .def_property_readonly(
            "walls_left",
            [](const Game& g) { return py::make_tuple(g.walls_left[0], g.walls_left[1]); })
        .def_property_readonly("pawns",
                               [](const Game& g) {
                                   return py::make_tuple(
                                       py::make_tuple(g.pawn[0] / g.W, g.pawn[0] % g.W),
                                       py::make_tuple(g.pawn[1] / g.W, g.pawn[1] % g.W));
                               })
        .def_property_readonly("walls", &walls)
        .def("describe_action", &describe_action, py::arg("action"))
        .def("legal_actions", &legal_actions)
        .def("legal_mask", &legal_mask)
        .def("apply", &apply_action, py::arg("action"))
        .def("is_terminal", &Game::is_terminal)
        .def("winner", &Game::winner)
        .def("distance", &Game::dist_to_goal, py::arg("player"))
        .def("features", &features)
        .def("mirror_action", &Game::mirror_action, py::arg("action"))
        .def("hash", &Game::hash)
        .def("clone", [](const Game& g) { return Game(g); })
        .def("__copy__", [](const Game& g) { return Game(g); })
        .def("__deepcopy__", [](const Game& g, py::dict) { return Game(g); })
        .def("__str__", &Game::to_string);

    m.def("batch_features", &batch_features, py::arg("games"));

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("action", &SearchResult::action)
        .def_readonly("score", &SearchResult::score)
        .def_readonly("depth", &SearchResult::depth)
        .def_readonly("nodes", &SearchResult::nodes)
        .def("__repr__", [](const SearchResult& r) {
            return "SearchResult(action=" + std::to_string(r.action) +
                   ", score=" + std::to_string(r.score) + ", depth=" + std::to_string(r.depth) +
                   ", nodes=" + std::to_string(r.nodes) + ")";
        });

    py::class_<Searcher>(m, "Searcher")
        .def(py::init<int>(), py::arg("tt_mb") = 64)
        .def_readwrite("wall_weight", &Searcher::wall_weight)
        .def("search", &Searcher::search, py::arg("game"), py::arg("max_depth") = 64,
             py::arg("time_ms") = 1000, py::call_guard<py::gil_scoped_release>())
        .def("evaluate", &Searcher::evaluate, py::arg("game"))
        .def("clear", &Searcher::clear);

    py::class_<BatchMCTS>(m, "MCTS")
        .def(py::init([](int sims, int max_considered, float c_puct, uint64_t seed) {
                 MCTSConfig c;
                 c.sims = sims;
                 c.max_considered = max_considered;
                 c.c_puct = c_puct;
                 return new BatchMCTS(c, seed);
             }),
             py::arg("sims") = 64, py::arg("max_considered") = 16, py::arg("c_puct") = 1.25f,
             py::arg("seed") = 0)
        .def("start", &mcts_start, py::arg("games"), py::arg("noise") = false)
        .def("leaf_features", &mcts_leaf_features)
        .def("feed", &mcts_feed, py::arg("logits"), py::arg("values"))
        .def("actions",
             [](const BatchMCTS& b) {
                 std::vector<int> out;
                 for (int i = 0; i < b.size(); i++)
                     out.push_back(b.tree(i).best_action());
                 return out;
             })
        .def("policies", &mcts_policies)
        .def("root_values",
             [](const BatchMCTS& b) {
                 std::vector<float> out;
                 for (int i = 0; i < b.size(); i++)
                     out.push_back(b.tree(i).root_value());
                 return out;
             })
        .def("raw_values",
             [](const BatchMCTS& b) {
                 std::vector<float> out;
                 for (int i = 0; i < b.size(); i++)
                     out.push_back(b.tree(i).root_raw_value());
                 return out;
             })
        .def(
            "visits", [](const BatchMCTS& b, int i) { return b.tree(i).root_visits(); },
            py::arg("index"));
}

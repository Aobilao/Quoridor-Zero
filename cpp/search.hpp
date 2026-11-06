#pragma once
#include "engine.hpp"

#include <chrono>
#include <vector>

namespace qd {

constexpr int WIN = 1000000;
constexpr int MATE_BOUND = WIN - 10000;
constexpr int INF = WIN + 1;

struct SearchResult {
    int action = -1;
    int move = -1;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
};

class Searcher {
    struct TTEntry {
        uint64_t key = 0;
        int32_t score = 0;
        int16_t move = -1;
        int8_t depth = -1;
        uint8_t flag = 0;
    };
    enum : uint8_t { EXACT = 0, LOWER = 1, UPPER = 2 };
    using Clock = std::chrono::steady_clock;

    std::vector<TTEntry> tt;
    uint64_t mask = 0;
    int history[2][256] = {};
    uint64_t nodes = 0;
    bool stop = false;
    bool timed = false;
    Clock::time_point deadline;
    int root_best = -1;

public:
    int wall_weight = 30;

    explicit Searcher(int tt_mb = 64) {
        resize(tt_mb);
    }

    void resize(int tt_mb) {
        size_t n = 1;
        while (n * 2 * sizeof(TTEntry) <= (size_t(tt_mb) << 20))
            n *= 2;
        tt.assign(n, TTEntry{});
        mask = n - 1;
    }

    void clear() {
        std::fill(tt.begin(), tt.end(), TTEntry{});
        std::memset(history, 0, sizeof(history));
    }

    int evaluate(const Game& g) const {
        int me = g.side, op = me ^ 1;
        int dm = g.dist_to_goal(me), dop = g.dist_to_goal(op);
        return 100 * (dop - dm) + wall_weight * (g.walls_left[me] - g.walls_left[op]);
    }

    SearchResult search(const Game& root, int max_depth, int time_ms) {
        nodes = 0;
        stop = false;
        timed = time_ms > 0;
        deadline = Clock::now() + std::chrono::milliseconds(time_ms);
        for (auto& side_hist : history)
            for (int& h : side_hist)
                h >>= 1;
        SearchResult res;
        if (root.is_terminal())
            return res;
        MoveList ml;
        root.legal_moves(ml);
        if (ml.n == 0)
            return res;
        res.move = ml.m[0];
        for (int d = 1; d <= max_depth; d++) {
            root_best = -1;
            int v = negamax(root, d, -INF, INF, 0);
            if (stop)
                break;
            res.move = root_best;
            res.score = v;
            res.depth = d;
            if (v > MATE_BOUND || v < -MATE_BOUND)
                break;
        }
        res.action = root.action_of(res.move);
        res.nodes = nodes;
        return res;
    }

private:
    static int to_tt(int s, int ply) {
        return s > MATE_BOUND ? s + ply : s < -MATE_BOUND ? s - ply : s;
    }
    static int from_tt(int s, int ply) {
        return s > MATE_BOUND ? s - ply : s < -MATE_BOUND ? s + ply : s;
    }

    void mark_shortest_paths(const Game& g, int p, const int* dmap, uint8_t* on) const {
        std::memset(on, 0, g.W * g.H);
        int q[MAX_CELLS];
        int head = 0, tail = 0;
        int start = g.pawn[p];
        if (dmap[start] < 0)
            return;
        q[tail++] = start;
        on[start] = 1;
        while (head < tail) {
            int x = q[head++];
            for (uint8_t d : DIRS) {
                if (g.blocked[x] & d)
                    continue;
                int y = x + g.step(d);
                if (dmap[y] == dmap[x] - 1 && !on[y]) {
                    on[y] = 1;
                    q[tail++] = y;
                }
            }
        }
    }

    static bool cuts(const uint8_t* on, int a, int b) {
        return on[a] && on[b];
    }

    bool wall_cuts_path(const Game& g, int m, const uint8_t* on) const {
        bool horiz = m < V_BASE;
        int s = m - (horiz ? H_BASE : V_BASE);
        int a = (s / g.SW) * g.W + s % g.SW;
        if (horiz)
            return cuts(on, a, a + g.W) || cuts(on, a + 1, a + 1 + g.W);
        return cuts(on, a, a + 1) || cuts(on, a + g.W, a + g.W + 1);
    }

    void score_moves(const Game& g, const MoveList& ml, int tt_move, int* sc) const {
        int me = g.side, op = me ^ 1;
        int dme[MAX_CELLS], dop[MAX_CELLS];
        uint8_t on[MAX_CELLS];
        g.dist_map(me, dme);
        g.dist_map(op, dop);
        mark_shortest_paths(g, op, dop, on);
        int here = dme[g.pawn[me]];
        for (int i = 0; i < ml.n; i++) {
            int m = ml.m[i];
            int h = history[me][m];
            if (m == tt_move)
                sc[i] = 1 << 30;
            else if (m < H_BASE)
                sc[i] = (dme[m] < here ? 1 << 22 : 1 << 18) + h;
            else
                sc[i] = (wall_cuts_path(g, m, on) ? 1 << 20 : 0) + h;
        }
    }

    int negamax(const Game& g, int depth, int alpha, int beta, int ply) {
        if (g.reached(g.side ^ 1))
            return -WIN + ply;
        if (g.ply >= g.max_plies)
            return 0;
        if (depth <= 0)
            return evaluate(g);
        if ((++nodes & 1023) == 0 && timed && Clock::now() > deadline)
            stop = true;
        if (stop)
            return 0;

        const uint64_t key = g.hash();
        TTEntry& e = tt[key & mask];
        int tt_move = -1;
        if (e.key == key) {
            tt_move = e.move;
            if (e.depth >= depth && ply > 0) {
                int s = from_tt(e.score, ply);
                if (e.flag == EXACT)
                    return s;
                if (e.flag == LOWER && s >= beta)
                    return s;
                if (e.flag == UPPER && s <= alpha)
                    return s;
            }
        }

        MoveList ml;
        g.legal_moves(ml);
        if (ml.n == 0)
            return 0;
        int sc[MAX_MOVES];
        score_moves(g, ml, tt_move, sc);

        const int alpha0 = alpha;
        int best = -INF, best_move = ml.m[0];
        for (int i = 0; i < ml.n; i++) {
            int j = i;
            for (int k = i + 1; k < ml.n; k++)
                if (sc[k] > sc[j])
                    j = k;
            std::swap(ml.m[i], ml.m[j]);
            std::swap(sc[i], sc[j]);
            const int m = ml.m[i];

            Game c = g;
            c.apply(m);
            int v = -negamax(c, depth - 1, -beta, -alpha, ply + 1);
            if (stop)
                return 0;
            if (v > best) {
                best = v;
                best_move = m;
                if (ply == 0)
                    root_best = m;
                if (v > alpha)
                    alpha = v;
                if (alpha >= beta) {
                    history[g.side][m] += depth * depth;
                    break;
                }
            }
        }

        uint8_t flag = best <= alpha0 ? UPPER : best >= beta ? LOWER : EXACT;
        if (e.key != key || depth >= e.depth) {
            e.key = key;
            e.score = to_tt(best, ply);
            e.move = int16_t(best_move);
            e.depth = int8_t(std::min(depth, 127));
            e.flag = flag;
        }
        return best;
    }
};

} // namespace qd

#include "engine.hpp"
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace qd;

static int failures = 0;

static void check(bool ok, const char* what, const Game& g) {
    if (ok)
        return;
    failures++;
    if (failures <= 5)
        std::printf("FAIL: %s\n%s\n", what, g.to_string().c_str());
}

static std::vector<int> brute_walls(const Game& g) {
    std::vector<int> out;
    if (g.walls_left[g.side] == 0)
        return out;
    for (int s = 0; s < g.S; s++) {
        for (int o = 0; o < 2; o++) {
            bool horiz = o == 0;
            if (!g.wall_fits(horiz, s))
                continue;
            Game t = g;
            t.place_wall(horiz, s);
            if (t.dist_to_goal(0) >= 0 && t.dist_to_goal(1) >= 0)
                out.push_back((horiz ? H_BASE : V_BASE) + s);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static void fuzz(int w, int h, int walls, int games, std::mt19937& rng) {
    for (int gi = 0; gi < games; gi++) {
        Game g(w, h, walls);
        while (!g.is_terminal()) {
            MoveList ml;
            g.legal_moves(ml);
            check(ml.n > 0, "no legal moves", g);
            if (ml.n == 0)
                break;

            std::vector<int> fast;
            for (int i = 0; i < ml.n; i++)
                if (is_wall(ml.m[i]))
                    fast.push_back(ml.m[i]);
            std::sort(fast.begin(), fast.end());
            check(fast == brute_walls(g), "wall legality mismatch", g);

            std::vector<int> seen;
            for (int i = 0; i < ml.n; i++) {
                int a = g.action_of(ml.m[i]);
                check(a >= 0 && a < g.num_actions(), "action out of range", g);
                check(g.move_of(a) == ml.m[i], "action round trip", g);
                check(g.mirror_action(g.mirror_action(a)) == a, "mirror involution", g);
                seen.push_back(a);
            }
            std::sort(seen.begin(), seen.end());
            check(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "duplicate actions",
                  g);

            int n_pawn = 0;
            while (n_pawn < ml.n && !is_wall(ml.m[n_pawn]))
                n_pawn++;
            bool pawn_only = std::uniform_int_distribution<int>(0, 1)(rng) == 0;
            int hi = pawn_only ? n_pawn - 1 : ml.n - 1;
            g.apply(ml.m[std::uniform_int_distribution<int>(0, hi)(rng)]);
            check(g.dist_to_goal(0) >= 0 && g.dist_to_goal(1) >= 0, "path lost", g);
        }
    }
}

struct SolveCase {
    int w, h, walls, winner;
};

static void solve_tests() {
    const SolveCase cases[] = {
        {3, 3, 0, 1}, {3, 3, 1, 1}, {3, 3, 2, 1}, {3, 3, 3, 1}, {5, 3, 0, 1}, {5, 3, 1, 1},
        {5, 3, 2, 1}, {5, 3, 3, 0}, {5, 3, 4, 0}, {3, 5, 0, 1}, {3, 5, 1, 1}, {3, 5, 2, 1},
        {3, 5, 3, 0}, {3, 5, 4, 1}, {5, 5, 0, 1}, {5, 5, 1, 1}, {5, 5, 2, 1},
    };
    Searcher s(256);
    for (const auto& c : cases) {
        Game g(c.w, c.h, c.walls, 1000);
        s.clear();
        auto t0 = std::chrono::steady_clock::now();
        SearchResult r = s.search(g, 100, 60000);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool proven = r.score > MATE_BOUND || r.score < -MATE_BOUND;
        int win = r.score > MATE_BOUND ? 0 : 1;
        bool ok = proven && win == c.winner;
        if (!ok)
            failures++;
        std::printf("%dx%d walls=%d  expected P%d  got %s  depth=%d nodes=%llu  %.2fs  %s\n", c.w,
                    c.h, c.walls, c.winner + 1, proven ? (win == 0 ? "P1" : "P2") : "unproven",
                    r.depth, (unsigned long long)r.nodes, sec, ok ? "ok" : "FAIL");
    }
}

static void bench() {
    Game g(9, 9, 10);
    Searcher s(256);
    auto t0 = std::chrono::steady_clock::now();
    SearchResult r = s.search(g, 64, 3000);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("9x9 start: depth=%d score=%d action=%d nodes=%llu  %.0f nodes/s\n", r.depth,
                r.score, r.action, (unsigned long long)r.nodes, r.nodes / sec);

    MoveList ml;
    auto t1 = std::chrono::steady_clock::now();
    long long total = 0;
    for (int i = 0; i < 100000; i++) {
        g.legal_moves(ml);
        total += ml.n;
    }
    double sec1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("9x9 movegen: %.0f ns/call (%lld moves)\n", sec1 / 100000 * 1e9, total / 100000);
}

int main() {
    std::mt19937 rng(12345);
    fuzz(3, 3, 3, 300, rng);
    fuzz(5, 5, 5, 300, rng);
    fuzz(9, 9, 10, 200, rng);
    fuzz(4, 7, 6, 200, rng);
    std::printf("fuzz: %s (%d failures)\n", failures ? "FAIL" : "ok", failures);
    solve_tests();
    bench();
    std::printf("%s\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}

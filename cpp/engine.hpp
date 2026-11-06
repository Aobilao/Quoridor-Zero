#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace qd {

constexpr int MAX_W = 9;
constexpr int MAX_H = 9;
constexpr int MAX_CELLS = MAX_W * MAX_H;
constexpr int MAX_CORNERS = (MAX_W + 1) * (MAX_H + 1);
constexpr int MAX_MOVES = 5 + 2 * (MAX_W - 1) * (MAX_H - 1);
constexpr int N_PAWN_ACTIONS = 12;
constexpr int N_PLANES = 8;
constexpr int H_BASE = 128;
constexpr int V_BASE = 192;

enum : uint8_t { UP = 1, DOWN = 2, LEFT = 4, RIGHT = 8 };
constexpr uint8_t DIRS[4] = {UP, DOWN, LEFT, RIGHT};

constexpr int PAWN_DR[12] = {1, -1, 0, 0, 2, -2, 0, 0, 1, 1, -1, -1};
constexpr int PAWN_DC[12] = {0, 0, -1, 1, 0, 0, -2, 2, -1, 1, -1, 1};
constexpr int PAWN_MIRROR[12] = {0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10};

inline bool is_wall(int m) {
    return m >= H_BASE;
}

struct MoveList {
    int n = 0;
    int m[MAX_MOVES];
    void add(int x) {
        m[n++] = x;
    }
};

class Game {
public:
    int W, H, SW, SH, S;
    int pawn[2];
    int walls_left[2];
    int start_walls;
    int side = 0;
    int ply = 0;
    int max_plies;
    uint64_t hw = 0;
    uint64_t vw = 0;
    uint8_t blocked[MAX_CELLS];
    uint8_t corner[MAX_CORNERS];

    Game(int w = 9, int h = 9, int walls = 10, int max_plies_ = 0) {
        W = std::clamp(w, 2, MAX_W);
        H = std::clamp(h, 2, MAX_H);
        SW = W - 1;
        SH = H - 1;
        S = SW * SH;
        start_walls = walls;
        walls_left[0] = walls_left[1] = walls;
        max_plies = max_plies_ > 0 ? max_plies_ : 4 * W * H;
        pawn[0] = W / 2;
        pawn[1] = (H - 1) * W + W / 2;
        std::memset(blocked, 0, sizeof(blocked));
        std::memset(corner, 0, sizeof(corner));
        for (int x = 0; x < W * H; x++) {
            int r = x / W, c = x % W;
            if (r == H - 1)
                blocked[x] |= UP;
            if (r == 0)
                blocked[x] |= DOWN;
            if (c == 0)
                blocked[x] |= LEFT;
            if (c == W - 1)
                blocked[x] |= RIGHT;
        }
    }

    int step(uint8_t d) const {
        return d == UP ? W : d == DOWN ? -W : d == LEFT ? -1 : 1;
    }
    int goal_row(int p) const {
        return p == 0 ? H - 1 : 0;
    }
    bool reached(int p) const {
        return pawn[p] / W == goal_row(p);
    }
    int winner() const {
        return reached(0) ? 0 : reached(1) ? 1 : -1;
    }
    bool is_terminal() const {
        return winner() >= 0 || ply >= max_plies;
    }
    int num_actions() const {
        return N_PAWN_ACTIONS + 2 * S;
    }

    bool wall_fits(bool horiz, int s) const {
        if (((hw | vw) >> s) & 1)
            return false;
        int r = s / SW, c = s % SW;
        if (horiz) {
            if (c > 0 && ((hw >> (s - 1)) & 1))
                return false;
            if (c < SW - 1 && ((hw >> (s + 1)) & 1))
                return false;
        } else {
            if (r > 0 && ((vw >> (s - SW)) & 1))
                return false;
            if (r < SH - 1 && ((vw >> (s + SW)) & 1))
                return false;
        }
        return true;
    }

    void wall_corners(bool horiz, int s, int k[3]) const {
        int r = s / SW, c = s % SW, cw = W + 1;
        if (horiz) {
            int y = r + 1;
            k[0] = y * cw + c;
            k[1] = y * cw + c + 1;
            k[2] = y * cw + c + 2;
        } else {
            int x = c + 1;
            k[0] = r * cw + x;
            k[1] = (r + 1) * cw + x;
            k[2] = (r + 2) * cw + x;
        }
    }

    bool corner_touched(int k) const {
        int x = k % (W + 1), y = k / (W + 1);
        return x == 0 || x == W || y == 0 || y == H || corner[k] > 0;
    }

    bool needs_path_check(bool horiz, int s) const {
        int k[3];
        wall_corners(horiz, s, k);
        return corner_touched(k[0]) + corner_touched(k[1]) + corner_touched(k[2]) >= 2;
    }

    void place_wall(bool horiz, int s) {
        int r = s / SW, c = s % SW, a = r * W + c;
        if (horiz) {
            hw |= 1ULL << s;
            blocked[a] |= UP;
            blocked[a + 1] |= UP;
            blocked[a + W] |= DOWN;
            blocked[a + W + 1] |= DOWN;
        } else {
            vw |= 1ULL << s;
            blocked[a] |= RIGHT;
            blocked[a + W] |= RIGHT;
            blocked[a + 1] |= LEFT;
            blocked[a + W + 1] |= LEFT;
        }
        int k[3];
        wall_corners(horiz, s, k);
        corner[k[0]]++;
        corner[k[1]]++;
        corner[k[2]]++;
    }

    int dist_to_goal(int p) const {
        const int goal = goal_row(p);
        const int start = pawn[p];
        if (start / W == goal)
            return 0;
        uint8_t seen[MAX_CELLS];
        std::memset(seen, 0, W * H);
        int q[MAX_CELLS];
        int head = 0, tail = 0, depth = 0;
        q[tail++] = start;
        seen[start] = 1;
        while (head < tail) {
            int level_end = tail;
            depth++;
            while (head < level_end) {
                int x = q[head++];
                uint8_t b = blocked[x];
                for (uint8_t d : DIRS) {
                    if (b & d)
                        continue;
                    int y = x + step(d);
                    if (seen[y])
                        continue;
                    if (y / W == goal)
                        return depth;
                    seen[y] = 1;
                    q[tail++] = y;
                }
            }
        }
        return -1;
    }

    void dist_map(int p, int* out) const {
        const int n = W * H;
        std::fill(out, out + n, -1);
        int q[MAX_CELLS];
        int head = 0, tail = 0;
        const int r0 = goal_row(p) * W;
        for (int c = 0; c < W; c++) {
            out[r0 + c] = 0;
            q[tail++] = r0 + c;
        }
        while (head < tail) {
            int x = q[head++];
            uint8_t b = blocked[x];
            for (uint8_t d : DIRS) {
                if (b & d)
                    continue;
                int y = x + step(d);
                if (out[y] >= 0)
                    continue;
                out[y] = out[x] + 1;
                q[tail++] = y;
            }
        }
    }

    void legal_pawn_moves(MoveList& ml) const {
        const int me = pawn[side], op = pawn[side ^ 1];
        for (uint8_t d : DIRS) {
            if (blocked[me] & d)
                continue;
            int n = me + step(d);
            if (n != op) {
                ml.add(n);
                continue;
            }
            if (!(blocked[op] & d)) {
                ml.add(op + step(d));
                continue;
            }
            uint8_t perp = (d == UP || d == DOWN) ? (LEFT | RIGHT) : (UP | DOWN);
            for (uint8_t e : DIRS)
                if ((perp & e) && !(blocked[op] & e))
                    ml.add(op + step(e));
        }
    }

    bool wall_legal(bool horiz, int s) const {
        if (!wall_fits(horiz, s))
            return false;
        if (!needs_path_check(horiz, s))
            return true;
        Game t = *this;
        t.place_wall(horiz, s);
        return t.dist_to_goal(0) >= 0 && t.dist_to_goal(1) >= 0;
    }

    void legal_moves(MoveList& ml) const {
        ml.n = 0;
        legal_pawn_moves(ml);
        if (walls_left[side] == 0)
            return;
        for (int s = 0; s < S; s++) {
            if (wall_legal(true, s))
                ml.add(H_BASE + s);
            if (wall_legal(false, s))
                ml.add(V_BASE + s);
        }
    }

    void apply(int m) {
        if (m < H_BASE) {
            pawn[side] = m;
        } else {
            bool horiz = m < V_BASE;
            place_wall(horiz, m - (horiz ? H_BASE : V_BASE));
            walls_left[side]--;
        }
        side ^= 1;
        ply++;
    }

    int action_of(int m) const {
        if (m < H_BASE) {
            int me = pawn[side];
            int dr = m / W - me / W, dc = m % W - me % W;
            if (side == 1)
                dr = -dr;
            for (int i = 0; i < N_PAWN_ACTIONS; i++)
                if (PAWN_DR[i] == dr && PAWN_DC[i] == dc)
                    return i;
            return -1;
        }
        bool horiz = m < V_BASE;
        int s = m - (horiz ? H_BASE : V_BASE);
        int r = s / SW, c = s % SW;
        if (side == 1)
            r = SH - 1 - r;
        return N_PAWN_ACTIONS + (horiz ? 0 : S) + r * SW + c;
    }

    int move_of(int a) const {
        if (a < N_PAWN_ACTIONS) {
            int dr = side == 0 ? PAWN_DR[a] : -PAWN_DR[a];
            return pawn[side] + dr * W + PAWN_DC[a];
        }
        a -= N_PAWN_ACTIONS;
        bool horiz = a < S;
        int s = horiz ? a : a - S;
        int r = s / SW, c = s % SW;
        if (side == 1)
            r = SH - 1 - r;
        return (horiz ? H_BASE : V_BASE) + r * SW + c;
    }

    int mirror_action(int a) const {
        if (a < N_PAWN_ACTIONS)
            return PAWN_MIRROR[a];
        a -= N_PAWN_ACTIONS;
        int base = a < S ? 0 : S;
        int s = a - base;
        int r = s / SW, c = s % SW;
        return N_PAWN_ACTIONS + base + r * SW + (SW - 1 - c);
    }

    void features(float* out) const {
        const int n = W * H;
        std::fill(out, out + N_PLANES * n, 0.0f);
        const int me = side, op = side ^ 1;
        auto cell = [&](int plane, int r, int c) -> float& {
            int rr = side == 0 ? r : H - 1 - r;
            return out[plane * n + rr * W + c];
        };
        cell(0, pawn[me] / W, pawn[me] % W) = 1.0f;
        cell(1, pawn[op] / W, pawn[op] % W) = 1.0f;
        for (int s = 0; s < S; s++) {
            int r = s / SW, c = s % SW;
            int rr = side == 0 ? r : SH - 1 - r;
            if ((hw >> s) & 1)
                out[2 * n + rr * W + c] = 1.0f;
            if ((vw >> s) & 1)
                out[3 * n + rr * W + c] = 1.0f;
        }
        float wm = start_walls ? float(walls_left[me]) / start_walls : 0.0f;
        float wo = start_walls ? float(walls_left[op]) / start_walls : 0.0f;
        std::fill(out + 4 * n, out + 5 * n, wm);
        std::fill(out + 5 * n, out + 6 * n, wo);
        int dm[MAX_CELLS], dop[MAX_CELLS];
        dist_map(me, dm);
        dist_map(op, dop);
        const float norm = float(n);
        for (int x = 0; x < n; x++) {
            cell(6, x / W, x % W) = dm[x] < 0 ? 1.0f : dm[x] / norm;
            cell(7, x / W, x % W) = dop[x] < 0 ? 1.0f : dop[x] / norm;
        }
    }

    static uint64_t mix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    uint64_t hash() const {
        uint64_t k = uint64_t(pawn[0]) | uint64_t(pawn[1]) << 8 | uint64_t(walls_left[0]) << 16 |
                     uint64_t(walls_left[1]) << 24 | uint64_t(side) << 32;
        return mix64(hw ^ mix64(vw ^ mix64(k)));
    }

    std::string to_string() const {
        std::string s;
        for (int r = H - 1; r >= 0; r--) {
            for (int c = 0; c < W; c++) {
                int x = r * W + c;
                s += x == pawn[0] ? '0' : x == pawn[1] ? '1' : '.';
                if (c < W - 1)
                    s += (blocked[x] & RIGHT) ? '|' : ' ';
            }
            s += '\n';
            if (r > 0) {
                for (int c = 0; c < W; c++) {
                    s += (blocked[r * W + c] & DOWN) ? '-' : ' ';
                    if (c < W - 1)
                        s += ' ';
                }
                s += '\n';
            }
        }
        s += "to move: " + std::to_string(side) + "  walls: " + std::to_string(walls_left[0]) +
             " " + std::to_string(walls_left[1]) + "\n";
        return s;
    }
};

} // namespace qd

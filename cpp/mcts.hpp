#pragma once
#include "engine.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace qd {

constexpr int MAX_ACTIONS = N_PAWN_ACTIONS + 2 * (MAX_W - 1) * (MAX_H - 1);

struct MCTSConfig {
    int sims = 64;
    int max_considered = 16;
    float c_puct = 1.25f;
    float fpu_reduction = 0.2f;
    float c_visit = 50.0f;
    float c_scale = 0.1f;
};

inline std::vector<int> considered_visit_sequence(int m, int n) {
    std::vector<int> seq;
    if (m <= 1) {
        for (int i = 0; i < n; i++)
            seq.push_back(i);
        return seq;
    }
    int log2m = int(std::ceil(std::log2(double(m))));
    std::vector<int> visits(m, 0);
    int considered = m;
    while (int(seq.size()) < n) {
        int extra = std::max(1, n / (log2m * considered));
        for (int e = 0; e < extra; e++) {
            for (int i = 0; i < considered; i++)
                seq.push_back(visits[i]++);
        }
        considered = std::max(2, considered / 2);
    }
    seq.resize(n);
    return seq;
}

class Tree {
    struct Edge {
        int move;
        int action;
        float log_prior;
        float prior;
        int child = -1;
    };
    struct Node {
        Game g;
        int parent = -1;
        int edge_begin = 0;
        int edge_count = 0;
        int visits = 0;
        float value_sum = 0.0f;
        float raw_value = 0.0f;
        bool expanded = false;
        bool terminal = false;
        float terminal_value = 0.0f;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<float> gumbel;
    std::vector<int> seq;
    const MCTSConfig* cfg = nullptr;

public:
    int sims_done = 0;
    int sims_target = 0;
    int leaf = -1;

    void reset(const Game& root, const MCTSConfig& config) {
        cfg = &config;
        nodes.clear();
        edges.clear();
        gumbel.clear();
        seq.clear();
        Node r;
        r.g = root;
        nodes.push_back(r);
        sims_done = 0;
        sims_target = config.sims;
        leaf = -1;
    }

    const Game& leaf_game() const {
        return nodes[leaf].g;
    }
    const Game& root_game() const {
        return nodes[0].g;
    }

    bool select_leaf() {
        while (true) {
            if (!nodes[0].expanded) {
                leaf = 0;
                return true;
            }
            if (sims_done >= sims_target)
                return false;
            int n = 0;
            while (true) {
                int e = n == 0 ? select_root() : select_puct(n);
                if (edges[e].child < 0) {
                    edges[e].child = make_child(n, edges[e].move);
                }
                n = edges[e].child;
                if (nodes[n].terminal || !nodes[n].expanded)
                    break;
            }
            if (nodes[n].terminal) {
                backup(n, nodes[n].terminal_value);
                sims_done++;
                continue;
            }
            leaf = n;
            return true;
        }
    }

    void feed(const float* logits, float value, std::mt19937_64& rng, bool noise) {
        expand(leaf, logits, value);
        if (leaf == 0) {
            nodes[0].visits = 1;
            nodes[0].value_sum = value;
            const Node& r = nodes[0];
            gumbel.assign(r.edge_count, 0.0f);
            if (noise) {
                std::uniform_real_distribution<double> u(1e-12, 1.0 - 1e-12);
                for (float& x : gumbel)
                    x = float(-std::log(-std::log(u(rng))));
            }
            if (r.edge_count <= 1)
                sims_target = 0;
            seq =
                considered_visit_sequence(std::min(cfg->max_considered, r.edge_count), sims_target);
        } else {
            backup(leaf, value);
            sims_done++;
        }
        leaf = -1;
    }

    int best_action() const {
        return edges[nodes[0].edge_begin + best_index()].action;
    }

    int best_index() const {
        const Node& r = nodes[0];
        std::vector<float> sigma = root_sigma();
        int max_n = 0;
        for (int i = 0; i < r.edge_count; i++)
            max_n = std::max(max_n, visits_of(r.edge_begin + i));
        int best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < r.edge_count; i++) {
            const Edge& e = edges[r.edge_begin + i];
            if (visits_of(r.edge_begin + i) != max_n)
                continue;
            float s = gumbel[i] + e.log_prior + sigma[i];
            if (s > best_score) {
                best_score = s;
                best = i;
            }
        }
        return best;
    }

    void policy(float* out, int num_actions) const {
        std::fill(out, out + num_actions, 0.0f);
        const Node& r = nodes[0];
        std::vector<float> sigma = root_sigma();
        float mx = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < r.edge_count; i++)
            mx = std::max(mx, edges[r.edge_begin + i].log_prior + sigma[i]);
        float total = 0.0f;
        for (int i = 0; i < r.edge_count; i++) {
            const Edge& e = edges[r.edge_begin + i];
            float p = std::exp(e.log_prior + sigma[i] - mx);
            out[e.action] = p;
            total += p;
        }
        for (int i = 0; i < r.edge_count; i++)
            out[edges[r.edge_begin + i].action] /= total;
    }

    float root_value() const {
        const Node& r = nodes[0];
        if (r.edge_count == 0)
            return r.raw_value;
        int e = r.edge_begin + best_index();
        return visits_of(e) ? q_of(e) : r.raw_value;
    }
    float root_raw_value() const {
        return nodes[0].raw_value;
    }

    std::vector<std::pair<int, int>> root_visits() const {
        std::vector<std::pair<int, int>> out;
        const Node& r = nodes[0];
        for (int i = 0; i < r.edge_count; i++)
            out.emplace_back(edges[r.edge_begin + i].action, visits_of(r.edge_begin + i));
        return out;
    }

private:
    int visits_of(int e) const {
        return edges[e].child < 0 ? 0 : nodes[edges[e].child].visits;
    }

    float q_of(int e) const {
        const Node& c = nodes[edges[e].child];
        return -c.value_sum / c.visits;
    }

    int make_child(int parent, int move) {
        Node c;
        c.g = nodes[parent].g;
        c.g.apply(move);
        c.parent = parent;
        if (c.g.is_terminal()) {
            c.terminal = true;
            int w = c.g.winner();
            c.terminal_value = w < 0 ? 0.0f : w == c.g.side ? 1.0f : -1.0f;
        }
        nodes.push_back(c);
        return int(nodes.size()) - 1;
    }

    void expand(int n, const float* logits, float value) {
        MoveList ml;
        nodes[n].g.legal_moves(ml);
        int begin = int(edges.size());
        float mx = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < ml.n; i++) {
            int a = nodes[n].g.action_of(ml.m[i]);
            edges.push_back(Edge{ml.m[i], a, logits[a], 0.0f, -1});
            mx = std::max(mx, logits[a]);
        }
        float total = 0.0f;
        for (int i = begin; i < int(edges.size()); i++) {
            edges[i].prior = std::exp(edges[i].log_prior - mx);
            total += edges[i].prior;
        }
        for (int i = begin; i < int(edges.size()); i++) {
            edges[i].prior /= total;
            edges[i].log_prior = std::log(std::max(edges[i].prior, 1e-30f));
        }
        Node& node = nodes[n];
        node.edge_begin = begin;
        node.edge_count = ml.n;
        node.raw_value = value;
        node.expanded = true;
    }

    void backup(int n, float v) {
        while (n >= 0) {
            nodes[n].visits++;
            nodes[n].value_sum += v;
            v = -v;
            n = nodes[n].parent;
        }
    }

    int select_puct(int n) const {
        const Node& node = nodes[n];
        float fpu = node.value_sum / node.visits - cfg->fpu_reduction;
        float sq = std::sqrt(float(node.visits));
        int best = node.edge_begin;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int i = node.edge_begin; i < node.edge_begin + node.edge_count; i++) {
            int nv = visits_of(i);
            float q = nv ? q_of(i) : fpu;
            float s = q + cfg->c_puct * edges[i].prior * sq / (1 + nv);
            if (s > best_score) {
                best_score = s;
                best = i;
            }
        }
        return best;
    }

    std::vector<float> root_sigma() const {
        const Node& r = nodes[0];
        std::vector<float> q(r.edge_count);
        int sum_n = 0, max_n = 0;
        float sum_p = 0.0f, sum_pq = 0.0f;
        for (int i = 0; i < r.edge_count; i++) {
            int e = r.edge_begin + i, nv = visits_of(e);
            sum_n += nv;
            max_n = std::max(max_n, nv);
            if (nv) {
                sum_p += edges[e].prior;
                sum_pq += edges[e].prior * q_of(e);
            }
        }
        float mixed = sum_n ? (r.raw_value + sum_n * (sum_pq / sum_p)) / (1 + sum_n) : r.raw_value;
        float lo = std::numeric_limits<float>::infinity(), hi = -lo;
        for (int i = 0; i < r.edge_count; i++) {
            int e = r.edge_begin + i;
            q[i] = visits_of(e) ? q_of(e) : mixed;
            lo = std::min(lo, q[i]);
            hi = std::max(hi, q[i]);
        }
        float scale = (cfg->c_visit + max_n) * cfg->c_scale / std::max(hi - lo, 1e-8f);
        for (float& x : q)
            x = (x - lo) * scale;
        return q;
    }

    int select_root() const {
        const Node& r = nodes[0];
        std::vector<float> sigma = root_sigma();
        int considered = seq[sims_done];
        int best = r.edge_begin;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < r.edge_count; i++) {
            int e = r.edge_begin + i;
            if (visits_of(e) != considered)
                continue;
            float s = gumbel[i] + edges[e].log_prior + sigma[i];
            if (s > best_score) {
                best_score = s;
                best = e;
            }
        }
        return best;
    }
};

class BatchMCTS {
    std::vector<Tree> trees;
    std::vector<int> waiting;
    std::mt19937_64 rng;
    bool noise = false;

public:
    MCTSConfig cfg;

    explicit BatchMCTS(const MCTSConfig& c, uint64_t seed = 0) : rng(seed), cfg(c) {
    }

    void start(const std::vector<Game>& roots, bool gumbel_noise) {
        noise = gumbel_noise;
        trees.resize(roots.size());
        waiting.clear();
        for (size_t i = 0; i < roots.size(); i++)
            trees[i].reset(roots[i], cfg);
    }

    int size() const {
        return int(trees.size());
    }
    const std::vector<int>& pending() const {
        return waiting;
    }

    int prepare() {
        waiting.clear();
        for (int i = 0; i < int(trees.size()); i++)
            if (trees[i].select_leaf())
                waiting.push_back(i);
        return int(waiting.size());
    }

    const Game& leaf_game(int k) const {
        return trees[waiting[k]].leaf_game();
    }

    void feed(const float* logits, const float* values, int num_actions) {
        for (size_t k = 0; k < waiting.size(); k++)
            trees[waiting[k]].feed(logits + k * num_actions, values[k], rng, noise);
        waiting.clear();
    }

    const Tree& tree(int i) const {
        return trees[i];
    }
};

} // namespace qd

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include <memory>
#include <map>
#include <random>
#include <set>
#include <iomanip>
#include <ctime>
#include <cassert>


struct Point {
private:
    int x, y;
public:
    Point(const Point &p) {
        x = p.X();
        y = p.Y();
    }

    Point(int x, int y) : x(x), y(y) {}

    Point() : x(int()), y(int()) {}

    int X() const {
        return x;
    }

    int Y() const {
        return y;
    }

    inline int len_sqr() const {
        return x * x + y * y;
    }

    inline double len() const {
        return std::sqrt(x * x + y * y);
    }


    friend int operator^(const Point &a, const Point &b) {
        return a.x * b.y - a.y * b.x;
    }

    friend int operator*(const Point &a, const Point &b) {
        return a.x * b.x + a.y * b.y;
    }

    friend Point operator+(const Point &a, const Point &b) {
        return Point(a.x + b.x, a.y + b.y);;
    }

    friend Point operator-(const Point &a, const Point &b) {
        return Point(a.x - b.x, a.y - b.y);;
    }

    friend Point operator*(const Point &a, int k) {
        return Point(a.x * k, a.y * k);
    }

    friend Point operator*(int k, const Point &a) {
        return Point(a.x * k, a.y * k);
    }

    friend Point operator/(const Point &a, int k) {
        return Point(a.x / k, a.y / k);
    }


    friend std::istream &operator>>(std::istream &is, Point &pt) {
        is >> pt.x >> pt.y;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, Point pt) {
        os << pt.x << ' ' << pt.y;
        return os;
    }

    const bool operator==(const Point &v) {
        return v.x == x && v.y == y;
    }

    const Point &operator+=(const Point &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    const Point &operator-=(const Point &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    const Point &operator*=(const int k) {
        x *= k;
        y *= k;
        return *this;
    }

    const Point &operator/=(const int k) {
        x /= k;
        y /= k;
        return *this;
    }

    Point &operator=(const Point &o) {
        x = o.X();
        y = o.Y();
        return *this;
    }

    bool operator<(const Point &o) const {
        return (x == o.x) ? (y < o.y) : x < o.x;
    }

    Point operator-() {
        return Point(-x, -y);
    }
};

class Problem {
public:
    size_t num_taxis = 0, num_fans = 0, num_zones = 0;
    std::vector<Point> taxis, fans, zones;

    Problem() : num_taxis(0), num_fans(0), num_zones(0) {}

    friend std::istream &operator>>(std::istream &in, Problem &p) {
        in >> p.num_taxis;
        p.taxis.resize(p.num_taxis);
        for (auto &taxi:p.taxis)
            in >> taxi;

        in >> p.num_fans;
        p.fans.resize(p.num_fans);
        for (auto &fan:p.fans)
            in >> fan;

        in >> p.num_zones;
        p.zones.resize(p.num_zones);
        for (auto &zone:p.zones) {
            in >> zone;
        }

        return in;
    }
};

class Move {
public:
    Point delta;
    std::vector<size_t> indices;

    Move() = default;

    Move(const Point &p, std::vector<size_t> v) : delta(p), indices(std::move(v)) {}

    friend std::ostream &operator<<(std::ostream &out, Move m) {
        out << "MOVE " << m.delta.X() << ' ' << m.delta.Y() << ' ';
        out << m.indices.size();
        for (auto it:m.indices) {
            out << " ";
            out << it + 1;
        }
        return out;
    }
};

class Solution {
public:
    std::vector<Move> moves;
    Problem p;
    double score = 0;
    double TLE_LIMIT = 10;

    explicit Solution(Problem p) : p(std::move(p)) {
    }

    double evaluate() {
        double ans = 0;
        for (const auto &move:moves) {
            ans += eval_move(move);
        }
        this->score = ans;
        return ans;
    }

    double eval_move(const Move &move) const {
        return move.delta.len() * (1. + static_cast<double>(move.indices.size()) / p.num_taxis);
    }


    virtual ~Solution() = default;

    virtual void solve() {

    }

};

class SolutionGreedy1 : public Solution {
public:
    explicit SolutionGreedy1(const Problem &p) : Solution(p) {

    }

    void solve() override {
        std::vector<Move> moves;
        Point taxi = p.taxis.front();
        for (const auto &fan : p.fans) {
            Point direction_to_fan = fan - taxi;
            moves.emplace_back(direction_to_fan, std::vector<size_t>(1, 0));
            taxi += direction_to_fan;

            Point direction_to_zone = p.zones.front() - taxi;
            moves.emplace_back(direction_to_zone, std::vector<size_t>(1, 0));
            taxi += direction_to_zone;
        }
        this->moves = moves;
    }
};

namespace GLobalScope {
    std::vector<Point> best_move_points;
    double c_start = 0;

    double elapsed_seconds_main() {
        return double(clock() - c_start) / CLOCKS_PER_SEC;
    }

    double elapsed_seconds_local(double c_start) {
        return double(clock() - c_start) / CLOCKS_PER_SEC;
    }
    std::default_random_engine generator;

    double BEST_SCORE = 1e15;
    std::uniform_real_distribution<double> q(0.5,2.3);

    int gen_seed = 1834;

    void set_rand(int x) {
        gen_seed = x;
    }

    inline int gen_rand() {
        gen_seed = (214013 * gen_seed + 2531011);
        return (gen_seed >> 16) & 0x7FFF;
    }

    void generate_best_move_points() {
        for (int i = -20; i < 20; i++) {
            for (int j = -20; j < 20; j++) {
                if (i != 0 || j != 0) {
                    best_move_points.emplace_back(i, j);
                }
            }
        }
        sort(best_move_points.begin(), best_move_points.end(),
             [](const Point &lhs, const Point &rhs) {
                 return lhs.len_sqr() < rhs.len_sqr();
             });
    }

};

class GreedySolution : public Solution {
private:

    size_t num_direction_to_look;
    bool add_direction_move;
    double dir_len_constant;
    int score_direction_type;
    std::vector<double> state_constants;

public:
    GreedySolution(const Problem &p,
                   size_t _num_d,
                   bool _add_direction_move,
                   double _dir_len_constant,
                   int _score_direction_dype
    ) :
            Solution(p),
            num_direction_to_look(_num_d),
            add_direction_move(_add_direction_move),
            dir_len_constant(_dir_len_constant),
            score_direction_type(_score_direction_dype) {

    }

    bool is_direction_better(const std::vector<Point> &one_taxi_targets, int taxi_state, Point taxi,
                             const Point &direction) {
        // IMPROVE
        Point old_taxi = taxi;
        taxi += direction;
        if (abs(taxi.X()) > 10000 || abs(taxi.Y()) > 10000) {
            return false;
        }
        auto taxi_in_zones = std::find(p.zones.begin(), p.zones.end(), taxi);
        if (taxi_state == 0 && taxi_in_zones != p.zones.end()) return false;

        if (std::find(p.taxis.begin(), p.taxis.end(), taxi) != p.taxis.end()) {
            return false;
        }

        auto taxi_in_fans = std::find(p.fans.begin(), p.fans.end(), taxi);
        if (taxi_state == 1 && taxi_in_fans != p.fans.end()) return false;

        if (one_taxi_targets.empty()) {
            return true;
        }

        auto nearest_after = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                              [&taxi](const Point &lhs, const Point &rhs) {
                                                  return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                              });
        auto nearest_before = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                               [&old_taxi](const Point &lhs, const Point &rhs) {
                                                   return (lhs - old_taxi).len_sqr() < (rhs - old_taxi).len_sqr();
                                               });

        if (add_direction_move) {
            return (*nearest_after - taxi).len() + 1.5 * direction.len() / (p.taxis.size()) <=
                   (*nearest_before - old_taxi).len();
        } else {
            return (*nearest_after - taxi).len_sqr() <= (*nearest_before - old_taxi).len_sqr();
        }

    }

    Move choose_best_move(const std::vector<Point> &taxis, const std::vector<int> &taxi_state,
                          const std::vector<std::vector<Point>> &targets, const Point &direction) {
        Move move(direction, std::vector<size_t>());
        for (size_t taxi_ind = 0; taxi_ind < taxis.size(); taxi_ind++) {
            if (is_direction_better(targets[taxi_state[taxi_ind]], taxi_state[taxi_ind], taxis[taxi_ind], direction)) {
                move.indices.push_back(taxi_ind);
            }
        }
        return move;
    }


    std::pair<double, Move> score_direction(std::vector<Point> taxis, const std::vector<int> &taxi_state,
                                            const std::vector<std::vector<Point>> &targets, const Point &direction,
                                            const std::vector<Point> &all_direction) {
        double score = 0;
        Move move = choose_best_move(taxis, taxi_state, targets, direction);
        for (auto taxi_ind:move.indices) {
            taxis[taxi_ind] = taxis[taxi_ind] + direction;
        }
        int cnt_fanzone_moves = 0;
        double mult_constant = 1;
        for (size_t taxi_ind = 0; taxi_ind < taxis.size(); taxi_ind++) {
            auto taxi = taxis[taxi_ind];
            auto state = taxi_state[taxi_ind];
            if (targets[taxi_state[taxi_ind]].empty()) continue;
            auto nearest_target = std::min_element(targets[state].begin(),
                                                   targets[state].end(),
                                                   [&taxi](const Point &lhs, const Point &rhs) {
                                                       return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                                   });
            // auto neares_target_direction = *nearest_target - taxi;
            score -= (*nearest_target - taxi).len();
            long target_ind = std::distance(targets[state].begin(), nearest_target);
        }
        if ((score_direction_type >> 1) & 1)
            score = score / (p.num_taxis + cnt_fanzone_moves);

        score -= eval_move(move);


        if (move.indices.empty()) {
            score = -1e16;
        }
        return {score, move};
    }

    void update_targets(std::vector<std::vector<Point>> &targets) {
        targets[0] = p.fans;
    }

    void solve() override {
        std::vector<Move> moves;
        std::vector<std::vector<Point>> targets;
        std::vector<Point> directions;
        std::vector<int> taxi_state(p.taxis.size(), 0);
        /*
         * 0 - empty
         * 1 - carried passenger
         */
        targets.resize(2);
        targets[1] = p.zones;
        targets[0] = p.fans;

        int num_iterations = 0;
        while (!p.fans.empty() || std::count(taxi_state.begin(), taxi_state.end(), 1) != 0) {

            // directions.reserve(p.taxis.size() * targets.size());
            if (num_iterations > 10 * p.num_fans) {
                add_direction_move = false;
            }
            directions.clear();
            if (num_direction_to_look == 1) {
                for (size_t taxi_ind = 0; taxi_ind < p.taxis.size(); taxi_ind++) {
                    auto taxi = p.taxis[taxi_ind];
                    if (!targets[taxi_state[taxi_ind]].empty()) {
                        auto nearest_target = std::min_element(targets[taxi_state[taxi_ind]].begin(),
                                                               targets[taxi_state[taxi_ind]].end(),
                                                               [&taxi](const Point &lhs, const Point &rhs) {
                                                                   return (lhs - taxi).len_sqr() <
                                                                          (rhs - taxi).len_sqr();
                                                               });
                        directions.push_back(*nearest_target - p.taxis[taxi_ind]);
                    }
                }
            } else {
                for (size_t taxi_ind = 0; taxi_ind < p.taxis.size(); taxi_ind++) {
                    for (auto &&target : targets[taxi_state[taxi_ind]]) {
                        directions.push_back(target - p.taxis[taxi_ind]);
                    }
                }
            }
            if(num_direction_to_look < directions.size()){
                std::nth_element(directions.begin(),directions.begin() + num_direction_to_look - 1, directions.end(), [](const Point &lhs, const Point &rhs) {
                    return lhs.len_sqr() < rhs.len_sqr();
                });
            }
            directions.resize(std::min(static_cast<size_t>(num_direction_to_look), directions.size()));



            Move move;
            std::vector<std::pair<double, Move>> scores(directions.size());
            for (size_t dir_ind = 0; dir_ind < directions.size(); dir_ind++) {
                auto res = score_direction(p.taxis, taxi_state, targets, directions[dir_ind], directions);
                scores[dir_ind].first = res.first;
                scores[dir_ind].second = res.second;
            }

            auto best_direction = std::max_element(scores.begin(), scores.end(),
                                                   [](std::pair<double, Move> &lhs, std::pair<double, Move> &rhs) {
                                                       return lhs.first < rhs.first;
                                                   });
            move = best_direction->second;
            for (auto taxi_ind:move.indices) {
                p.taxis[taxi_ind] = p.taxis[taxi_ind] + move.delta;
            }

            moves.push_back(move);

            for (size_t ind = 0; ind < p.taxis.size(); ++ind) {
                auto taxi = p.taxis[ind];
                auto equal_taxi_to_zone = std::find(p.zones.begin(), p.zones.end(), taxi);
                if (equal_taxi_to_zone != p.zones.end()) {
                    // came to zone
                    taxi_state[ind] = 0;
                    for (auto &move_direction : GLobalScope::best_move_points) {
                        std::vector<Point> empty_vector;
                        if (is_direction_better(empty_vector, 0, taxi, move_direction) &&
                            std::count(p.taxis.begin(), p.taxis.end(), taxi + move_direction) == 0) {
                            p.taxis[ind] += move_direction;
                            taxi += move_direction;
                            moves.emplace_back(move_direction, std::vector<size_t>(1, ind));
                            break;
                        }
                    }

                    auto equal_taxi_to_fan = std::find(p.fans.begin(), p.fans.end(), taxi);
                    if (equal_taxi_to_fan != p.fans.end()) {
                        // came to fan accidentaly
                        taxi_state[ind] = 1;
                        p.fans.erase(std::remove(p.fans.begin(), p.fans.end(), taxi), p.fans.end());
                    }

                    update_targets(targets);
                }

                auto equal_taxi_to_fan = std::find(p.fans.begin(), p.fans.end(), taxi);
                if (equal_taxi_to_fan != p.fans.end()) {
                    // came to fan
                    taxi_state[ind] = 1;
                    p.fans.erase(std::remove(p.fans.begin(), p.fans.end(), taxi), p.fans.end());
                    update_targets(targets);
                }
            }
            num_iterations++;
        }

        this->moves = moves;
    }
};

class RandomSolution : public Solution {
private:
    bool add_direction_move = true;
    double poisson_lambda = 1.0;
    std::vector<double> state_constants;
    std::vector<std::vector<std::vector<Point> > > kd;
public:
    Point transPoint(const Point &a){
        const int BLOCK_SIZE = 2001;
        const int MAX_COORD = 10000;
        return Point((a.X() + MAX_COORD) / BLOCK_SIZE, (a.Y() + MAX_COORD) / BLOCK_SIZE);
    }


    void addPoint(Point t){
        Point q = transPoint(t);
        kd[q.X()][q.Y()].push_back(t);
    }
    void deletePoint(Point t){
        Point q = transPoint(t);
        kd[q.X()][q.Y()].erase(std::remove(kd[q.X()][q.Y()].begin(), kd[q.X()][q.Y()].end(), t), kd[q.X()][q.Y()].end());
    }
    Point getNearestFan(Point t){
        Point q = transPoint(t);
        if(!kd[q.X()][q.Y()].empty())
            return *std::min_element(kd[q.X()][q.Y()].begin(), kd[q.X()][q.Y()].end(),
                             [t](const Point &lhs, const Point &rhs) {
                                 return (lhs - t).len_sqr() < (rhs - t).len_sqr();
                             });
        else{
            return *std::min_element(p.fans.begin(), p.fans.end(),
                                     [t](const Point &lhs, const Point &rhs) {
                                         return (lhs - t).len_sqr() < (rhs - t).len_sqr();
                                     });
        }
    }
    double cnst = 0;
//    double TLE_LIMIT = 5;
    RandomSolution(const Problem &p, double _poisson_lambda) : Solution(p), poisson_lambda(_poisson_lambda) {}

    bool is_direction_better(const std::vector<Point> &one_taxi_targets, int taxi_state, Point taxi,
                             const Point &direction) {
        // IMPROVE
        Point old_taxi = taxi;
        taxi += direction;
        if (abs(taxi.X()) > 10000 || abs(taxi.Y()) > 10000) {
            return false;
        }
        if (std::find(p.taxis.begin(), p.taxis.end(), taxi) != p.taxis.end()) {
            return false;
        }

        if (taxi_state == 0 && std::find(p.zones.begin(), p.zones.end(), taxi) != p.zones.end()) return false;
        if (taxi_state == 1 && std::find(p.fans.begin(), p.fans.end(), taxi) != p.fans.end()) return false;

        if (one_taxi_targets.empty()) {
            return true;
        }

        auto nearest_after = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                              [&taxi](const Point &lhs, const Point &rhs) {
                                                  return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                              });
        auto nearest_before = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                               [&old_taxi](const Point &lhs, const Point &rhs) {
                                                   return (lhs - old_taxi).len_sqr() < (rhs - old_taxi).len_sqr();
                                               });
        if (add_direction_move) {
            return (*nearest_after - taxi).len() + cnst * direction.len() / (p.taxis.size()) <=
                   (*nearest_before - old_taxi).len();
        } else {
            return (*nearest_after - taxi).len_sqr() <=
                   (*nearest_before - old_taxi).len_sqr();
        }
    }

    Move choose_best_move(const std::vector<Point> &taxis, const std::vector<int> &taxi_state,
                          const std::vector<std::vector<Point>> &targets, const Point &direction) {
        Move move(direction, std::vector<size_t>());
        for (size_t taxi_ind = 0; taxi_ind < taxis.size(); taxi_ind++) {
            if (is_direction_better(targets[taxi_state[taxi_ind]], taxi_state[taxi_ind], taxis[taxi_ind], direction)) {
                move.indices.push_back(taxi_ind);
            }
        }
        return move;
    }


    void update_targets(std::vector<std::vector<Point>> &targets) {
        targets[0] = p.fans;
    }

    void add_move(const Move &move, std::vector<Move> &moves, double &score) {
        score += eval_move(move);
        moves.push_back(move);
    }

    void solve() override {

        const int gen_seed_constant = 11;
        unsigned last_gen_seed = 11;

        Problem p_saved = p;
        int num_TLE_iterations = 0;
        double best_TLE_score = 1e15;

        std::vector<Move> best_moves;
        double local_clock = clock();
        for (;; num_TLE_iterations++) {
            kd.resize(11, std::vector<std::vector<Point> > (11, std::vector<Point>(0)));
            if (num_TLE_iterations != 0 && GLobalScope::elapsed_seconds_local(local_clock) > TLE_LIMIT) {
                double TLE_SCORE = (p.num_taxis * (p.num_fans + p.num_zones));
                std::cerr << TLE_LIMIT << std::endl;
                std::cerr << num_TLE_iterations << " " << poisson_lambda << std::endl;
                break;
            }
            cnst = GLobalScope::q(GLobalScope::generator);
            double current_TLE_score = 0;
            last_gen_seed += gen_seed_constant;
            srand(last_gen_seed);
            // GLobalScope::set_rand(last_gen_seed);
            p = p_saved;
            std::vector<Move> moves;
            std::vector<std::vector<Point>> targets;
            std::vector<Point> directions;
            std::vector<int> taxi_state(p.taxis.size(), 0);
            /*
             * 0 - empty
             * 1 - carried passenger
             */
            targets.resize(2);
            targets[1] = p.zones;
            targets[0] = p.fans;
            for(auto it:p.fans)addPoint(it);
            int num_iterations = 0;
            bool flag = true;
            assert(!p.fans.empty());
            // add_direction_move = false;
            while (!p.fans.empty() || std::count(taxi_state.begin(), taxi_state.end(), 1) != 0) {
                cnst = GLobalScope::q(GLobalScope::generator);
                if (num_iterations > 10 * p.num_fans) {
                    assert(false);
                    add_direction_move = false;
                }

                directions.clear();
                size_t directions_size = 0;
                for (size_t taxi_ind = 0; taxi_ind < p.taxis.size(); taxi_ind++) {
                    directions_size += targets[taxi_state[taxi_ind]].size();
                }
                directions.reserve(directions_size);

                for (size_t taxi_ind = 0; taxi_ind < p.taxis.size(); taxi_ind++) {
                    for (auto &&target : targets[taxi_state[taxi_ind]]) {
                        directions.emplace_back(target - p.taxis[taxi_ind]);
                    }
                }
                int dir_ind;
                if(poisson_lambda > 0) {
                    dir_ind = 0;
                    directions[0] = *std::min_element(directions.begin(), directions.end(),
                                                      [](const Point &lhs, const Point &rhs) {
                                                          return lhs.len_sqr() < rhs.len_sqr();
                                                      });
                }else{
                    dir_ind = rand() & 1;
                    if(directions.size() == 1){
                        dir_ind = 0;
                    }
                    std::nth_element(directions.begin(), directions.begin() + dir_ind, directions.end(),
                                     [](const Point &lhs, const Point &rhs) {
                                         return lhs.len_sqr() < rhs.len_sqr();
                                     });
                }

                assert(!directions.empty());


                Move move = choose_best_move(p.taxis, taxi_state, targets, directions[dir_ind]);
                if (move.indices.empty()) {
                    add_direction_move = false;
                    move = choose_best_move(p.taxis, taxi_state, targets, directions[dir_ind]);
                    add_direction_move = true;
                }
                for (auto taxi_ind:move.indices) {
                    p.taxis[taxi_ind] += move.delta;
                }

                add_move(move, moves, current_TLE_score);
                // moves.push_back(move);


                for (size_t ind = 0; ind < p.taxis.size(); ++ind) {
                    auto taxi = p.taxis[ind];
                    auto equal_taxi_to_zone = std::find(p.zones.begin(), p.zones.end(), taxi);
                    if (equal_taxi_to_zone != p.zones.end()) {
                        // came to zone
                        taxi_state[ind] = 0;
                        for (auto &move_direction : GLobalScope::best_move_points) {
                            std::vector<Point> empty_vector;
                            if (is_direction_better(empty_vector, 0, taxi, move_direction) &&
                                std::count(p.taxis.begin(), p.taxis.end(), taxi + move_direction) == 0) {
                                p.taxis[ind] += move_direction;
                                taxi += move_direction;
                                add_move(Move(move_direction, std::vector<size_t>(1, ind)), moves, current_TLE_score);
                                break;
                            }
                        }

                        auto equal_taxi_to_fan = std::find(p.fans.begin(), p.fans.end(), taxi);
                        if (getNearestFan(taxi) == taxi) {
                            // came to fan accidentaly
                            taxi_state[ind] = 1;
                            p.fans.erase(std::remove(p.fans.begin(), p.fans.end(), taxi), p.fans.end());
                            deletePoint(taxi);
                        }

                        update_targets(targets);
                    }


                    if (getNearestFan(taxi) == taxi) {
                        // came to fan
                        taxi_state[ind] = 1;
                        p.fans.erase(std::remove(p.fans.begin(), p.fans.end(), taxi), p.fans.end());
                        deletePoint(taxi);
                        update_targets(targets);
                    }
                }
                num_iterations++;
                if (current_TLE_score > GLobalScope::BEST_SCORE) {
                    break;
                }
            }
            if (current_TLE_score < best_TLE_score) {
                best_moves = moves;
                best_TLE_score = current_TLE_score;
                GLobalScope::BEST_SCORE = std::min(best_TLE_score, GLobalScope::BEST_SCORE);
            }
            // break;
        }
        this->moves = best_moves;
    }
};

void run(std::istream &in, std::ostream &out) {
    Problem p;
    in >> p;

    std::map<std::string, std::unique_ptr<Solution>> solutions;


    // std::set<std::string> good_solutions = {"2Ig4", "2Ig6", "2Ig2", "6Ig4", "4Ig1", "4g1", "2g3"};
    std::set<std::string> good_solutions = {"2g3", "2Ig4", "2Ig2", "2g4"};
    for (int mask = 0; mask < (1 << 3); mask++) {
        // if(!good.count(mask))continue;
        for (size_t dir_look = 1; dir_look <= 6; dir_look++) {
            for (bool improved_func:{false, true}) {
                std::string solution_name =
                        std::to_string(mask) + ((improved_func) ? "I" : "") + "g" + std::to_string(dir_look);
                if (good_solutions.count(solution_name)) {
                    solutions[solution_name] = std::unique_ptr<Solution>(
                            new GreedySolution(p, dir_look, improved_func, 1, mask));
                }
            }

        }
    }

    GLobalScope::c_start = clock();
    solutions["zrandom"] = std::unique_ptr<Solution>(new RandomSolution(p, 5));
    solutions["zrandomL"] = std::unique_ptr<Solution>(new RandomSolution(p, -1.0));
    double all_time = clock();
    double rand_time = 0;
    double MAX_TLE = 1.9;
    int rnd_cnt = 0;
    for (auto &sol:solutions) {
        if (sol.first.front() != 'z') {
            sol.second->solve();
            sol.second->evaluate();
            if (sol.second->score < 1e-4) {
                sol.second->score = 1e15;
            }
            GLobalScope::BEST_SCORE = std::min(sol.second->score, GLobalScope::BEST_SCORE);
        }else{
            rnd_cnt++;
        }
    }
    for (auto &sol:solutions) {
        double local_start = 0;
        if (sol.first.front() == 'z') {
            // std::cerr << "here";
            // std::cerr << ">>>>" << (MAX_TLE - GLobalScope::elapsed_seconds_local(all_time))/rnd_cnt << std::endl;
            local_start = clock();
            sol.second->TLE_LIMIT = (MAX_TLE - GLobalScope::elapsed_seconds_local(all_time))/rnd_cnt;
            sol.second->solve();
            sol.second->evaluate();
            if (sol.second->score < 1e-4) {
                sol.second->score = 1e15;
            }
            std::cerr << ">" << GLobalScope::elapsed_seconds_local(local_start) << std::endl;
            rand_time += GLobalScope::elapsed_seconds_local(local_start);
            rnd_cnt--;
        }
    }


    std::vector<std::string> solution_names;
    solution_names.reserve(solutions.size());
    for (auto const &imap: solutions)
        solution_names.push_back(imap.first);


    auto best_solution_name = *std::min_element(solution_names.begin(), solution_names.end(),
                                                [&solutions](const std::string &lhs,
                                                             const std::string &rhs) {
                                                    return solutions[lhs]->score <= solutions[rhs]->score;
                                                });

    std::vector<double> scores;
    std::map<std::string, int> solutions_stat;
    for (auto &sol:solutions) {
        scores.push_back(sol.second->score);
    }

    auto ans = solutions[best_solution_name]->moves;
    out << ans.size() << std::endl;
    for (auto &move:ans) {
        out << move << std::endl;
    }
}

void genTests(int num_tests, const std::string &folder_name, int seed) {
    std::mt19937 gen(seed);
    for (int test_ind = 0; test_ind < num_tests; ++test_ind) {
        std::ofstream fout(folder_name + "/" + std::to_string(test_ind) + ".in");
        std::uniform_int_distribution<int> dist20(1, 20);
        std::uniform_int_distribution<int> dist500(1, 500);
        std::uniform_int_distribution<int> dist10000(1, 10000);
        int t, p, z, s;
        do {
            t = dist20(gen);
            p = dist500(gen);
            z = dist20(gen);
            s = dist10000(gen);
        } while (t + p + z > (2 * s + 1) * (2 * s + 1));
        std::uniform_int_distribution<int> distS(-s, s);
        std::set<Point> points;
        std::set<Point> allpoints;
        fout << t << std::endl;
        do {
            Point point;
            do {
                point = Point({distS(gen), distS(gen)});
            } while (allpoints.count(point));
            points.insert(point);
            allpoints.insert(point);
        } while (points.size() < t);
        for (const auto &point:points) {
            fout << point << std::endl;
        }

        points.clear();
        fout << p << std::endl;
        do {
            Point point;
            do {
                point = Point({distS(gen), distS(gen)});
            } while (allpoints.count(point));
            points.insert(point);
            allpoints.insert(point);
        } while (points.size() < p);
        for (const auto &point:points) {
            fout << point << std::endl;
        }

        points.clear();
        fout << z << std::endl;
        do {
            Point point;
            do {
                point = Point({distS(gen), distS(gen)});
            } while (allpoints.count(point));
            points.insert(point);
            allpoints.insert(point);
        } while (points.size() < z);
        for (const auto &point:points) {
            fout << point << std::endl;
        }

    }
}

void runTests(int num_tests, const std::string &folder_name) {
    std::ofstream fout;
    fout.open(folder_name + "_test-results.log");
    std::clock_t c_start = 0, c_end = 0;
    double time_sum = 0;
    std::map<std::string, std::pair<int, std::pair<double, double>>> solutions_map;
    const std::vector<std::vector<std::string>> groups = {
            {"0g1", "2g2",  "2g3",  "2Ig2", "2Ig3"},
            {"0g1", "2g2",  "2g3",  "2Ig2", "2Ig3", "zrandom"},
            {"0g1", "0Ig1", "2g3",  "2Ig4", "zrandom"},
            {"0g1", "0Ig1", "2Ig4", "zrandom"},
            {"0g1", "2g2",  "2g3",  "2Ig2", "2Ig3", "zrandomL"},
            {"0g1", "0Ig1", "2g3",  "2Ig4", "zrandomL"},
            {"0g1", "0Ig1", "2Ig4", "zrandomL"},
            {"0g1", "0Ig1", "2Ig4", "zrandom", "zrandomL"}
    };
    std::vector<double> groups_score(groups.size());
    const int score_padding = 15;
    const int penalty_padding = 25;
    const int name_padding = 6;
    double all_time_max = 0;
    for (int test_ind = 0; test_ind < num_tests; ++test_ind) {
        std::cerr << "Running test index " << test_ind << std::endl;
        if (test_ind != 0) {
            time_sum += (c_end - c_start);
            std::cerr << "-- Time left(appr): ";
            double sec_left = (num_tests - test_ind) * (time_sum / test_ind) / CLOCKS_PER_SEC;
            if (sec_left > 60) {
                std::cerr << static_cast<int>(sec_left / 60) << "m";
            }
            std::cerr << sec_left - 60 * (floor(sec_left / 60)) << " s" << std::endl;
        }
        c_start = std::clock();
        std::ifstream fin(folder_name + "/" + std::to_string(test_ind) + ".in");
        double all_time = clock();
        Problem p;
        fin >> p;

        std::map<std::string, std::unique_ptr<Solution>> solutions;
        // {"TTg1",  "TIg1",  "TTg2",  "TIg2", "Ig1", "Ig2"},

        // const std::set<int> good = {0,1,2,5,7,40,44,60,31,61};
        std::set<std::string> good_solutions = {"0g1", "2g2", "2g3", "2g4", "2Ig2", "2Ig3", "2Ig4", "0Ig1", "2g3",
                                                "0g1", "2g2", "2g3", "2g4", "2Ig2", "2Ig3", "2Ig4"};
        for (int mask = 0; mask < (1 << 2); mask++) {
            for (size_t dir_look = 1; dir_look <= 6; dir_look++) {
                for (bool improved_func:{false, true}) {
                    std::string solution_name =
                            std::to_string(mask) + ((improved_func) ? "I" : "") + "g" + std::to_string(dir_look);
                    if (good_solutions.count(solution_name)) {
                        solutions[solution_name] = std::unique_ptr<Solution>(
                                new GreedySolution(p, dir_look, improved_func, 1, mask));
                    }
                }
            }
        }

        GLobalScope::c_start = clock();
        solutions["zrandom"] = std::unique_ptr<Solution>(new RandomSolution(p, 1.0));
        solutions["zrandomL"] = std::unique_ptr<Solution>(new RandomSolution(p, -1.0));
        double rand_time = 0;
        double MAX_TLE = 1.3;
        for (auto &sol:solutions) {
            if (sol.first.front() != 'z') {
                sol.second->solve();
                sol.second->evaluate();
                if (sol.second->score < 1e-4) {
                    sol.second->score = 1e15;
                }
            }
        }
        for (auto &sol:solutions) {
            double local_start = 0;
            if (sol.first.front() == 'z') {
                local_start = clock();
                sol.second->TLE_LIMIT = (MAX_TLE - GLobalScope::elapsed_seconds_local(local_start));
                sol.second->solve();
                sol.second->evaluate();
                if (sol.second->score < 1e-4) {
                    sol.second->score = 1e15;
                }
                rand_time += GLobalScope::elapsed_seconds_local(local_start);
            }
        }

        all_time_max = std::max(all_time_max, GLobalScope::elapsed_seconds_local(all_time));
        fout << "Best on test #" << std::setw(3) << test_ind << "all_time on test:"
             << GLobalScope::elapsed_seconds_local(all_time) << "s. Random on test:" << rand_time << "s  " << std::endl;
        fout << "Max time" << all_time_max << std::endl;
        std::vector<double> scores;
        std::map<std::string, int> solutions_stat;

        std::vector<std::string> solution_names;
        solution_names.reserve(solutions.size());
        for (auto const &imap: solutions) {
            solution_names.push_back(imap.first);
        }
        auto best_solution_name = *std::min_element(solution_names.begin(), solution_names.end(),
                                                    [&solutions](const std::string &lhs,
                                                                 const std::string &rhs) {
                                                        return solutions[lhs]->score <= solutions[rhs]->score;
                                                    });

        for (auto &sol:solutions) {
            scores.push_back(sol.second->score);
            solutions_map[sol.first].second.first +=
                    100 * (sol.second->score / (1e-4 + solutions[best_solution_name]->score));
        }


        for (auto &sol:solutions) {
            solutions_map[sol.first].second.second += pow(solutions[best_solution_name]->score - sol.second->score, 2);
        }

        solutions_map[best_solution_name].first++;

        fout << std::fixed << std::setprecision(3);
        for (int group_ind = 0; group_ind < groups.size(); ++group_ind) {
            auto group = groups[group_ind];
            auto best_solution_name_in_group = *std::min_element(group.begin(), group.end(),
                                                                 [&solutions](const std::string &lhs,
                                                                              const std::string &rhs) {
                                                                     return solutions[lhs]->score <
                                                                            solutions[rhs]->score;
                                                                 });

            groups_score[group_ind] += solutions_map[best_solution_name_in_group].second.first / (test_ind + 1);
        }

        c_end = std::clock();
        fout << std::endl;
        fout << std::string(160, '-');
        fout << "\nBest solutions:" << std::endl;
        std::vector<std::pair<std::string, std::pair<int, std::pair<double, double>>>> vec_out;
        for (const auto &it:solutions_map) {
            vec_out.push_back(it);
        }
        std::sort(vec_out.begin(), vec_out.end(),
                  [](const std::pair<std::string, std::pair<int, std::pair<double, double>>> &lhs,
                     const std::pair<std::string, std::pair<int, std::pair<double, double>>> &rhs) {
                      return lhs.second.second.first < rhs.second.second.first;
                  });

        for (auto it:vec_out) {
            fout << std::setw(name_padding) << it.first << ":" << std::setw(3) << it.second.first;
            fout << " score: " << std::setw(score_padding) << it.second.second.first / (test_ind + 1) << " penalty"
                 << std::setw(penalty_padding)
                 << it.second.second.second << std::endl;
        }

        fout << "\nGroup scores:" << std::endl;


        std::vector<std::pair<double, int>> groups_sorted(groups.size());
        for (int groups_ind = 0; groups_ind < groups.size(); ++groups_ind) {
            groups_sorted[groups_ind] = {groups_score[groups_ind], groups_ind};
        }
        sort(groups_sorted.begin(), groups_sorted.end());
        for (int groups_ind = 0; groups_ind < groups.size(); groups_ind++) {
            auto &group = groups[groups_sorted[groups_ind].second];
            // fout.width(70);
            std::stringstream ss;
            ss << std::left << "{";
            for (int i = 0; i < group.size(); i++) {
                ss << std::left << std::setw(name_padding) << std::left << group[i] << std::left;
                if (i != group.size() - 1) {
                    ss << ",";
                }
            }
            ss << std::left << "}> ";

            fout << std::setw(60) << std::left << ss.str() << std::right
                 << groups_score[groups_sorted[groups_ind].second] / (test_ind + 1)
                 << std::endl;
        }
        fout << std::string(160, '#') << std::string(3, '\n') << std::endl;
    }
}

int main() {
    std::cin.sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::ifstream fin("manual-tests/input.txt");
    std::ofstream fout("manual-tests/output.txt");
    GLobalScope::generate_best_move_points();

#ifndef LOCAL
    run(fin, fout);
#else
    #ifdef TEST
    int num_tests = 300;
    genTests(num_tests, "big-tests", 145672);
    runTests(num_tests, "big-tests");
#else
    run(fin, fout);
#endif
#endif
    return 0;
}

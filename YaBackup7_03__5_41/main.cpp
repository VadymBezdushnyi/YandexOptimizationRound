#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <cassert>
#include <random>
#include <set>
#include <iomanip>


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
        for (auto &zone:p.zones)
            in >> zone;

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

class SolutionGreedy2 : public Solution {
private:
    std::vector<Point> best_move_points;
    size_t num_direction_to_look;
    bool add_direction_move;
public:
    SolutionGreedy2(const Problem &p, size_t num_d, bool add_direction_move) : Solution(p), num_direction_to_look(num_d) {

    }

    void generate_best_move_points() {
        for (int i = -20; i < 20; i++) {
            for (int j = -20; j < 20; j++) {
                best_move_points.emplace_back(i, j);
            }
        }
        sort(best_move_points.begin(), best_move_points.end(),
             [](const Point &lhs, const Point &rhs) {
                 return lhs.len_sqr() < rhs.len_sqr();
             });
    }

    bool is_direction_better(const std::vector<Point> &one_taxi_targets, int taxi_state, Point taxi,
                             const Point &direction) {
        // IMPROVE
        auto nearest_now = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                            [taxi](const Point &lhs, const Point &rhs) {
                                                return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                            });
        Point old_taxi = taxi;
        taxi += direction;
        auto nearest_after = std::min_element(one_taxi_targets.begin(), one_taxi_targets.end(),
                                              [taxi](const Point &lhs, const Point &rhs) {
                                                  return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                              });

        if (abs(taxi.X()) > 10000 || abs(taxi.Y()) > 10000) {
            return false;
        }
        auto taxi_in_fans = std::find(p.fans.begin(), p.fans.end(), taxi);
        auto taxi_in_zones = std::find(p.zones.begin(), p.zones.end(), taxi);
        if (taxi_state == 1 && taxi_in_fans != p.fans.end()) return false;
        if (taxi_state == 0 && taxi_in_zones != p.zones.end()) return false;
        if (std::find(p.taxis.begin(), p.taxis.end(), taxi) != p.taxis.end()) {
            return false;
        }
        if (one_taxi_targets.empty()) {

            return true;
        }
        if(add_direction_move){
            return (*nearest_after - taxi).len() + direction.len()/p.num_taxis <= (*nearest_now - old_taxi).len();
        }else{
            return (*nearest_after - taxi).len() <= (*nearest_now - old_taxi).len();
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

    double eval_targets(const std::vector<Point> &taxis,
                        const std::vector<std::vector<Point>> &targets,
                        const std::vector<int> &taxi_state) {
        double score = 0;
        for (size_t taxi_ind = 0; taxi_ind < taxis.size(); taxi_ind++) {
            auto taxi = taxis[taxi_ind];
            if (targets[taxi_state[taxi_ind]].empty()) continue;
            auto nearest_target = std::min_element(targets[taxi_state[taxi_ind]].begin(),
                                                   targets[taxi_state[taxi_ind]].end(),
                                                   [&taxi](const Point &lhs, const Point &rhs) {
                                                       return (lhs - taxi).len_sqr() < (rhs - taxi).len_sqr();
                                                   });
            score += nearest_target->len();
        }
        return score;
    }

    std::pair<double, Move> score_direction(std::vector<Point> taxis, const std::vector<int> &taxi_state,
                                            const std::vector<std::vector<Point>> &targets, const Point &direction) {
        double score = 0;
        Move move = choose_best_move(taxis, taxi_state, targets, direction);
        for (auto taxi_ind:move.indices) {
            taxis[taxi_ind] = taxis[taxi_ind] + direction;
            if (std::find(p.taxis.begin(), p.taxis.end(), taxis[taxi_ind]) != p.taxis.end()) {
                score = -1e16;
            }
        }
        for (const auto &taxi:taxis) {
            if (std::find(p.taxis.begin(), p.taxis.end(), taxi) != p.taxis.end()) {
                score = -1e16;
            }
        }
        score += eval_targets(taxis, targets, taxi_state);
        score -= eval_move(move);
        if (move.indices.empty()) {
            score = -1e16;//TODO
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
        generate_best_move_points();
        /*
         * 0 - empty
         * 1 - carried passenger
         */
        targets.resize(2);
        targets[1] = p.zones;
        targets[0] = p.fans;
        while (!p.fans.empty() || std::count(taxi_state.begin(), taxi_state.end(), 1) != 0) {
            // directions.reserve(p.taxis.size() * targets.size());
            directions.clear();
            for (size_t taxi_ind = 0; taxi_ind < p.taxis.size(); taxi_ind++) {
                for (auto &&target : targets[taxi_state[taxi_ind]]) {
                    directions.push_back(target - p.taxis[taxi_ind]);
                }
            }

            sort(directions.begin(), directions.end(), [](const Point &lhs, const Point &rhs) {
                return lhs.len_sqr() < rhs.len_sqr();
            });

            directions.resize(std::min(static_cast<size_t>(num_direction_to_look), directions.size()));

            std::vector<std::pair<double, Move>> scores(directions.size());
            for (size_t dir_ind = 0; dir_ind < directions.size(); dir_ind++) {
                auto res = score_direction(p.taxis, taxi_state, targets, directions[dir_ind]);
                scores[dir_ind].first = res.first;
                scores[dir_ind].second = res.second;
            }

            auto best_direction = std::max_element(scores.begin(), scores.end(),
                                                   [](std::pair<double, Move> &lhs, std::pair<double, Move> &rhs) {
                                                       return lhs.first < rhs.first;
                                                   });

            Move move = best_direction->second;
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
                    for (auto &move_direction : best_move_points) {
                        std::vector<Point> empty_vector;
                        if (is_direction_better(empty_vector, 0, taxi, move_direction) &&
                            std::count(p.taxis.begin(), p.taxis.end(), taxi + move_direction) == 0) {
                            p.taxis[ind] += move_direction;
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
        }
        this->moves = moves;
    }
};

void run(std::istream &in, std::ostream &out) {
    Problem p;
    in >> p;
    std::vector<std::unique_ptr<Solution>> solutions;
    solutions.emplace_back(std::unique_ptr<Solution>(new SolutionGreedy2(p, 1, false)));
    solutions.emplace_back(std::unique_ptr<Solution>(new SolutionGreedy2(p, 2, false)));

    solutions.emplace_back(std::unique_ptr<Solution>(new SolutionGreedy2(p, 1, true)));
    solutions.emplace_back(std::unique_ptr<Solution>(new SolutionGreedy2(p, 2, true)));

    std::vector<double> scores;
    for (auto &sol:solutions) {
        sol->solve();
        sol->evaluate();
        scores.push_back(sol->score);
    }

    auto best_solution = std::min_element(solutions.begin(), solutions.end(),
                                          [](const std::unique_ptr<Solution> &lhs,
                                             const std::unique_ptr<Solution> &rhs) {
                                              return lhs->score < rhs->score;
                                          });

    auto ans = (*best_solution)->moves;
    out << ans.size() << std::endl;
    for (auto &move:ans) {
        out << move << std::endl;
    }
}

void genTests(int num_tests, const std::string &folder_name)
{
    std::mt19937 gen(1834);
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
void runTests(int num_tests, const std::string &folder_name){
    std::ofstream fout(folder_name + "_test-results.log");
    std::clock_t c_start = 0, c_end = 0;
    double time_sum = 0;
    std::map<std::string, std::pair<int, double>> solutions_map;
    for (int test_ind = 0; test_ind < num_tests; ++test_ind) {
        std::cerr << "Running test index " << test_ind << std::endl;
        if(test_ind != 0){
            time_sum += (c_end - c_start);
            std::cerr << "-- Time left(appr): " <<
                      (num_tests - test_ind)*(time_sum/test_ind) / CLOCKS_PER_SEC << " s" << std::endl;
        }
        c_start = std::clock();
        std::ifstream fin(folder_name + "/" + std::to_string(test_ind) + ".in");
        Problem p;
        fin >> p;

        std::vector<std::pair<std::string, std::unique_ptr<Solution>>> solutions;
        solutions.emplace_back("g1", std::unique_ptr<Solution>(new SolutionGreedy2(p, 1, false)));
        solutions.emplace_back("g2", std::unique_ptr<Solution>(new SolutionGreedy2(p, 2, false)));
        solutions.emplace_back("g3", std::unique_ptr<Solution>(new SolutionGreedy2(p, 3, false)));

        solutions.emplace_back("Ig1", std::unique_ptr<Solution>(new SolutionGreedy2(p, 1, true)));
        solutions.emplace_back("Ig2", std::unique_ptr<Solution>(new SolutionGreedy2(p, 2, true)));
        solutions.emplace_back("Ig3", std::unique_ptr<Solution>(new SolutionGreedy2(p, 3, true)));

        std::vector<double> scores;
        std::map<std::string, int> solutions_stat;
        for (auto &sol:solutions) {
            sol.second->solve();
            sol.second->evaluate();
            scores.push_back(sol.second->score);
            solutions_map[sol.first].second += sol.second->score;
        }


        auto best_solution = std::min_element(solutions.begin(), solutions.end(),
                                              [](const std::pair<std::string, std::unique_ptr<Solution>> &lhs,
                                                 const std::pair<std::string, std::unique_ptr<Solution>> &rhs) {
                                                  return lhs.second->score <= rhs.second->score;
                                              });
        solutions_map[best_solution->first].first++;


        fout << std::fixed << std::setprecision(3);
        fout << "Best on test #" << std::setw(3) << test_ind << ":" << std::setw(6) << best_solution->first << ">";

        for(auto score:scores){

            if(score == best_solution->second->score){
                fout << std::setw(13) << score << std::setw(2) << "@ ";
            }else{
                fout << std::setw(13) << score << std::setw(2) << " ";
            }
        }
        c_end = std::clock();
        fout << std::endl;
    }
    fout << "\nBest solutions:" << std::endl;
    for(auto it:solutions_map){
        fout << std::setw(5) << it.first << ":" << std::setw(3) << it.second.first;
        fout << " score: " << std::setw(3) << it.second.second/num_tests  << std::endl;
    }
}
int main() {
    std::cin.sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::ifstream fin("input.txt");
    std::ofstream fout("output.txt");
    run(fin, fout);
#ifdef TEST
    int num_tests = 100;
    genTests(num_tests, "big-tests");
    runTests(num_tests, "big-tests");
#else
    run(fin, fout);
#endif
    return 0;
}
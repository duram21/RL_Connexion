#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/// 게임 보드의 열 번호
enum class Column {
    A,
    B,
    C,
    D,
    E,
    F
};
/// 게임 보드의 행 번호
enum class Row {
    _1,
    _2,
    _3,
    _4,
    _5,
    _6
};

/// 게임 보드의 부호
enum class Sign {
    Minus,
    Plus
};

/// 게임 타일의 색
enum class Color {
    R,
    G,
    B,
    Y
};

/// 게임 타일의 문양
enum class Symbol {
    _1,
    _2,
    _3,
    _4
};

/// 게임 보드의 칸
struct Cell {
    /// 게임 보드의 열
    Column col;
    /// 게임 보드의 행
    Row row;
    /// 게임 보드의 부호
    Sign sign;

    Cell(Column col, Row row, Sign sign) : col(col), row(row), sign(sign) {}

    /// 해당 칸이 금지된 칸인 a1-, a4-, c3+, c6+, d1-, d4-, f3+, f6+ 중 하나가 아니라면 true를 반환
    bool is_valid() const {
        return !((col == Column::A && row == Row::_1 && sign == Sign::Minus) ||
                 (col == Column::A && row == Row::_4 && sign == Sign::Minus) ||
                 (col == Column::C && row == Row::_3 && sign == Sign::Plus) ||
                 (col == Column::C && row == Row::_6 && sign == Sign::Plus) ||
                 (col == Column::D && row == Row::_1 && sign == Sign::Minus) ||
                 (col == Column::D && row == Row::_4 && sign == Sign::Minus) ||
                 (col == Column::F && row == Row::_3 && sign == Sign::Plus) ||
                 (col == Column::F && row == Row::_6 && sign == Sign::Plus));
    }

    /// 주어진 두 칸이 인접한 칸인지 여부를 반환
    bool is_adjacent(const Cell& other) const {
        // 두 칸의 행, 열 번호 차이를 계산
        int dr = (int)other.row - (int)row;
        int dc = (int)other.col - (int)col;

        if (sign == Sign::Minus && other.sign == Sign::Plus) {
            return (dr == 0 && dc == 0) || (dr == 0 && dc == -1) || (dr == -1 && dc == 0);
        } else if (sign == Sign::Plus && other.sign == Sign::Minus) {
            return (dr == 0 && dc == 0) || (dr == 0 && dc == 1) || (dr == 1 && dc == 0);
        }
        return false;
    }

    /// 배치 가능한 모든 칸을 반환
    static vector<Cell> get_all_cells() {
        vector<Cell> cells;

        const Column columns[] = {Column::A, Column::B, Column::C, Column::D, Column::E, Column::F};
        const Row rows[] = {Row::_1, Row::_2, Row::_3, Row::_4, Row::_5, Row::_6};
        const Sign signs[] = {Sign::Minus, Sign::Plus};

        for (Column col : columns) {
            for (Row row : rows) {
                for (Sign sign : signs) {
                    Cell cell(col, row, sign);
                    if (cell.is_valid()) {
                        cells.push_back(cell);
                    }
                }
            }
        }
        return cells;
    }

    bool operator==(const Cell& other) const {
        return col == other.col && row == other.row && sign == other.sign;
    }
};

/// 타일을 나타내는 구조체
struct Tile {
    /// 타일의 색
    Color color;
    /// 타일의 문양
    Symbol symbol;

    Tile(Color color, Symbol symbol) : color(color), symbol(symbol) {}

    /// 점수 계산시 두 타일을 같은 타일로 볼 것인가를 반환
    ///
    /// 인자 목록
    /// - other: 비교할 타일
    /// - first: 선후공 (true: 선공 / 문양을 사용, false: 후공 / 색을 사용)
    bool is_same(const Tile& other, bool first) const {
        if (first) {
            return symbol == other.symbol;
        } else {
            return color == other.color;
        }
    }

    bool operator==(const Tile& other) const {
        return color == other.color && symbol == other.symbol;
    }
};

/// 게임 상태를 관리하는 구조체
class Game {
   private:
    /// 내 타일 목록
    vector<Tile> my_tiles;
    /// 상대 타일 목록
    vector<Tile> opp_tiles;
    /// 선후공 (true: 선공, false: 후공)
    bool is_first;
    /// 현재 보드에 배치된 칸과 타일 목록
    vector<pair<Cell, Tile>> board;

   public:
    Game(vector<Tile> my_tiles, vector<Tile> opp_tiles, bool is_first)
        : my_tiles(my_tiles), opp_tiles(opp_tiles), is_first(is_first) {}

    // ================================ [필수 구현] ================================
    /// 현재 상태를 기반으로 놓아야 할 칸과 타일을 계산함
    ///
    /// 인자 목록
    /// - my_time: 내 프로그램의 남은 시간
    /// - opp_time: 상대 프로그램의 남은 시간
    pair<Cell, Tile> calculate_move(int my_time, int opp_time) {
        (void)my_time, (void)opp_time;
        optional<pair<Cell, Tile>> best_move = nullopt;
        int best_score = 0;

        // 가능한 모든 행동중 최대 점수를 찾음
        for (const Tile& tile : my_tiles) {
            for (const Cell& cell : Cell::get_all_cells()) {
                // 이미 배치된 칸인지 확인
                if (find_if(board.begin(), board.end(), [&](const pair<Cell, Tile>& p) { return p.first == cell; }) != board.end()) {
                    continue;
                }
                vector<pair<Cell, Tile>> test_board = board;
                test_board.push_back({cell, tile});

                // 점수를 계산 후 최고 점수를 갱신
                int score = calculate_score(test_board, is_first);
                if (score > best_score) {
                    best_score = score;
                    best_move = {cell, tile};
                }
            }
        }

        assert(best_move && "No valid move found");
        return best_move.value();
    }
    // ============================== [필수 구현 끝] ==============================

    /// 자신 혹은 상대의 행동을 기반으로 상태를 업데이트 함
    ///
    /// 인자 목록
    /// - my_action: 자신의 행동 여부 (true: 자신, false: 상대)
    /// - action: 자신 혹은 상대가 배치한 칸과 타일
    /// - get: 자신 혹은 상대가 뽑은 타일. 없으면 nullopt.
    /// - used_time: 상대가 사용한 시간. 자신의 행동인 경우 nullopt.
    void update_action(bool my_action, pair<Cell, Tile> action, optional<Tile> get, optional<int> used_time) {
        (void)used_time;

        // 전체 보드에 칸과 타일 추가
        board.push_back(action);

        if (my_action) {
            // 내 타일에서 사용한 타일 제거
            auto it = find(my_tiles.begin(), my_tiles.end(), action.second);
            assert(it != my_tiles.end());
            my_tiles.erase(it);

            // 뽑아온 타일이 있으면 타일 목록에 타일 추가
            if (get) {
                my_tiles.push_back(*get);
            }
        } else {
            // 상대 타일에서 사용한 타일 제거
            auto it = find(opp_tiles.begin(), opp_tiles.end(), action.second);
            assert(it != opp_tiles.end());
            opp_tiles.erase(it);

            // 뽑아온 타일이 있으면 타일 목록에 타일 추가
            if (get) {
                opp_tiles.push_back(*get);
            }
        }
    }

    /// 보드와 선공 여부를 기반으로 점수를 계산함
    ///
    /// 인자 목록
    /// - board: 보드
    /// - first: 선후공 (true: 선공 / 문양을 사용, false: 후공 / 색을 사용)
    static int calculate_score(const vector<pair<Cell, Tile>>& board, bool first) {
        if (board.empty()) {
            return 0;
        }

        int n = board.size();
        vector<vector<bool>> adj(n, vector<bool>(n, false));

        // 선후공을 고려하여 타일들의 인접행렬을 계산함
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j || (board[i].first.is_adjacent(board[j].first) &&
                               board[i].second.is_same(board[j].second, first))) {
                    adj[i][j] = true;
                }
            }
        }

        // Warshall algorithm을 사용하여 연결 성분을 계산
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (adj[i][k] && adj[k][j]) {
                        adj[i][j] = true;
                    }
                }
            }
        }

        // 각 타일의 점수는 같은 연결 성분에 속한 타일의 수, 총점은 모든 타일의 점수 총합
        int total_score = 0;
        for (int i = 0; i < n; i++) {
            total_score += count(adj[i].begin(), adj[i].end(), true);
        }
        return total_score;
    }
};

// ================================ 입출력 및 파싱을 위한 Helper 시작 ================================

optional<Column> column_from_char(char c) {
    if ('a' <= c && c <= 'f') {
        return Column(c - 'a');
    }
    return nullopt;
}
char column_to_char(Column col) {
    return 'a' + (int)col;
}

optional<Row> row_from_char(char c) {
    if ('1' <= c && c <= '6') {
        return Row(c - '1');
    }
    return nullopt;
}
char row_to_char(Row row) {
    return '1' + (int)row;
}

optional<Sign> sign_from_char(char c) {
    switch (c) {
        case '-':
            return Sign::Minus;
        case '+':
            return Sign::Plus;
        default:
            return nullopt;
    }
}

char sign_to_char(Sign sign) {
    switch (sign) {
        case Sign::Minus:
            return '-';
        case Sign::Plus:
            return '+';
    }
    assert(false);
}

optional<Color> color_from_char(char c) {
    switch (c) {
        case 'R':
            return Color::R;
        case 'G':
            return Color::G;
        case 'B':
            return Color::B;
        case 'Y':
            return Color::Y;
        default:
            return nullopt;
    }
}

char color_to_char(Color color) {
    switch (color) {
        case Color::R:
            return 'R';
        case Color::G:
            return 'G';
        case Color::B:
            return 'B';
        case Color::Y:
            return 'Y';
    }
    assert(false);
}

optional<Symbol> symbol_from_char(char c) {
    switch (c) {
        case '1':
            return Symbol::_1;
        case '2':
            return Symbol::_2;
        case '3':
            return Symbol::_3;
        case '4':
            return Symbol::_4;
        default:
            return nullopt;
    }
}

char symbol_to_char(Symbol symbol) {
    switch (symbol) {
        case Symbol::_1:
            return '1';
        case Symbol::_2:
            return '2';
        case Symbol::_3:
            return '3';
        case Symbol::_4:
            return '4';
    }
    assert(false);
}

optional<Cell> parse_cell(const string& s) {
    if (s.length() != 3) {
        return nullopt;
    }

    auto col = column_from_char(s[0]);
    auto row = row_from_char(s[1]);
    auto sign = sign_from_char(s[2]);

    if (!col || !row || !sign) {
        return nullopt;
    }

    Cell cell(col.value(), row.value(), sign.value());
    if (cell.is_valid()) {
        return cell;
    } else {
        return nullopt;
    }
}

string cell_to_string(const Cell& cell) {
    return string(1, column_to_char(cell.col)) +
           string(1, row_to_char(cell.row)) +
           string(1, sign_to_char(cell.sign));
}

optional<Tile> parse_tile(const string& s) {
    if (s.length() != 2) {
        return nullopt;
    }

    auto color = color_from_char(s[0]);
    auto symbol = symbol_from_char(s[1]);

    if (!color || !symbol) {
        return nullopt;
    }

    return Tile(color.value(), symbol.value());
}

string tile_to_string(const Tile& tile) {
    return string(1, color_to_char(tile.color)) +
           string(1, symbol_to_char(tile.symbol));
}
// ================================ 입출력 및 파싱을 위한 Helper 끝 ================================

/// 표준 입력을 통해 명령어를 처리하는 메인 함수
int main() {
    optional<Game> game = nullopt;
    optional<bool> is_first = nullopt;
    optional<pair<Cell, Tile>> last_move = nullopt;

    string line;
    while (getline(cin, line)) {
        if (line.empty()) {
            continue;
        }

        istringstream iss(line);
        string command;
        iss >> command;

        if (command == "READY") {
            // 게임 시작
            string first_str;
            iss >> first_str;
            is_first = (first_str == "FIRST");
            cout << "OK" << endl;
            cout.flush();
        } else if (command == "INIT") {
            // 준비 단계 시작
            vector<Tile> my_tiles;
            for (int i = 0; i < 5; i++) {
                string tile_str;
                iss >> tile_str;
                auto tile = parse_tile(tile_str);
                assert(tile);
                my_tiles.push_back(*tile);
            }
            vector<Tile> opp_tiles;
            for (int i = 0; i < 5; i++) {
                string tile_str;
                iss >> tile_str;
                auto tile = parse_tile(tile_str);
                assert(tile);
                opp_tiles.push_back(*tile);
            }

            assert(is_first);
            game = Game(my_tiles, opp_tiles, *is_first);
        } else if (command == "TIME") {
            // 배치 단계 시작
            int my_time, opp_time;
            iss >> my_time >> opp_time;

            assert(game);
            last_move = game->calculate_move(my_time, opp_time);

            const auto& [cell, tile] = *last_move;
            cout << "PUT " << cell_to_string(cell) << " " << tile_to_string(tile) << endl;
        } else if (command == "GET") {
            // 배치 단계가 끝날때 뽑아온 타일을 이용해서 상태를 업데이트 함
            string get_str;
            iss >> get_str;

            optional<Tile> get_tile = nullopt;
            if (get_str != "X0") {
                get_tile = parse_tile(get_str);
                assert(get_tile);
            }

            assert(game && last_move);
            game->update_action(true, *last_move, get_tile, nullopt);
        } else if (command == "OPP") {
            // 상대의 행동을 기반으로 상태를 업데이트 함
            string cell_str, tile_str, get_str;
            int opp_time;
            iss >> cell_str >> tile_str >> get_str >> opp_time;

            auto cell = parse_cell(cell_str);
            auto tile = parse_tile(tile_str);
            assert(cell && tile);

            optional<Tile> get_tile = nullopt;
            if (get_str != "X0") {
                get_tile = parse_tile(get_str);
                assert(get_tile);
            }

            assert(game);
            game->update_action(false, {*cell, *tile}, get_tile, opp_time);
        } else if (command == "FINISH") {
            // 게임 종료
            break;
        } else {
            // 알 수 없는 명령어 처리
            cerr << "Invalid command: " << command << endl;
            return 1;
        }
    }

    return 0;
}
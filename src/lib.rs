

// A "negative infinity" score that won't overflow.
const AWFUL_SCORE: i64 = -(1 << 62);

fn is_awful(score: i64) -> bool {
    score < AWFUL_SCORE / 2
}

#[derive(PartialEq, Clone, Copy)]
enum Action {
    Miss,
    Match,
}

#[derive(Clone, Copy)]
struct Score {
    pub last_action_miss: Action,
    pub last_action_match: Action,
    pub miss_score: i64,
    pub match_score: i64,
}

impl Default for Score {
    fn default() -> Self {
        Self {
            last_action_miss: Action::Miss,
            last_action_match: Action::Miss,
            miss_score: AWFUL_SCORE,
            match_score: AWFUL_SCORE,
        }
    }
}

pub fn fuzzy_match(line: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
    let num_line_chars = line.chars().count();
    let num_pattern_chars = pattern.chars().count();

    let mut dp: Vec<Vec<Score>> = Vec::with_capacity(num_pattern_chars + 1);
    for _ in 0..(num_pattern_chars + 1) {
        dp.push(vec![Score::default(); num_line_chars + 1]);
    }

    dp[0][0].miss_score = 0;

    // first line
    for (idx, ch) in line.chars().enumerate() {
        dp[0][idx + 1] = Score {
            miss_score: &dp[0][idx].miss_score - skip_penalty(idx, ch, Action::Miss),
            last_action_miss: Action::Miss,
            match_score: AWFUL_SCORE,
            last_action_match: Action::Miss,
        };
    }

    // build the matrix
    let mut pat_prev_ch = '\0';
    for (pat_idx, pat_ch) in pattern.chars().enumerate() {
        let mut line_prev_ch = '\0';
        for (line_idx, line_ch) in line.chars().enumerate() {
            if line_idx < pat_idx {
                line_prev_ch = line_ch;
                continue;
            }

            let pre_miss = &dp[pat_idx + 1][line_idx];
            let mut match_miss_score = pre_miss.match_score;
            let mut miss_miss_score = pre_miss.miss_score;
            if pat_idx < num_pattern_chars - 1 {
                match_miss_score -= skip_penalty(pat_idx, pat_ch, Action::Match);
                miss_miss_score -= skip_penalty(pat_idx, pat_ch, Action::Miss);
            }

            let (miss_score, last_action_miss) = if match_miss_score > miss_miss_score {
                (match_miss_score, Action::Match)
            } else {
                (miss_miss_score, Action::Miss)
            };

            let pre_match = &dp[pat_idx][line_idx];
            let match_match_score = if allow_match(pat_ch, line_ch, Action::Match) {
                pre_match.match_score
                    + match_bonus(
                        pat_idx,
                        pat_ch,
                        pat_prev_ch,
                        line_idx,
                        line_ch,
                        line_prev_ch,
                        Action::Match,
                    )
            } else {
                AWFUL_SCORE
            };

            let miss_match_score = if allow_match(pat_ch, line_ch, Action::Miss) {
                pre_match.miss_score
                    + match_bonus(
                        pat_idx,
                        pat_ch,
                        pat_prev_ch,
                        line_idx,
                        line_ch,
                        line_prev_ch,
                        Action::Match,
                    )
            } else {
                AWFUL_SCORE
            };

            let (match_score, last_action_match) = if match_match_score > miss_match_score {
                (match_match_score, Action::Match)
            } else {
                (miss_match_score, Action::Miss)
            };

            dp[pat_idx + 1][line_idx + 1] = Score {
                miss_score,
                last_action_miss,
                match_score,
                last_action_match,
            };

            line_prev_ch = line_ch;
        }
        pat_prev_ch = pat_ch;
    }

    // search backwards for the matched indices
    let mut indices_reverse = Vec::new();
    let mut row = num_pattern_chars;
    let mut col = num_line_chars;

    let cell = dp[row][col];
    let (mut last_action, score) = if cell.match_score > cell.miss_score {
        (cell.last_action_match, cell.match_score)
    } else {
        (cell.last_action_miss, cell.miss_score)
    };

    if is_awful(score) {
        return None;
    }

    while row > 0 || col > 0 {
        if last_action == Action::Match {
            indices_reverse.push(col - 1);
        }

        let cell = &dp[row][col];
        if last_action == Action::Match {
            last_action = cell.last_action_match;
            row -= 1;
            col -= 1;
        } else {
            last_action = cell.last_action_miss;
            col -= 1;
        }
    }

    indices_reverse.reverse();
    Some((score as i64, indices_reverse))
}

fn skip_penalty(_ch_idx: usize, _ch: char, last_action: Action) -> i64 {
    if last_action == Action::Match {
        // Non-consecutive match.
        return 2;
    } else {
        return 0;
    }
}

fn allow_match(pat_ch: char, line_ch: char, _last_action: Action) -> bool {
    pat_ch.to_ascii_lowercase() == line_ch.to_ascii_lowercase()
}

fn match_bonus(
    pat_idx: usize,
    pat_ch: char,
    pat_prev_ch: char,
    line_idx: usize,
    line_ch: char,
    line_prev_ch: char,
    last_action: Action,
) -> i64 {
    let mut score = 0;
    let pat_role = char_role(pat_prev_ch, pat_ch);
    let line_role = char_role(line_prev_ch, line_ch);

    // Bonus for case match
    if pat_ch == line_ch {
        score += 1;

        // Bonus for prefix match or case match when the pattern contains upper-case letters
        if pat_role == CharRole::Head || pat_idx == line_idx {
            score += 1;
        }
    }

    // For initial positions of pattern words
    if pat_role == CharRole::Head {
        // Bonus if it is matched to an initial position of some text word
        if line_role == CharRole::Head {
            score += 30;
        // Penalty for non-initial positions
        } else {
            score -= 10;
        }
    }

    if line_role == CharRole::Tail && pat_idx > 0 && last_action == Action::Match {
        score -= 30;
    }

    if pat_idx == 0 && line_role == CharRole::Tail {
        score -= 40;
    }

    score
}

#[derive(Debug)]
enum CharType {
    Empty,
    Lower,
    Upper,
    Separ,
}

#[inline]
fn char_type_of(ch: char) -> CharType {
    if ch == '\0' {
        CharType::Empty
    } else if ch == ' ' || ch == '_' || ch == '-' || ch == '/' || ch == '\\' {
        CharType::Separ
    } else if ch.is_ascii_uppercase() {
        CharType::Upper
    } else {
        CharType::Lower
    }
}

#[derive(Debug, PartialEq)]
enum CharRole {
    Tail,
    Head,
}

// checkout https://github.com/llvm-mirror/clang-tools-extra/blob/master/clangd/FuzzyMatch.cpp
// The Role can be determined from the Type of a character and its neighbors:
//
//   Example  | Chars | Type | Role
//   ---------+--------------+-----
//   F(o)oBar | Foo   | Ull  | Tail
//   Foo(B)ar | oBa   | lUl  | Head
//   (f)oo    | ^fo   | Ell  | Head
//   H(T)TP   | HTT   | UUU  | Tail
//
//      Curr= Empty Lower Upper Separ
// Prev=Empty 0x00, 0xaa, 0xaa, 0xff, // At start, Lower|Upper->Head
// Prev=Lower 0x00, 0x55, 0xaa, 0xff, // In word, Upper->Head;Lower->Tail
// Prev=Upper 0x00, 0x55, 0x59, 0xff, // Ditto, but U(U)U->Tail
// Prev=Separ 0x00, 0xaa, 0xaa, 0xff, // After separator, like at start
fn char_role(prev: char, cur: char) -> CharRole {
    use CharRole::*;
    use CharType::*;
    match (char_type_of(prev), char_type_of(cur)) {
        (Empty, Lower) | (Empty, Upper) | (Lower, Upper) | (Separ, Lower) | (Separ, Upper) => Head,
        _ => Tail,
    }
}

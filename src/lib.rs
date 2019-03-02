/// Implement the fuzzy match algorithm based on
/// https://github.com/llvm-mirror/clang-tools-extra/blob/master/clangd/FuzzyMatch.cpp
///
/// # Example:
/// ```edition2018
/// use fuzzy_matcher::{fuzzy_match, fuzzy_matcher};
///
/// assert_eq!(None, fuzzy_match("abc", "abx"));
/// assert!(fuzzy_match("axbycz", "abc").is_some());
/// assert!(fuzzy_match("axbycz", "xyz").is_some());
///
/// let (score, indices) = fuzzy_indices("axbycz", "abc").unwrap();
/// assert_eq!(indices, [0, 2, 4]);
///
/// ```
// Algorithm modified from
// https://github.com/llvm-mirror/clang-tools-extra/blob/master/clangd/FuzzyMatch.cpp
// Also: https://github.com/lewang/flx/issues/98
use std::cmp::max;

/// fuzzy match `line` with `pattern`, returning the score and indices of matches
pub fn fuzzy_indices(line: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
    if !cheap_matches(line, pattern) {
        return None;
    }

    let num_pattern_chars = pattern.chars().count();
    let num_line_chars = line.chars().count();

    let dp = build_graph(line, pattern, false);
    //print_dp(line, pattern, &dp);

    // search backwards for the matched indices
    let mut indices_reverse = Vec::new();
    let cell = dp[num_pattern_chars][num_line_chars];

    let (mut last_action, score) = if cell.match_score > cell.miss_score {
        (Action::Match, cell.match_score)
    } else {
        (Action::Miss, cell.miss_score)
    };

    let mut row = num_pattern_chars;
    let mut col = num_line_chars;

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

/// fuzzy match `line` with `pattern`, returning the score(the larger the better) on match
pub fn fuzzy_match(line: &str, pattern: &str) -> Option<i64> {
    if !cheap_matches(line, pattern) {
        return None;
    }

    let num_pattern_chars = pattern.chars().count();
    let num_line_chars = line.chars().count();

    let dp = build_graph(line, pattern, true);

    let cell = dp[num_pattern_chars & 1][num_line_chars];
    let score = max(cell.match_score, cell.miss_score);

    Some(score)
}

// checkout https://github.com/llvm-mirror/clang-tools-extra/blob/master/clangd/FuzzyMatch.cpp
// for the description
fn build_graph(line: &str, pattern: &str, compressed: bool) -> Vec<Vec<Score>> {
    let num_line_chars = line.chars().count();
    let num_pattern_chars = pattern.chars().count();
    let max_rows = if compressed { 2 } else { num_pattern_chars + 1 };

    let mut dp: Vec<Vec<Score>> = Vec::with_capacity(max_rows);

    for _ in 0..max_rows {
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
        let current_row_idx = if compressed {
            (pat_idx + 1) & 1
        } else {
            pat_idx + 1
        };
        let prev_row_idx = if compressed { pat_idx & 1 } else { pat_idx };

        let mut line_prev_ch = '\0';
        for (line_idx, line_ch) in line.chars().enumerate() {
            if line_idx < pat_idx {
                line_prev_ch = line_ch;
                continue;
            }

            // what if we skip current line character?
            // we need to calculate the cases where the pre line character is matched/missed
            let pre_miss = &dp[current_row_idx][line_idx];
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

            // what if we want to match current line character?
            // so we need to calculate the cases where the pre pattern character is matched/missed
            let pre_match = &dp[prev_row_idx][line_idx];
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

            dp[current_row_idx][line_idx + 1] = Score {
                miss_score,
                last_action_miss,
                match_score,
                last_action_match,
            };

            line_prev_ch = line_ch;
        }

        pat_prev_ch = pat_ch;
    }

    dp
}

fn cheap_matches(line: &str, pattern: &str) -> bool {
    let mut line_iter = line.chars().peekable();
    let mut pat_iter = pattern.chars().peekable();
    while line_iter.peek().is_some() && pat_iter.peek().is_some() {
        let line_lower = line_iter.peek().unwrap().to_ascii_lowercase();
        let pat_lower = pat_iter.peek().unwrap().to_ascii_lowercase();
        if line_lower == pat_lower {
            pat_iter.next();
        }
        line_iter.next();
    }

    !pat_iter.peek().is_some()
}

const AWFUL_SCORE: i64 = -(1 << 62);

#[derive(Debug, PartialEq, Clone, Copy)]
enum Action {
    Miss,
    Match,
}

#[derive(Debug, Clone, Copy)]
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
    let mut score = 1;
    let pat_role = char_role(pat_prev_ch, pat_ch);
    let line_role = char_role(line_prev_ch, line_ch);

    // Bonus: pattern so far is a (case-insensitive) prefix of the word.
    if pat_idx == line_idx {
        score += 1;
    }

    // Bonus: case matches, or a Head in the pattern aligns with one in the word.
    if (pat_ch == line_ch && (pat_ch.is_ascii_uppercase() || pat_idx == line_idx))
        || (pat_role == CharRole::Head && line_role == CharRole::Head)
    {
        score += 1;
    }

    // Penalty: matching inside a segment (and previous char wasn't matched).
    if line_role == CharRole::Tail && pat_idx > 0 && last_action == Action::Miss {
        score -= 3;
    }

    // Penalty: a Head in the pattern matches in the middle of a word segment.
    if pat_role == CharRole::Head && line_role == CharRole::Tail {
        score -= 1;
    }

    // Penalty: matching the first pattern character in the middle of a segment.
    if pat_idx == 0 && line_role == CharRole::Tail {
        score -= 4;
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

#[cfg(test)]
mod tests {
    use super::*;
    fn wrap_matches(line: &str, indices: &[usize]) -> String {
        let mut ret = String::new();
        let mut peekable = indices.iter().peekable();
        for (idx, ch) in line.chars().enumerate() {
            let next_id = **peekable.peek().unwrap_or(&&line.len());
            if next_id == idx {
                ret.push_str(format!("[{}]", ch).as_str());
                peekable.next();
            } else {
                ret.push(ch);
            }
        }

        ret
    }

    fn filter_and_sort(pattern: &str, lines: Vec<&'static str>) -> Vec<&'static str> {
        let scores: Vec<Option<i64>> = lines.iter().map(|k| fuzzy_match(k, pattern)).collect();
        println!("{:?}: {:?}, pattern: {}", scores, lines, pattern);

        let mut lines = lines;
        lines.sort_by_key(|k| -fuzzy_match(k, pattern).unwrap_or(-(1 << 10)));
        lines
    }

    fn wrap_fuzzy_match(line: &str, pattern: &str) -> Option<String> {
        let (_score, indices) = fuzzy_indices(line, pattern)?;
        Some(wrap_matches(line, &indices))
    }

    #[test]
    fn test_match_or_not() {
        assert_eq!(None, fuzzy_match("abcdefaghi", "中"));
        assert_eq!(None, fuzzy_match("abc", "abx"));
        assert!(fuzzy_match("axbycz", "abc").is_some());
        assert!(fuzzy_match("axbycz", "xyz").is_some());

        assert_eq!("[a]x[b]y[c]z", &wrap_fuzzy_match("axbycz", "abc").unwrap());
        assert_eq!("a[x]b[y]c[z]", &wrap_fuzzy_match("axbycz", "xyz").unwrap());
        assert_eq!(
            "[H]ello, [世]界",
            &wrap_fuzzy_match("Hello, 世界", "H世").unwrap()
        );
    }

    #[test]
    fn test_match_quality() {
        let choices1 = vec!["Monad", "mONAD", "monad"];
        let sorted1 = filter_and_sort("monad", choices1);
        assert_eq!(sorted1, ["monad", "Monad", "mONAD"]);
    }
}

#[allow(dead_code)]
fn print_dp(line: &str, pattern: &str, dp: &[Vec<Score>]) {
    let num_line_chars = line.chars().count();
    let num_pattern_chars = pattern.chars().count();

    print!("\t");
    for (idx, ch) in line.chars().enumerate() {
        print!("\t\t{}/{}", idx + 1, ch);
    }

    for row in 0..(num_pattern_chars + 1) {
        print!("\n{}\t", row);
        for col in 0..(num_line_chars + 1) {
            let cell = &dp[row][col];
            print!(
                "({},{})/({},{})\t",
                cell.miss_score,
                if cell.last_action_miss == Action::Miss {
                    'X'
                } else {
                    'O'
                },
                cell.match_score,
                if cell.last_action_match == Action::Miss {
                    'X'
                } else {
                    'O'
                }
            );
        }
    }
}

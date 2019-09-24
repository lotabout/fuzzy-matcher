///! The fuzzy matching algorithm used by skim
///! It focus more on path matching
///
///! # Example:
///! ```edition2018
///! use fuzzy_matcher::skim::{fuzzy_match, fuzzy_indices};
///!
///! assert_eq!(None, fuzzy_match("abc", "abx"));
///! assert!(fuzzy_match("axbycz", "abc").is_some());
///! assert!(fuzzy_match("axbycz", "xyz").is_some());
///!
///! let (score, indices) = fuzzy_indices("axbycz", "abc").unwrap();
///! assert_eq!(indices, [0, 2, 4]);
///!
///! ```
///!
///! It is modeled after <https://github.com/felipesere/icepick.git>

use std::cmp::max;
use crate::util::*;

const BONUS_MATCHED: i64 = 4;
const BONUS_CASE_MATCH: i64 = 4;
const BONUS_UPPER_MATCH: i64 = 6;
const BONUS_ADJACENCY: i64 = 10;
const BONUS_SEPARATOR: i64 = 8;
const BONUS_CAMEL: i64 = 8;
const PENALTY_CASE_UNMATCHED: i64 = -1;
const PENALTY_LEADING: i64 = -6; // penalty applied for every letter before the first match
const PENALTY_MAX_LEADING: i64 = -18; // maxing penalty for leading letters
const PENALTY_UNMATCHED: i64 = -2;

pub fn fuzzy_match(choice: &str, pattern: &str) -> Option<i64> {
    if pattern.is_empty() {
        return Some(0);
    }

    let scores = build_graph(choice, pattern)?;

    let last_row = &scores[scores.len() - 1];
    let (_, &MatchingStatus { final_score, .. }) = last_row
        .iter()
        .enumerate()
        .max_by_key(|&(_, x)| x.final_score)
        .expect("fuzzy_indices failed to iterate over last_row");
    Some(final_score)
}

pub fn fuzzy_indices(choice: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
    if pattern.is_empty() {
        return Some((0, Vec::new()));
    }

    let mut picked = vec![];
    let scores = build_graph(choice, pattern)?;

    let last_row = &scores[scores.len() - 1];
    let (mut next_col, &MatchingStatus { final_score, .. }) = last_row
        .iter()
        .enumerate()
        .max_by_key(|&(_, x)| x.final_score)
        .expect("fuzzy_indices failed to iterate over last_row");
    let mut pat_idx = scores.len() as i64 - 1;
    while pat_idx >= 0 {
        let status = scores[pat_idx as usize][next_col];
        next_col = status.back_ref;
        picked.push(status.idx);
        pat_idx -= 1;
    }
    picked.reverse();
    Some((final_score, picked))
}

#[derive(Clone, Copy, Debug)]
struct MatchingStatus {
    pub idx: usize,
    pub score: i64,
    pub final_score: i64,
    pub adj_num: usize,
    pub back_ref: usize,
}

impl Default for MatchingStatus {
    fn default() -> Self {
        MatchingStatus {
            idx: 0,
            score: 0,
            final_score: 0,
            adj_num: 1,
            back_ref: 0,
        }
    }
}

fn build_graph(choice: &str, pattern: &str) -> Option<Vec<Vec<MatchingStatus>>> {
    let mut scores = vec![];

    let mut match_start_idx = 0; // to ensure that the pushed char are able to match the pattern
    let mut pat_prev_ch = '\0';

    // initialize the match positions and inline scores
    for (pat_idx, pat_ch) in pattern.chars().enumerate() {
        let mut vec = vec![];
        let mut choice_prev_ch = '\0';
        for (idx, ch) in choice.chars().enumerate() {
            if ch.to_ascii_lowercase() == pat_ch.to_ascii_lowercase() && idx >= match_start_idx {
                let score = fuzzy_score(ch, idx, choice_prev_ch, pat_ch, pat_idx, pat_prev_ch);
                vec.push(MatchingStatus {
                    idx,
                    score,
                    final_score: score,
                    adj_num: 1,
                    back_ref: 0,
                });
            }
            choice_prev_ch = ch;
        }

        if vec.is_empty() {
            // not matched
            return None;
        }
        match_start_idx = vec[0].idx + 1;
        scores.push(vec);
        pat_prev_ch = pat_ch;
    }

    // calculate max scores considering adjacent characters
    for pat_idx in 1..scores.len() {
        let (first_half, last_half) = scores.split_at_mut(pat_idx);

        let prev_row = &first_half[first_half.len() - 1];
        let cur_row = &mut last_half[0];

        for idx in 0..cur_row.len() {
            let next = cur_row[idx];
            let prev = if idx > 0 {
                cur_row[idx - 1]
            } else {
                MatchingStatus::default()
            };

            let mut score_before_idx = prev.final_score - prev.score + next.score;
            score_before_idx += PENALTY_UNMATCHED * ((next.idx - prev.idx) as i64);
            score_before_idx -= if prev.adj_num == 0 {
                BONUS_ADJACENCY
            } else {
                0
            };

            let (back_ref, score, adj_num) = prev_row
                .iter()
                .enumerate()
                .take_while(|&(_, &MatchingStatus { idx, .. })| idx < next.idx)
                .skip_while(|&(_, &MatchingStatus { idx, .. })| idx < prev.idx)
                .map(|(back_ref, cur)| {
                    let adj_num = next.idx - cur.idx - 1;
                    let mut final_score = cur.final_score + next.score;
                    final_score += if adj_num == 0 {
                        BONUS_ADJACENCY
                    } else {
                        PENALTY_UNMATCHED * adj_num as i64
                    };
                    (back_ref, final_score, adj_num)
                })
                .max_by_key(|&(_, x, _)| x)
                .unwrap_or((prev.back_ref, score_before_idx, prev.adj_num));

            cur_row[idx] = if idx > 0 && score < score_before_idx {
                MatchingStatus {
                    final_score: score_before_idx,
                    back_ref: prev.back_ref,
                    adj_num,
                    ..next
                }
            } else {
                MatchingStatus {
                    final_score: score,
                    back_ref,
                    adj_num,
                    ..next
                }
            };
        }
    }

    Some(scores)
}

// judge how many scores the current index should get
fn fuzzy_score(
    choice_ch: char,
    choice_idx: usize,
    choice_prev_ch: char,
    pat_ch: char,
    pat_idx: usize,
    _pat_prev_ch: char,
) -> i64 {
    let mut score = BONUS_MATCHED;

    let choice_prev_ch_type = char_type_of(choice_prev_ch);
    let choice_role = char_role(choice_prev_ch, choice_ch);

    if pat_ch == choice_ch {
        if pat_ch.is_uppercase() {
            score += BONUS_UPPER_MATCH;
        } else {
            score += BONUS_CASE_MATCH;
        }
    } else {
        score += PENALTY_CASE_UNMATCHED;
    }

    // apply bonus for camelCases
    if choice_role == CharRole::Head {
        score += BONUS_CAMEL;
    }

    // apply bonus for matches after a separator
    if choice_prev_ch_type == CharType::Separ {
        score += BONUS_SEPARATOR;
    }

    if pat_idx == 0 {
        score += max((choice_idx as i64) * PENALTY_LEADING, PENALTY_MAX_LEADING);
    }

    score
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

    fn filter_and_sort(pattern: &str, lines: &[&'static str]) -> Vec<&'static str> {
        let mut lines_with_score: Vec<(i64, &'static str)> = lines
            .into_iter()
            .map(|&s| (fuzzy_match(s, pattern).unwrap_or(-(1 << 62)), s))
            .collect();
        lines_with_score.sort_by_key(|(score, _)| -score);
        lines_with_score
            .into_iter()
            .map(|(_, string)| string)
            .collect()
    }

    fn wrap_fuzzy_match(line: &str, pattern: &str) -> Option<String> {
        let (_score, indices) = fuzzy_indices(line, pattern)?;
        Some(wrap_matches(line, &indices))
    }

    fn assert_order(pattern: &str, choices: &[&'static str]) {
        let result = filter_and_sort(pattern, choices);

        if result != choices {
            // debug print
            println!("pattern: {}", pattern);
            for &choice in choices.iter() {
                if let Some((score, indices)) = fuzzy_indices(choice, pattern) {
                    println!("{}: {:?}", score, wrap_matches(choice, &indices));
                } else {
                    println!("NO MATCH for {}", choice);
                }
            }
        }

        assert_eq!(result, choices);
    }

    #[test]
    fn test_match_or_not() {
        assert_eq!(Some(0), fuzzy_match("", ""));
        assert_eq!(Some(0), fuzzy_match("abcdefaghi", ""));
        assert_eq!(None, fuzzy_match("", "a"));
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
        // case
        // assert_order("monad", &["monad", "Monad", "mONAD"]);

        // initials
        assert_order("ab", &["ab", "aoo_boo", "acb"]);
        assert_order("CC", &["CamelCase", "camelCase", "camelcase"]);
        assert_order("cC", &["camelCase", "CamelCase", "camelcase"]);
        assert_order(
            "cc",
            &[
                "camel case",
                "camelCase",
                "camelcase",
                "CamelCase",
                "camel ace",
            ],
        );
        assert_order(
            "Da.Te",
            &["Data.Text", "Data.Text.Lazy", "Data.Aeson.Encoding.text"],
        );
        // prefix
        assert_order("is", &["isIEEE", "inSuf"]);
        // shorter
        assert_order("ma", &["map", "many", "maximum"]);
        assert_order("print", &["printf", "sprintf"]);
        // score(PRINT) = kMinScore
        assert_order("ast", &["ast", "AST", "INT_FAST16_MAX"]);
        // score(PRINT) > kMinScore
        assert_order("Int", &["int", "INT", "PRINT"]);
        assert_order("skim", &["Code/External/skim", "Code/External/skim/man"]);
    }
}

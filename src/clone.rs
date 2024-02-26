use std::i64::MAX;
use std::thread::sleep;
use std::time::Duration;

use crate::util::char_equal;
use crate::FuzzyMatcher;
use crate::IndexType;
use crate::ScoreType;

impl FuzzyMatcher for CloneMatcher {
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(ScoreType, Vec<IndexType>)> {
        self.fuzzy(choice, pattern)
    }

    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<ScoreType> {
        self.fuzzy(choice, pattern).map(|(score, _)| score)
    }
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
enum CaseMatching {
    Respect,
    Ignore,
    Smart,
}

pub struct CloneMatcher {
    case: CaseMatching,
}

impl Default for CloneMatcher {
    fn default() -> Self {
        CloneMatcher {
            case: CaseMatching::Smart,
        }
    }
}

impl CloneMatcher {
    fn fuzzy(&self, choice: &str, pattern: &str) -> Option<(ScoreType, Vec<IndexType>)> {
        let case_sensitive = true;

        let matches = forward_matches(choice, pattern, case_sensitive)?;

        let mut start_idx = *matches.first()?;
        let end_idx = *matches.last()?;

        start_idx = reverse_matches(choice, pattern, case_sensitive, start_idx, end_idx)?;

        if choice.len() == 0 {
            return None;
        }

        let max = usize::MAX;

        let base = max.div_euclid(choice.len());

        let abs_diff = end_idx.abs_diff(start_idx) + 1;

        if abs_diff == 0 {
            return None;
        }

        let close_to_beginning = choice.len() - start_idx;

        let divisor = abs_diff.saturating_mul(close_to_beginning);

        if divisor == 0 {
            return None;
        }

        let score: ScoreType = base.div_euclid(divisor.checked_pow(3)?) as i64;

        Some((score, matches))
    }
}

pub fn forward_matches(choice: &str, pattern: &str, case_sensitive: bool) -> Option<Vec<usize>> {
    let mut start_idx = 0usize;

    let mut pattern_indices: Vec<usize> = Vec::new();

    for p_char in pattern.chars() {
        let byte_idx = choice
            .char_indices()
            .skip(start_idx)
            .find_map(|(idx, c_char)| {
                if char_equal(p_char, c_char, case_sensitive) {
                    start_idx = idx;
                    return Some(idx);
                }

                None
            })?;
        pattern_indices.push(byte_idx);
    }

    if pattern_indices.len() == pattern.len() {
        return Some(pattern_indices);
    }

    None
}

pub fn reverse_matches(
    choice: &str,
    pattern: &str,
    case_sensitive: bool,
    start_idx: usize,
    end_idx: usize,
) -> Option<usize> {
    let mut end_idx = choice.len() - end_idx;

    let mut pattern_indices: Vec<usize> = Vec::new();

    for p_char in pattern.chars().rev() {
        let byte_idx = choice
            .char_indices()
            .rev()
            .skip(end_idx)
            .find_map(|(idx, c_char)| {
                if char_equal(p_char, c_char, case_sensitive) {
                    end_idx = idx;
                    return Some(idx);
                }

                None
            })?;
        pattern_indices.push(byte_idx);
    }

    let last = pattern_indices.last()?;

    let last_as_first = choice.len() - last;

    if last_as_first > start_idx {
        return Some(last_as_first);
    }

    None
}

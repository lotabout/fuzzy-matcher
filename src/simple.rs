use crate::util::char_equal;
use crate::FuzzyMatcher;
use crate::IndexType;
use crate::ScoreType;

const BASELINE: i64 = 100_000;

impl FuzzyMatcher for SimpleMatcher {
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

pub struct SimpleMatcher {
    case: CaseMatching,
}

impl Default for SimpleMatcher {
    fn default() -> Self {
        SimpleMatcher {
            case: CaseMatching::Smart,
        }
    }
}

impl SimpleMatcher {
    pub fn ignore_case(mut self) -> Self {
        self.case = CaseMatching::Ignore;
        self
    }

    pub fn smart_case(mut self) -> Self {
        self.case = CaseMatching::Smart;
        self
    }

    pub fn respect_case(mut self) -> Self {
        self.case = CaseMatching::Respect;
        self
    }

    fn fuzzy(&self, choice: &str, pattern: &str) -> Option<(ScoreType, Vec<IndexType>)> {
        let case_sensitive = true;

        if choice.len() == 0 {
            return None;
        }

        if pattern.len() > choice.len() {
            return None;
        }

        if pattern.len() == 0 {
            return Some((0, Vec::new()));
        }

        let forward_matches = Self::forward_matches(choice, pattern, case_sensitive)?;

        let mut start_idx = *forward_matches.first()?;
        let end_idx = *forward_matches.last()?;

        Self::reverse_matches(choice, pattern, case_sensitive, &mut start_idx, end_idx);

        let score = Self::score(start_idx, end_idx, pattern, choice);

        if score >= BASELINE {
            return Some((score, forward_matches));
        }

        None
    }

    pub fn score(start_idx: usize, end_idx: usize, pattern: &str, choice: &str) -> i64 {
        let idx_abs_diff = end_idx.abs_diff(start_idx);
        let choice_len = choice.chars().count();
        let pattern_len = pattern.chars().count();
        let choice_abs_diff = choice_len.abs_diff(pattern_len);

        let group_closeness = if idx_abs_diff == 0 {
            10_000_000
        } else if idx_abs_diff >= 4 {
            0
        } else {
            1_000_000 / idx_abs_diff
        };

        let first_letter_bonus = if start_idx == 0 {
            100_000
        } else if idx_abs_diff <= 4 {
            10_000 / start_idx
        } else {
            0
        };

        let total_choice_diff_len = if choice_abs_diff == 0 {
            100_000
        } else {
            10_000 / choice_abs_diff
        };

        let score: ScoreType =
            (group_closeness + first_letter_bonus + total_choice_diff_len) as i64;

        score
    }

    pub fn forward_matches(
        choice: &str,
        pattern: &str,
        case_sensitive: bool,
    ) -> Option<Vec<usize>> {
        let mut skip = 0usize;

        let mut pattern_indices: Vec<usize> = Vec::new();

        for p_char in pattern.chars() {
            let byte_idx = choice.char_indices().skip(skip).find_map(|(idx, c_char)| {
                if char_equal(p_char, c_char, case_sensitive) {
                    skip = idx;
                    return Some(idx);
                }

                None
            })?;
            pattern_indices.push(byte_idx);
        }

        if pattern_indices.len() == pattern.chars().count() {
            return Some(pattern_indices);
        }

        None
    }

    pub fn reverse_matches(
        choice: &str,
        pattern: &str,
        case_sensitive: bool,
        start_idx: &mut usize,
        end_idx: usize,
    ) {
        let mut skip = 0usize;
        let idx_abs_diff = end_idx.abs_diff(*start_idx);

        let mut pattern_indices: Vec<usize> = Vec::new();

        for p_char in pattern.chars().rev() {
            let Some(byte_idx) =
                choice
                    .char_indices()
                    .rev()
                    .skip(skip)
                    .find_map(|(idx, c_char)| {
                        if char_equal(p_char, c_char, case_sensitive) {
                            skip = idx;
                            return Some(idx);
                        }

                        None
                    })
            else {
                return;
            };
            pattern_indices.push(byte_idx);
        }

        if idx_abs_diff > pattern_indices.len() {
            pattern_indices.reverse();

            let Some(first) = pattern_indices.first() else {
                return;
            };

            if first > start_idx {
                *start_idx = *first;
            }
        }
    }
}

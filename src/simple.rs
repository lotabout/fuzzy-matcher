use crate::util::char_equal;
use crate::FuzzyMatcher;
use crate::IndexType;
use crate::ScoreType;

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

        if pattern.len() == 0 {
            return Some((0, Vec::new()));
        }

        let matches = Self::forward_matches(choice, pattern, case_sensitive)?;

        let mut start_idx = *matches.first()?;
        let end_idx = *matches.last()?;

        Self::reverse_matches(choice, pattern, case_sensitive, &mut start_idx);

        let score = Self::score(start_idx, end_idx, pattern, choice);

        Some((score, matches))
    }

    pub fn score(start_idx: usize, end_idx: usize, pattern: &str, choice: &str) -> i64 {
        let abs_diff = end_idx.abs_diff(start_idx);

        let close_to_beginning = (choice.len() - start_idx) * 20;

        let first_letter_bonus = if start_idx == 0 { 2000 } else { 0 };

        let pattern_match_size = pattern.len() - abs_diff;

        let group_closeness = if pattern_match_size == 0 {
            100_000
        } else {
            100_000 / pattern_match_size
        };

        let choice_size = choice.len() * 40;

        let score: ScoreType =
            (group_closeness + close_to_beginning + first_letter_bonus - choice_size) as i64;

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
            let byte_idx = choice[skip..]
                .char_indices()
                .skip(skip)
                .find_map(|(idx, c_char)| {
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
    ) {
        let mut skip = 0usize;

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

        let Some(last) = pattern_indices.last() else {
            return;
        };

        let last_as_first = choice.chars().count() - last;

        if last_as_first > *start_idx {
            *start_idx = last_as_first;
        }
    }
}

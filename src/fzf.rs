///! The fuzzy matching algorithm used by fzf(almost the same)
///
///! # Example:
///! ```edition2018
///! use fuzzy_matcher::fzf::FzfMatcherV1;
///!
///! let matcher = FzfMatcherV1::default();
///! assert_eq!(None, matcher.fuzzy_match("abc", "abx"));
///! assert!(matcher.fuzzy_match("axbycz", "abc").is_some());
///! assert!(matcher.fuzzy_match("axbycz", "xyz").is_some());
///!
///! let (score, indices) = matcher.fuzzy_indices("axbycz", "abc").unwrap();
///! assert_eq!(indices, [0, 2, 4]);
///! ```
use crate::util::{char_type_of, CharType};
use crate::FuzzyMatcher;
use std::cmp::max;

pub trait FzfScoreConfig {
    fn score_match(&self) -> i64;
    fn score_gap_start(&self) -> i64;
    fn score_gap_extension(&self) -> i64;

    /// We prefer matches at the beginning of a word, but the bonus should not be
    /// too great to prevent the longer acronym matches from always winning over
    /// shorter fuzzy matches. The bonus point here was specifically chosen that
    /// the bonus is cancelled when the gap between the acronyms grows over
    /// 8 characters, which is approximately the average length of the words found
    /// in web2 dictionary and my file system.
    fn bonus_boundary(&self) -> i64 {
        self.score_match() / 2
    }

    /// Although bonus point for non-word characters is non-contextual, we need it
    /// for computing bonus points for consecutive chunks starting with a non-word
    /// character.
    fn bonus_non_word(&self) -> i64 {
        self.score_match() / 2
    }

    /// Edge-triggered bonus for matches in camelCase words.
    /// Compared to word-boundary case, they don't accompany single-character gaps
    /// (e.g. FooBar vs. foo-bar), so we deduct bonus point accordingly.
    fn bonus_camel123(&self) -> i64 {
        self.bonus_boundary() + self.score_gap_extension()
    }

    /// Minimum bonus point given to characters in consecutive chunks.
    /// Note that bonus points for consecutive matches shouldn't have needed if we
    /// used fixed match score as in the original algorithm.
    fn bonus_consecutive(&self) -> i64 {
        -(self.score_gap_start() + self.score_gap_extension())
    }

    /// The first character in the typed pattern usually has more significance
    /// than the rest so it's important that it appears at special positions where
    /// bonus points are given. e.g. "to-go" vs. "ongoing" on "og" or on "ogo".
    /// The amount of the extra bonus should be limited so that the gap penalty is
    /// still respected.
    fn bonus_first_char_multiplier(&self) -> i64;
}

#[derive(Default)]
pub struct DefaultFzfConfig {}

impl FzfScoreConfig for DefaultFzfConfig {
    fn score_match(&self) -> i64 {
        16
    }

    fn score_gap_start(&self) -> i64 {
        -3
    }

    fn score_gap_extension(&self) -> i64 {
        -1
    }

    fn bonus_first_char_multiplier(&self) -> i64 {
        2
    }
}

pub struct FzfMatcherV1 {
    score_config: Box<dyn FzfScoreConfig>,
    case_sensitive: bool,
}

impl Default for FzfMatcherV1 {
    fn default() -> Self {
        Self::new(Box::new(DefaultFzfConfig::default()), false)
    }
}

impl FzfMatcherV1 {
    pub fn new(score_config: Box<dyn FzfScoreConfig>, case_sensitive: bool) -> Self {
        Self {
            score_config,
            case_sensitive,
        }
    }

    /// fuzzy_match_v1 finds the first "fuzzy" occurrence of the pattern within the given
    /// text in O(n) time where n is the length of the text. Once the position of the
    /// last character is located, it traverses backwards to see if there's a shorter
    /// substring that matches the pattern.
    ///
    /// ```text
    /// a_____b___abc__  To find "abc"
    /// *-----*-----*>   1. Forward scan
    ///          <***    2. Backward scan
    /// ```
    ///
    /// It is modeled after fzf's v1 algorithm. Removed some complicated things.
    fn fuzzy_match_v1(
        &self,
        choice: &str,
        pattern: &str,
        case_sensitive: bool,
    ) -> Option<(i64, Vec<usize>)> {
        let _ = ascii_fuzzy_first_index(choice, pattern, case_sensitive)?;
        let choice_iter = choice.char_indices();
        let mut pattern_iter = pattern.chars().peekable();

        let mut o_start_idx = None;
        let mut o_end_idx = None;

        // scan forward to find the first match of whole pattern
        for (c_idx, c) in choice_iter {
            match pattern_iter.peek() {
                Some(&p) => {
                    if char_equal(c, p, case_sensitive) {
                        let _ = pattern_iter.next();
                        o_start_idx = o_start_idx.or(Some(c_idx));
                        o_end_idx = Some(c_idx)
                    }
                }
                None => break,
            }
        }

        if o_start_idx.is_none() || o_end_idx.is_none() {
            return None;
        }

        // scan backward to find the first match of whole pattern
        if o_start_idx < o_end_idx {
            let end_idx = o_end_idx.unwrap();
            let choice_iter = choice
                .char_indices()
                .rev()
                .skip_while(|&(idx, _)| idx > end_idx);
            let mut pattern_iter = pattern.chars().rev().peekable();

            for (c_idx, c) in choice_iter {
                match pattern_iter.peek() {
                    Some(&p) => {
                        if char_equal(c, p, case_sensitive) {
                            let _ = pattern_iter.next();
                            o_end_idx = o_end_idx.or(Some(c_idx));
                            o_start_idx = Some(c_idx)
                        }
                    }
                    None => break,
                }
            }
        }

        Some(self.calculate_score_with_pos(
            choice,
            pattern,
            o_start_idx.unwrap(),
            o_end_idx.unwrap(),
            case_sensitive,
        ))
    }

    fn calculate_score_with_pos(
        &self,
        choice: &str,
        pattern: &str,
        start_idx: usize,
        _end_idx: usize,
        case_sensitive: bool,
    ) -> (i64, Vec<usize>) {
        let mut pos = Vec::new();

        let n_skip = max(start_idx, 1) - 1;
        let mut choice_iter = choice.char_indices().skip(n_skip);
        let mut pattern_iter = pattern.char_indices().peekable();

        let mut prev_char_type = if start_idx > 0 {
            let (_, prev_char) = choice_iter.next().unwrap();
            char_type_of(prev_char)
        } else {
            CharType::Empty
        };

        let mut score = 0;
        let mut in_gap = false;
        let mut consecutive = 0;
        let mut first_bonus = 0;

        for (c_idx, c) in choice_iter {
            let char_type = char_type_of(c);
            let op = pattern_iter.peek();
            if op.is_none() {
                break;
            }

            let (p_idx, p) = *op.unwrap();
            if char_equal(c, p, case_sensitive) {
                pos.push(c_idx);

                score += self.score_config.score_match();
                let mut bonus = self.bonus_for(&prev_char_type, &char_type);
                if consecutive == 0 || bonus == self.score_config.bonus_boundary() {
                    first_bonus = bonus;
                }

                if bonus == self.score_config.bonus_boundary() {
                    bonus = max(
                        max(bonus, first_bonus),
                        self.score_config.bonus_consecutive(),
                    );
                }

                if p_idx == 0 {
                    score += bonus * self.score_config.bonus_first_char_multiplier()
                } else {
                    score += bonus;
                }

                in_gap = false;
                consecutive += 1;
                let _ = pattern_iter.next();
            } else {
                if in_gap {
                    score += self.score_config.score_gap_extension();
                } else {
                    score += self.score_config.score_gap_start();
                }
                in_gap = true;
                consecutive = 0;
                first_bonus = 0;
            }

            prev_char_type = char_type;
        }

        (score, pos)
    }

    fn bonus_for(&self, prev_char_type: &CharType, char_type: &CharType) -> i64 {
        match (prev_char_type, char_type) {
            (CharType::NonWord, t) if *t != CharType::NonWord => self.score_config.bonus_boundary(),
            (CharType::Lower, CharType::Upper) => self.score_config.bonus_camel123(),
            (t, CharType::Number) if *t != CharType::Number => self.score_config.bonus_camel123(),
            (_, CharType::NonWord) => self.score_config.bonus_non_word(),
            _ => 0,
        }
    }
}

impl FuzzyMatcher for FzfMatcherV1 {
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
        self.fuzzy_match_v1(choice, pattern, self.case_sensitive)
    }
}

/// try to match the pattern as ASCII, and return the first matching index
/// If the choice is not all ASCII, return Some(0) indicating should try start full match from 0.
/// If the choice is all ASCII and patter is not, return None indicating not matching.
///
/// e.g. ("abcd", "BD", false) => Some(1)
fn ascii_fuzzy_first_index(choice: &str, pattern: &str, case_sensitive: bool) -> Option<usize> {
    if !choice.is_ascii() {
        return Some(0);
    } else if !pattern.is_ascii() {
        return None;
    }

    let choice_iter = choice.bytes().enumerate();
    let mut pattern_iter = pattern.bytes().peekable();

    let mut ret = None;
    for (choice_idx, c) in choice_iter {
        match pattern_iter.peek() {
            Some(&p) => {
                if byte_equal(c, p, case_sensitive) {
                    let _ = pattern_iter.next();
                    ret = ret.or(Some(choice_idx));
                }
            }
            None => break,
        }
    }

    ret
}

/// Given 2 ascii character, check if they are equal (considering case)
/// e.g. ('a', 'A', true) => false
/// e.g. ('a', 'A', false) => true
fn byte_equal(a: u8, b: u8, case_sensitive: bool) -> bool {
    if a == b {
        true
    } else if !case_sensitive {
        let ai = a | 0x20;
        let bi = b | 0x20;
        ai >= b'a' && ai <= b'z' && ai == bi
    } else {
        false
    }
}

/// Given 2 character, check if they are equal (considering ascii case)
/// e.g. ('a', 'A', true) => false
/// e.g. ('a', 'A', false) => true
fn char_equal(a: char, b: char, case_sensitive: bool) -> bool {
    if case_sensitive {
        a == b
    } else {
        a.eq_ignore_ascii_case(&b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::assert_order;

    #[test]
    fn test_ascii_fuzzy_first_index() {
        // not matched ascii
        assert_eq!(None, ascii_fuzzy_first_index("A", "B", true));
        assert_eq!(None, ascii_fuzzy_first_index("A", "B", false));

        // matched non-ascii
        assert_eq!(Some(0), ascii_fuzzy_first_index("中", "中", true));
        assert_eq!(Some(0), ascii_fuzzy_first_index("中", "中", false));

        // matched ascii(same case)
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd", "bd", true));
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd", "bd", false));

        // matched ascii(different case)
        assert_eq!(None, ascii_fuzzy_first_index("abcd", "BD", true));
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd", "BD", false));

        // matched ascii(contains non-letter)
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd135", "bd3", true));
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd135", "bd3", false));

        // matched ascii(contains non-letter)
        assert_eq!(None, ascii_fuzzy_first_index("abcd135", "BD3", true));
        assert_eq!(Some(1), ascii_fuzzy_first_index("abcd135", "BD3", false));
    }

    #[test]
    fn test_fuzzy_matcher_v1() {
        let matcher = FzfMatcherV1::default();
        assert_eq!(None, matcher.fuzzy_indices("a", "b"));
        assert_eq!(vec![0], matcher.fuzzy_indices("a", "a").unwrap().1);
        assert_eq!(vec![0], matcher.fuzzy_indices("a", "A").unwrap().1);

        // should match the first ab
        assert_eq!(
            vec![0, 6],
            matcher.fuzzy_indices("a_____b___abc__", "ab").unwrap().1
        );

        // should match the last ab
        assert_eq!(
            vec![10, 11, 12],
            matcher.fuzzy_indices("a_____b___abc__", "abc").unwrap().1
        );
    }

    #[test]
    fn test_quality_of_v1() {
        let matcher = FzfMatcherV1::default();

        // case
        // assert_order("monad", &["monad", "Monad", "mONAD"]);

        // initials
        assert_order(&matcher, "ab", &["aoo_boo", "ab", "acb"]);
        assert_order(&matcher, "CC", &["CamelCase", "camelCase", "camelcase"]);
        assert_order(&matcher, "cC", &["camelCase", "CamelCase", "camelcase"]);
        assert_order(
            &matcher,
            "cc",
            &[
                "camel case",
                "camelCase",
                "CamelCase",
                "camelcase",
                "camel ace",
            ],
        );
        assert_order(
            &matcher,
            "Da.Te",
            &["Data.Text", "Data.Text.Lazy", "Data.Aeson.Encoding.text"],
        );
        // shorter
        assert_order(&matcher, "ma", &["map", "many", "maximum"]);
        assert_order(&matcher, "print", &["printf", "sprintf"]);
        // score(PRINT) = kMinScore
        assert_order(&matcher, "ast", &["ast", "AST", "INT_FAST16_MAX"]);
        // score(PRINT) > kMinScore
        assert_order(&matcher, "Int", &["int", "INT", "PRINT"]);
    }
}

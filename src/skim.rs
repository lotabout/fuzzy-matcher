use crate::skim::Movement::{Match, Skip};
use crate::util::*;
///! The fuzzy matching algorithm used by skim
///!
///! # Example:
///! ```edition2018
///! use fuzzy_matcher::FuzzyMatcher;
///! use fuzzy_matcher::skim::SkimMatcherV2;
///!
///! let matcher = SkimMatcherV2::default();
///! assert_eq!(None, matcher.fuzzy_match("abc", "abx"));
///! assert!(matcher.fuzzy_match("axbycz", "abc").is_some());
///! assert!(matcher.fuzzy_match("axbycz", "xyz").is_some());
///!
///! let (score, indices) = matcher.fuzzy_indices("axbycz", "abc").unwrap();
///! assert_eq!(indices, [0, 2, 4]);
///! ```
use crate::FuzzyMatcher;
use std::cmp::max;
use std::ptr;

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

pub struct SkimMatcher {}

impl Default for SkimMatcher {
    fn default() -> Self {
        Self {}
    }
}

impl FuzzyMatcher for SkimMatcher {
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
        fuzzy_indices(choice, pattern)
    }

    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<i64> {
        fuzzy_match(choice, pattern)
    }
}

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
    if choice_prev_ch_type == CharType::NonWord {
        score += BONUS_SEPARATOR;
    }

    if pat_idx == 0 {
        score += max((choice_idx as i64) * PENALTY_LEADING, PENALTY_MAX_LEADING);
    }

    score
}

pub trait SkimScoreConfig: Send + Sync {
    fn score_match(&self) -> i32;
    fn gap_start(&self) -> i32;
    fn gap_extension(&self) -> i32;

    /// The first character in the typed pattern usually has more significance
    /// than the rest so it's important that it appears at special positions where
    /// bonus points are given. e.g. "to-go" vs. "ongoing" on "og" or on "ogo".
    /// The amount of the extra bonus should be limited so that the gap penalty is
    /// still respected.
    fn bonus_first_char_multiplier(&self) -> i32;

    /// We prefer matches at the beginning of a word, but the bonus should not be
    /// too great to prevent the longer acronym matches from always winning over
    /// shorter fuzzy matches. The bonus point here was specifically chosen that
    /// the bonus is cancelled when the gap between the acronyms grows over
    /// 8 characters, which is approximately the average length of the words found
    /// in web2 dictionary and my file system.
    fn bonus_boundary(&self) -> i32 {
        self.score_match() / 2
    }

    /// Although bonus point for non-word characters is non-contextual, we need it
    /// for computing bonus points for consecutive chunks starting with a non-word
    /// character.
    fn bonus_non_word(&self) -> i32 {
        self.score_match() / 2
    }

    /// Edge-triggered bonus for matches in camelCase words.
    /// Compared to word-boundary case, they don't accompany single-character gaps
    /// (e.g. FooBar vs. foo-bar), so we deduct bonus point accordingly.
    fn bonus_camel123(&self) -> i32 {
        self.bonus_boundary() + self.gap_extension()
    }

    /// Minimum bonus point given to characters in consecutive chunks.
    /// Note that bonus points for consecutive matches shouldn't have needed if we
    /// used fixed match score as in the original algorithm.
    fn bonus_consecutive(&self) -> i32 {
        -(self.gap_start() + self.gap_extension())
    }

    /// Skim will match case-sensitively if the pattern contains ASCII upper case,
    /// If case of case insensitive match, the penalty will be given to case mismatch
    fn penalty_case_mismatch(&self) -> i32 {
        self.gap_extension() * 2
    }
}

#[derive(Default, Copy, Clone)]
pub struct DefaultSkimScoreConfig {}

impl SkimScoreConfig for DefaultSkimScoreConfig {
    fn score_match(&self) -> i32 {
        16
    }

    fn gap_start(&self) -> i32 {
        -3
    }

    fn gap_extension(&self) -> i32 {
        -1
    }

    fn bonus_first_char_multiplier(&self) -> i32 {
        2
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Movement {
    Match,
    Skip,
}

/// Inner state of the score matrix
#[derive(Debug, Copy, Clone)]
struct MatrixCell {
    pub movement: Movement,
    pub score: i32, // The max score of align pattern[..i] & choice[..j]
}

const MATRIX_CELL_NEG_INFINITY: i32 = std::i16::MIN as i32;

impl Default for MatrixCell {
    fn default() -> Self {
        Self {
            movement: Skip,
            score: MATRIX_CELL_NEG_INFINITY,
        }
    }
}

use std::cell::RefCell;
use thread_local::CachedThreadLocal;

/// Fuzzy matching is a sub problem is sequence alignment.
/// Specifically what we'd like to implement is sequence alignment with affine gap penalty.
/// Ref: https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/gaps.pdf
///
/// Given `pattern`(i) and `choice`(j), we'll maintain 2 score matrix:
///
/// ```text
/// M[i][j] = match(i, j) + max(M[i-1][j-1] + consecutive, P[i-1][j-1])
/// M[i][j] = -infinity if p[i][j] do not match
///
/// M[i][j] means the score of best alignment of p[..=i] and c[..=j] ending with match/mismatch e.g.:
///
/// c: [.........]b
/// p: [.........]b
///
/// So that p[..=i-1] and c[..=j-1] could be any alignment
///
/// P[i][j] = max(M[i][j-k]-gap(k)) for k in 1..j
///
/// P[i][j] means the score of best alignment of p[..=i] and c[..=j] where c[j] is not matched.
/// So that we need to search through all the previous matches, and calculate the gap.
///
///   (j-k)--.   j
/// c: [....]bcdef
/// p: [....]b----
///          i
/// ```
///
/// Note that the above is O(n^3) in the worst case. However the above algorithm uses a general gap
/// penalty, but we use affine gap: `gap = gap_start + k * gap_extend` where:
/// - u: the cost of starting of gap
/// - v: the cost of extending a gap by one more space.
///
/// So that we could optimize the algorithm by:
///
/// ```text
/// P[i][j] = max(gap_start + gap_extend + M[i][j-1], gap_extend + P[i][j-1])
/// ```
///
/// In summary:
///
/// ```text
/// M[i][j] = match(i, j) + max(M[i-1][j-1] + consecutive, P[i-1][j-1])
/// M[i][j] = -infinity if p[i] and c[j] do not match
/// P[i][j] = max(gap_start + gap_extend + M[i][j-1], gap_extend + P[i][j-1])
/// ```
pub struct SkimMatcherV2 {
    score_config: Box<dyn SkimScoreConfig>,
    element_limit: usize,

    m_cache: CachedThreadLocal<RefCell<Vec<MatrixCell>>>,
    p_cache: CachedThreadLocal<RefCell<Vec<MatrixCell>>>,
}

impl Default for SkimMatcherV2 {
    fn default() -> Self {
        Self {
            score_config: Box::new(DefaultSkimScoreConfig::default()),
            m_cache: CachedThreadLocal::new(),
            p_cache: CachedThreadLocal::new(),
            element_limit: 0,
        }
    }
}

/// Simulate a 1-D vector as 2-D matrix
struct ScoreMatrix<'a> {
    matrix: &'a mut Vec<MatrixCell>,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> ScoreMatrix<'a> {
    /// given a matrix, extend it to be (rows x cols) and fill in as init_val
    pub fn new(matrix: &'a mut Vec<MatrixCell>, rows: usize, cols: usize) -> Self {
        matrix.resize(rows * cols, MatrixCell::default());
        ScoreMatrix { matrix, rows, cols }
    }

    #[inline]
    fn get_score(&self, row: usize, col: usize) -> i32 {
        self.matrix[row * self.cols + col].score
    }

    #[inline]
    fn get_movement(&self, row: usize, col: usize) -> Movement {
        self.matrix[row * self.cols + col].movement
    }

    #[inline]
    fn set_score(&mut self, row: usize, col: usize, score: i32) {
        self.matrix[row * self.cols + col].score = score;
    }

    #[inline]
    fn set_movement(&mut self, row: usize, col: usize, movement: Movement) {
        self.matrix[row * self.cols + col].movement = movement;
    }

    fn get_row(&self, row: usize) -> &[MatrixCell] {
        let start = row * self.cols;
        &self.matrix[start..start + self.cols]
    }
}

impl SkimMatcherV2 {
    pub fn score_config(mut self, score_config: Box<dyn SkimScoreConfig>) -> Self {
        self.score_config = score_config;
        self
    }

    pub fn element_limit(mut self, elements: usize) -> Self {
        self.element_limit = elements;
        self
    }

    /// Build the score matrix using the algorithm described above
    fn build_score_matrix(
        &self,
        m: &mut ScoreMatrix,
        p: &mut ScoreMatrix,
        choice: &str,
        pattern: &str,
        compressed: bool,
        case_sensitive: bool,
    ) {
        for i in 0..m.rows {
            m.set_score(i, 0, MATRIX_CELL_NEG_INFINITY);
            m.set_movement(i, 0, Movement::Skip);
        }

        for j in 0..m.cols {
            m.set_score(0, j, MATRIX_CELL_NEG_INFINITY);
            m.set_movement(0, j, Movement::Skip);
        }

        for i in 0..p.rows {
            p.set_score(i, 0, MATRIX_CELL_NEG_INFINITY);
            p.set_movement(i, 0, Movement::Skip);
        }

        // p[0][j]: the score of best alignment of p[] and c[..=j] where c[j] is not matched
        for j in 0..p.cols {
            p.set_score(0, j, self.score_config.gap_extension());
            p.set_movement(0, j, Movement::Skip);
        }

        // update the matrix;
        for (i, p_ch) in pattern.chars().enumerate() {
            let mut prev_ch = '\0';

            for (j, c_ch) in choice.chars().enumerate() {
                let row = self.adjust_row_idx(i + 1, compressed);
                let row_prev = self.adjust_row_idx(i, compressed);
                let col = j + 1;
                let col_prev = j;

                // update M matrix
                // M[i][j] = match(i, j) + max(M[i-1][j-1], P[i-1][j-1])
                if let Some(match_score) =
                    self.calculate_match_score(prev_ch, c_ch, p_ch, i, j, case_sensitive)
                {
                    let prev_match_score = m.get_score(row_prev, col_prev);
                    let prev_skip_score = p.get_score(row_prev, col_prev);
                    if prev_match_score >= prev_skip_score {
                        m.set_movement(row, col, Movement::Match);
                    }
                    m.set_score(
                        row,
                        col,
                        (match_score as i32)
                            + max(
                                prev_match_score + self.score_config.bonus_consecutive(),
                                prev_skip_score,
                            ),
                    );
                } else {
                    m.set_score(row, col, MATRIX_CELL_NEG_INFINITY);
                    m.set_movement(row, col, Movement::Skip);
                }

                // update P matrix
                // P[i][j] = max(gap_start + gap_extend + M[i][j-1], gap_extend + P[i][j-1])
                let prev_match_score = self.score_config.gap_start()
                    + self.score_config.gap_extension()
                    + m.get_score(row, col_prev);
                let prev_skip_score =
                    self.score_config.gap_extension() + p.get_score(row, col_prev);
                if prev_match_score >= prev_skip_score {
                    p.set_score(row, col, prev_match_score);
                    p.set_movement(row, col, Movement::Match);
                } else {
                    p.set_score(row, col, prev_skip_score);
                    p.set_movement(row, col, Movement::Skip);
                }

                prev_ch = c_ch;
            }
        }
    }

    /// In case we don't need to backtrack the matching indices, we could use only 2 rows for the
    /// matrix, this function could be used to rotate accessing these two rows.
    fn adjust_row_idx(&self, row_idx: usize, compressed: bool) -> usize {
        if compressed {
            row_idx & 1
        } else {
            row_idx
        }
    }

    /// Calculate the matching score of the characters
    /// return None if not matched.
    fn calculate_match_score(
        &self,
        prev_ch: char,
        c: char,
        p: char,
        c_idx: usize,
        _p_idx: usize,
        case_sensitive: bool,
    ) -> Option<u16> {
        if !char_equal(c, p, case_sensitive) {
            return None;
        }

        let score = self.score_config.score_match();

        // check bonus for start of camel case, etc.
        let prev_ch_type = char_type_of(prev_ch);
        let ch_type = char_type_of(c);
        let mut bonus = self.in_place_bonus(&prev_ch_type, &ch_type);

        // bonus for matching the start of the whole choice string
        if c_idx == 0 {
            bonus *= self.score_config.bonus_first_char_multiplier();
        }

        // penalty on case mismatch
        if !case_sensitive && p != c {
            bonus += self.score_config.penalty_case_mismatch();
        }

        Some(max(0, score + bonus) as u16)
    }

    fn in_place_bonus(&self, prev_char_type: &CharType, char_type: &CharType) -> i32 {
        match (prev_char_type, char_type) {
            (CharType::NonWord, t) if *t != CharType::NonWord => self.score_config.bonus_boundary(),
            (CharType::Lower, CharType::Upper) => self.score_config.bonus_camel123(),
            (t, CharType::Number) if *t != CharType::Number => self.score_config.bonus_camel123(),
            (_, CharType::NonWord) => self.score_config.bonus_non_word(),
            _ => 0,
        }
    }

    fn contains_upper(&self, string: &str) -> bool {
        for ch in string.chars() {
            if ch.is_ascii_uppercase() {
                return true;
            }
        }

        false
    }

    pub fn fuzzy(&self, choice: &str, pattern: &str, with_pos: bool) -> Option<(i64, Vec<usize>)> {
        if pattern.is_empty() {
            return Some((0, Vec::new()));
        }

        let case_sensitive = self.contains_upper(pattern);
        let compressed = !with_pos;

        if !cheap_matches(choice, pattern, case_sensitive) {
            return None;
        }

        if pattern.is_empty() {
            return Some((0, Vec::new()));
        }

        let cols = choice.chars().count() + 1;
        let num_char_pattern = pattern.chars().count();
        let rows = if compressed { 2 } else { num_char_pattern + 1 };

        if self.element_limit > 0 && self.element_limit < rows * cols {
            return self.simple_match(choice, pattern, case_sensitive, with_pos);
        }

        // initialize the score matrix
        let mut m = self
            .m_cache
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut();
        let mut m = ScoreMatrix::new(&mut m, rows, cols);
        let mut p = self
            .p_cache
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut();
        let mut p = ScoreMatrix::new(&mut p, rows, cols);

        self.build_score_matrix(&mut m, &mut p, choice, pattern, compressed, case_sensitive);
        let last_row = m.get_row(self.adjust_row_idx(num_char_pattern, compressed));
        let (pat_idx, &MatrixCell { score, .. }) = last_row
            .iter()
            .enumerate()
            .max_by_key(|&(_, x)| x.score)
            .expect("fuzzy_matcher failed to iterate over last_row");

        let mut positions = Vec::new();
        if with_pos {
            let mut i = m.rows - 1;
            let mut j = pat_idx;
            let mut matrix = &m;
            let mut current_move = Match;
            while i > 0 && j > 0 {
                if current_move == Match {
                    positions.push(j - 1);
                }

                current_move = matrix.get_movement(i, j);
                if ptr::eq(matrix, &m) {
                    i -= 1;
                }

                j -= 1;

                matrix = match current_move {
                    Match => &m,
                    Skip => &p,
                };
            }
            positions.reverse();
        }

        Some((score as i64, positions))
    }

    /// Borrowed from fzf v1, if the memory limit exceeded, fallback to simple linear search
    pub fn simple_match(
        &self,
        choice: &str,
        pattern: &str,
        case_sensitive: bool,
        with_pos: bool,
    ) -> Option<(i64, Vec<usize>)> {
        let mut choice_iter = choice.char_indices().peekable();
        let mut pattern_iter = pattern.chars().peekable();
        let mut o_start_byte = None;

        // scan forward to find the first match of whole pattern
        let mut start_chars = 0;
        while choice_iter.peek().is_some() && pattern_iter.peek().is_some() {
            let (byte_idx, c) = choice_iter.next().unwrap();
            match pattern_iter.peek() {
                Some(&p) => {
                    if char_equal(c, p, case_sensitive) {
                        let _ = pattern_iter.next();
                        o_start_byte = o_start_byte.or(Some(byte_idx));
                    }
                }
                None => break,
            }

            if o_start_byte.is_none() {
                start_chars += 1;
            }
        }

        if pattern_iter.peek().is_some() {
            return None;
        }

        let start_byte = o_start_byte.unwrap_or(0);
        let end_byte = choice_iter
            .next()
            .map(|(idx, _)| idx)
            .unwrap_or(choice.len());

        // scan backward to find the first match of whole pattern
        let mut o_nearest_start_byte = None;
        let mut pattern_iter = pattern.chars().rev().peekable();
        for (idx, c) in choice[start_byte..end_byte].char_indices().rev() {
            match pattern_iter.peek() {
                Some(&p) => {
                    if char_equal(c, p, case_sensitive) {
                        let _ = pattern_iter.next();
                        o_nearest_start_byte = Some(idx);
                    }
                }
                None => break,
            }
        }

        let start_byte = start_byte + o_nearest_start_byte.unwrap_or(0);
        Some(self.calculate_score_with_pos(
            choice,
            pattern,
            start_byte,
            end_byte,
            start_chars,
            case_sensitive,
            with_pos,
        ))
    }

    fn calculate_score_with_pos(
        &self,
        choice: &str,
        pattern: &str,
        start_bytes: usize,
        end_bytes: usize,
        start_chars: usize,
        case_sensitive: bool,
        with_pos: bool,
    ) -> (i64, Vec<usize>) {
        let mut pos = Vec::new();

        let choice_iter = choice[start_bytes..end_bytes].chars().enumerate();
        let mut pattern_iter = pattern.chars().enumerate().peekable();

        // unfortunately we could not get the the character before the first character's(for performance)
        // so we tread them as NonWord
        let mut prev_ch = '\0';

        let mut score: i32 = 0;
        let mut in_gap = false;
        for (c_idx, c) in choice_iter {
            let op = pattern_iter.peek();
            if op.is_none() {
                break;
            }

            let (p_idx, p) = *op.unwrap();

            if let Some(match_score) = self.calculate_match_score(
                prev_ch,
                c,
                p,
                c_idx + start_chars,
                p_idx,
                case_sensitive,
            ) {
                if with_pos {
                    pos.push(c_idx + start_chars);
                }
                score += match_score as i32;
                in_gap = false;
                let _ = pattern_iter.next();
            } else {
                if !in_gap {
                    score += self.score_config.gap_start();
                }

                score += self.score_config.gap_extension();
                in_gap = true;
            }

            prev_ch = c;
        }

        (score as i64, pos)
    }
}

impl FuzzyMatcher for SkimMatcherV2 {
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
        self.fuzzy(choice, pattern, true)
    }

    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<i64> {
        self.fuzzy(choice, pattern, false).map(|(score, _)| score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{assert_order, wrap_matches};

    fn wrap_fuzzy_match(matcher: &dyn FuzzyMatcher, line: &str, pattern: &str) -> Option<String> {
        let (_score, indices) = matcher.fuzzy_indices(line, pattern)?;
        Some(wrap_matches(line, &indices))
    }

    #[test]
    fn test_match_or_not() {
        let matcher = SkimMatcher::default();
        assert_eq!(Some(0), matcher.fuzzy_match("", ""));
        assert_eq!(Some(0), matcher.fuzzy_match("abcdefaghi", ""));
        assert_eq!(None, matcher.fuzzy_match("", "a"));
        assert_eq!(None, matcher.fuzzy_match("abcdefaghi", "中"));
        assert_eq!(None, matcher.fuzzy_match("abc", "abx"));
        assert!(matcher.fuzzy_match("axbycz", "abc").is_some());
        assert!(matcher.fuzzy_match("axbycz", "xyz").is_some());

        assert_eq!(
            "[a]x[b]y[c]z",
            &wrap_fuzzy_match(&matcher, "axbycz", "abc").unwrap()
        );
        assert_eq!(
            "a[x]b[y]c[z]",
            &wrap_fuzzy_match(&matcher, "axbycz", "xyz").unwrap()
        );
        assert_eq!(
            "[H]ello, [世]界",
            &wrap_fuzzy_match(&matcher, "Hello, 世界", "H世").unwrap()
        );
    }

    #[test]
    fn test_match_quality() {
        let matcher = SkimMatcher::default();

        // initials
        assert_order(&matcher, "ab", &["ab", "aoo_boo", "acb"]);
        assert_order(&matcher, "CC", &["CamelCase", "camelCase", "camelcase"]);
        assert_order(&matcher, "cC", &["camelCase", "CamelCase", "camelcase"]);
        assert_order(
            &matcher,
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
            &matcher,
            "Da.Te",
            &["Data.Text", "Data.Text.Lazy", "Data.Aeson.Encoding.text"],
        );
        // prefix
        assert_order(&matcher, "is", &["isIEEE", "inSuf"]);
        // shorter
        assert_order(&matcher, "ma", &["map", "many", "maximum"]);
        assert_order(&matcher, "print", &["printf", "sprintf"]);
        // score(PRINT) = kMinScore
        assert_order(&matcher, "ast", &["ast", "AST", "INT_FAST16_MAX"]);
        // score(PRINT) > kMinScore
        assert_order(&matcher, "Int", &["int", "INT", "PRINT"]);
    }

    #[test]
    fn test_match_or_not_simple() {
        let matcher = SkimMatcherV2::default();
        assert_eq!(
            matcher
                .simple_match("axbycz", "xyz", false, true)
                .unwrap()
                .1,
            vec![1, 3, 5]
        );

        assert_eq!(
            matcher.simple_match("", "", false, false),
            Some((0, vec![]))
        );
        assert_eq!(
            matcher.simple_match("abcdefaghi", "", false, false),
            Some((0, vec![]))
        );
        assert_eq!(matcher.simple_match("", "a", false, false), None);
        assert_eq!(
            matcher.simple_match("abcdefaghi", "中", false, false,),
            None
        );
        assert_eq!(matcher.simple_match("abc", "abx", false, false,), None);
        assert_eq!(
            matcher
                .simple_match("axbycz", "abc", false, true)
                .unwrap()
                .1,
            vec![0, 2, 4]
        );
        assert_eq!(
            matcher
                .simple_match("axbycz", "xyz", false, true)
                .unwrap()
                .1,
            vec![1, 3, 5]
        );
        assert_eq!(
            matcher
                .simple_match("Hello, 世界", "H世", false, true)
                .unwrap()
                .1,
            vec![0, 7]
        );
    }

    #[test]
    fn test_match_or_not_v2() {
        let matcher = SkimMatcherV2::default();

        assert_eq!(matcher.fuzzy_match("", ""), Some(0));
        assert_eq!(matcher.fuzzy_match("abcdefaghi", ""), Some(0));
        assert_eq!(matcher.fuzzy_match("", "a"), None);
        assert_eq!(matcher.fuzzy_match("abcdefaghi", "中"), None);
        assert_eq!(matcher.fuzzy_match("abc", "abx"), None);
        assert!(matcher.fuzzy_match("axbycz", "abc").is_some());
        assert!(matcher.fuzzy_match("axbycz", "xyz").is_some());

        // smart case
        assert!(matcher.fuzzy_match("aBc", "abc").is_some());
        assert!(matcher.fuzzy_match("aBc", "aBc").is_some());
        assert!(matcher.fuzzy_match("aBc", "aBC").is_none());

        assert_eq!(
            &wrap_fuzzy_match(&matcher, "axbycz", "abc").unwrap(),
            "[a]x[b]y[c]z"
        );
        assert_eq!(
            &wrap_fuzzy_match(&matcher, "axbycz", "xyz").unwrap(),
            "a[x]b[y]c[z]"
        );
        assert_eq!(
            &wrap_fuzzy_match(&matcher, "Hello, 世界", "H世").unwrap(),
            "[H]ello, [世]界"
        );
    }

    #[test]
    fn test_matcher_quality_v2() {
        let matcher = SkimMatcherV2::default();
        assert_order(&matcher, "ab", &["ab", "aoo_boo", "acb"]);
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
            &["Data.Text", "Data.Text.Lazy", "Data.Aeson.Encoding.Text"],
        );
        assert_order(&matcher, "is", &["isIEEE", "inSuf"]);
        assert_order(&matcher, "ma", &["map", "many", "maximum"]);
        assert_order(&matcher, "print", &["printf", "sprintf"]);
        assert_order(&matcher, "ast", &["ast", "AST", "INT_FAST16_MAX"]);
        assert_order(&matcher, "int", &["int", "INT", "PRINT"]);
    }
}

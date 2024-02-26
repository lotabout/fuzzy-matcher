pub mod clangd;
pub mod simple;
pub mod skim;
mod util;

type IndexType = usize;
type ScoreType = i64;

pub trait FuzzyMatcher: Send + Sync {
    /// fuzzy match choice with pattern, and return the score & matched indices of characters
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(ScoreType, Vec<IndexType>)>;

    /// fuzzy match choice with pattern, and return the score of matching
    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<ScoreType> {
        self.fuzzy_indices(choice, pattern).map(|(score, _)| score)
    }
}

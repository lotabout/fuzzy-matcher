pub mod clangd;
pub mod skim;
mod util;

#[cfg(not(feature = "compact"))]
type IndexType = usize;
#[cfg(not(feature = "compact"))]
type ScoreType = i64;

#[cfg(feature = "compact")]
type IndexType = u32;
#[cfg(feature = "compact")]
type ScoreType = i32;

pub trait FuzzyMatcher: Send + Sync {
    /// fuzzy match choice with pattern, and return the score & matched indices of characters
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(ScoreType, Vec<IndexType>)>;

    /// fuzzy match choice with pattern, and return the score of matching
    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<ScoreType> {
        self.fuzzy_indices(choice, pattern).map(|(score, _)| score)
    }
}

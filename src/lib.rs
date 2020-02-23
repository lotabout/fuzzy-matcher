pub mod clangd;
pub mod skim;
mod util;

pub trait FuzzyMatcher: Send + Sync {
    /// fuzzy match choice with pattern, and return the score & matched indices of characters
    fn fuzzy_indices(&self, choice: &str, pattern: &str) -> Option<(i64, Vec<usize>)>;

    /// fuzzy match choice with pattern, and return the score of matching
    fn fuzzy_match(&self, choice: &str, pattern: &str) -> Option<i64> {
        self.fuzzy_indices(choice, pattern).map(|(score, _)| score)
    }
}

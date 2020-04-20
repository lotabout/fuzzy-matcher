[![Crates.io](https://img.shields.io/crates/v/fuzzy-matcher.svg)](https://crates.io/crates/fuzzy-matcher)

# Fuzzy Matcher

Fuzzy matching algorithm(s) in Rust!

## Usage

In your Cargo.toml add the following:

```toml
[dependencies]
fuzzy-matcher = "*"
```

Here are some code example:

```rust
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

let matcher = SkimMatcherV2::default();
assert_eq!(None, matcher.fuzzy_match("abc", "abx"));
assert!(matcher.fuzzy_match("axbycz", "abc").is_some());
assert!(matcher.fuzzy_match("axbycz", "xyz").is_some());

let (score, indices) = matcher.fuzzy_indices("axbycz", "abc").unwrap();
assert_eq!(indices, [0, 2, 4]);
```

- `fuzzy_match` only return scores while `fuzzy_indices` returns the matching
    indices as well.
- Both function return None if the pattern won't match.
- The score is the higher the better.

## More example

`echo "axbycz" | cargo run --example fz "abc"` and check what happens.

## About the Algorithm

### Skim

The skim is currently used by [skim](https://github.com/lotabout/skim), a
fuzzy finder.

#### Skim V2

- Just like fzf v2, the algorithm is based on Smith-Waterman algorithm which
    is normally used in DNA sequence alignment
- Also checkout https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/gaps.pdf for
    more details
- The time complexity is `O(mn)` where `m, n` are the length of the pattern
    and input line.
- Space complexity is `O(mn)` for `fuzzy_indices` and `O(2n)` for
    `fuzzy_match` which will compress the table for dynamic programming.
- V2 matcher has an option to set the max element of the score matrix, if
    `m*n` exceeded the limit, it will fallback to a linear search.

#### Skim V1

- It's based on Smith's post [Reverse Engineering Sublime Text’s Fuzzy Match](https://www.forrestthewoods.com/blog/reverse_engineering_sublime_texts_fuzzy_match/)
- The implementation here actually has some flaws that don't perform well
    in certain cases.
- It's recommended to checkout original implementation in [C++](https://github.com/forrestthewoods/lib_fts/blob/master/code/fts_fuzzy_match.h) and [JavaScript](https://github.com/forrestthewoods/lib_fts/blob/master/code/fts_fuzzy_match.js)

### Clangd

- The algorithm is based on [clangd's FuzzyMatch.cpp](https://github.com/MaskRay/ccls/blob/master/src/fuzzy_match.cc).
- Also checkout https://github.com/lewang/flx/issues/98 for some variants.
- The algorithm is `O(mn)` where `m, n` are the length of the pattern and
    input line.
- Space complexity is `O(mn)` for `fuzzy_indices` and `O(2n)` for
    `fuzzy_match` which will compress the table for dynamic programming.

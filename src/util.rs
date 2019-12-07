use crate::FuzzyMatcher;

pub fn cheap_matches(line: &str, pattern: &str) -> bool {
    let mut line_iter = line.chars().peekable();
    let mut pat_iter = pattern.chars().peekable();
    while line_iter.peek().is_some() && pat_iter.peek().is_some() {
        let line_lower = line_iter.peek().unwrap().to_ascii_lowercase();
        let pat_lower = pat_iter.peek().unwrap().to_ascii_lowercase();
        if line_lower == pat_lower {
            pat_iter.next();
        }
        line_iter.next();
    }

    pat_iter.peek().is_none()
}

#[derive(Debug, PartialEq)]
pub enum CharType {
    Empty,
    NonWord,
    Lower,
    Upper,
    Number,
}

#[inline]
pub fn char_type_of(ch: char) -> CharType {
    if ch == '\0' {
        CharType::Empty
    } else if ch.is_lowercase() {
        CharType::Lower
    } else if ch.is_uppercase() {
        CharType::Upper
    } else if ch.is_numeric() {
        CharType::Number
    } else {
        CharType::NonWord
    }
}

#[derive(Debug, PartialEq)]
pub enum CharRole {
    Tail,
    Head,
}

// checkout https://github.com/llvm-mirror/clang-tools-extra/blob/master/clangd/FuzzyMatch.cpp
// The Role can be determined from the Type of a character and its neighbors:
//
//   Example  | Chars | Type | Role
//   ---------+--------------+-----
//   F(o)oBar | Foo   | Ull  | Tail
//   Foo(B)ar | oBa   | lUl  | Head
//   (f)oo    | ^fo   | Ell  | Head
//   H(T)TP   | HTT   | UUU  | Tail
//
//      Curr= Empty Lower Upper Separ
// Prev=Empty 0x00, 0xaa, 0xaa, 0xff, // At start, Lower|Upper->Head
// Prev=Lower 0x00, 0x55, 0xaa, 0xff, // In word, Upper->Head;Lower->Tail
// Prev=Upper 0x00, 0x55, 0x59, 0xff, // Ditto, but U(U)U->Tail
// Prev=Separ 0x00, 0xaa, 0xaa, 0xff, // After separator, like at start
pub fn char_role(prev: char, cur: char) -> CharRole {
    use self::CharRole::*;
    use self::CharType::*;
    match (char_type_of(prev), char_type_of(cur)) {
        (Empty, Lower) | (Empty, Upper) | (Lower, Upper) | (NonWord, Lower) | (NonWord, Upper) => {
            Head
        }
        _ => Tail,
    }
}

#[allow(dead_code)]
pub fn assert_order(matcher: &dyn FuzzyMatcher, pattern: &str, choices: &[&'static str]) {
    let result = filter_and_sort(matcher, pattern, choices);

    if result != choices {
        // debug print
        println!("pattern: {}", pattern);
        for &choice in choices.iter() {
            if let Some((score, indices)) = matcher.fuzzy_indices(choice, pattern) {
                println!("{}: {:?}", score, wrap_matches(choice, &indices));
            } else {
                println!("NO MATCH for {}", choice);
            }
        }
    }

    assert_eq!(result, choices);
}

#[allow(dead_code)]
fn filter_and_sort(
    matcher: &dyn FuzzyMatcher,
    pattern: &str,
    lines: &[&'static str],
) -> Vec<&'static str> {
    let mut lines_with_score: Vec<(i64, &'static str)> = lines
        .iter()
        .map(|&s| (matcher.fuzzy_match(s, pattern).unwrap_or(-(1 << 62)), s))
        .collect();
    lines_with_score.sort_by_key(|(score, _)| -score);
    lines_with_score
        .into_iter()
        .map(|(_, string)| string)
        .collect()
}

#[allow(dead_code)]
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

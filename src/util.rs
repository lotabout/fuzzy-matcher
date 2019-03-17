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

    !pat_iter.peek().is_some()
}

#[derive(Debug, PartialEq)]
pub enum CharType {
    Empty,
    Lower,
    Upper,
    Separ,
}

#[inline]
pub fn char_type_of(ch: char) -> CharType {
    if ch == '\0' {
        CharType::Empty
    } else if ch == ' ' || ch == '_' || ch == '-' || ch == '/' || ch == '\\' {
        CharType::Separ
    } else if ch.is_ascii_uppercase() {
        CharType::Upper
    } else {
        CharType::Lower
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
        (Empty, Lower) | (Empty, Upper) | (Lower, Upper) | (Separ, Lower) | (Separ, Upper) => Head,
        _ => Tail,
    }
}

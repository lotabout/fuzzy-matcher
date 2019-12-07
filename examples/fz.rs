use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcher;
use std::env;
use std::io::{self, BufRead};
use std::process::exit;
use termion::style::{Invert, Reset};

pub fn main() {
    let matcher = SkimMatcher::default();
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: echo <piped_input> | fz <pattern>");
        exit(1);
    }

    let pattern = &args[1];

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            if let Some((score, indices)) = matcher.fuzzy_indices(&line, pattern) {
                println!("{:8}: {}", score, wrap_matches(&line, &indices));
            }
        }
    }
}

fn wrap_matches(line: &str, indices: &[usize]) -> String {
    let mut ret = String::new();
    let mut peekable = indices.iter().peekable();
    for (idx, ch) in line.chars().enumerate() {
        let next_id = **peekable.peek().unwrap_or(&&line.len());
        if next_id == idx {
            ret.push_str(format!("{}{}{}", Invert, ch, Reset).as_str());
            peekable.next();
        } else {
            ret.push(ch);
        }
    }

    ret
}

use fuzzy_matcher::*;

fn wrap_matches(line: &str, indices: &[usize]) -> String {
    let mut ret = String::new();
    let mut peekable = indices.iter().peekable();
    for (idx, ch) in line.chars().enumerate() {
        let next_id = **peekable.peek().unwrap_or(&&line.len());
        if (next_id == idx) {
            ret.push_str(format!("[{}]", ch).as_str());
            peekable.next();
        } else {
            ret.push(ch);
        }
    }

    ret
}

pub fn main() {
    let string = "axbycz";
    let pattern = "abc";
    let (score, indices) = fuzzy_indices(string, pattern).unwrap();
    println!("{:?}", wrap_matches(string, &indices));
}

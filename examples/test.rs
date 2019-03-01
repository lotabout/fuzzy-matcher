use seq_align::*;

pub fn main() {
    let res = fuzzy_match("aaxbxcbbcc", "axc");
    println!("");
    println!("{:?}", res);
}

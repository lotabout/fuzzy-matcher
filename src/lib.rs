const AWFUL_SCORE: i32 = -(1 << 13);

#[derive(PartialEq, Clone, Copy)]
enum Action {
    Miss,
    Match,
}

#[derive(Clone, Copy)]
struct ScoreInfo {
    pub score: i32,
    pub last_action: Action,
}

impl Default for ScoreInfo {
    fn default() -> Self {
        Self {
            score: AWFUL_SCORE,
            last_action: Action::Miss,
        }
    }
}

#[derive(Clone, Copy)]
struct Score {
    pub miss_score: ScoreInfo,
    pub match_score: ScoreInfo,
}

impl Default for Score {
    fn default() -> Self {
        Self {
            miss_score: Default::default(),
            match_score: Default::default(),
        }
    }
}

pub fn fuzzy_match(line: &str, pattern: &str) -> Option<(i64, Vec<usize>)> {
    let num_line_chars = line.chars().count();
    let num_pattern_chars = pattern.chars().count();

    let mut dp: Vec<Vec<Score>> = Vec::with_capacity(num_pattern_chars + 1);
    for _ in 0..(num_pattern_chars + 1) {
        dp.push(vec![Score::default(); num_line_chars + 1]);
    }

    // first line
    for (idx, ch) in line.chars().enumerate() {
        dp[0][idx + 1] = Score {
            miss_score: ScoreInfo {
                score: &dp[0][idx].miss_score.score - skip_penalty(idx, ch, Action::Miss),
                last_action: Action::Miss,
            },
            match_score: ScoreInfo {
                score: AWFUL_SCORE,
                last_action: Action::Miss,
            },
        };
    }

    // build the matrix
    for (pat_idx, pat_ch) in pattern.chars().enumerate() {
        for (line_idx, line_ch) in line.chars().enumerate().skip(pat_idx) {
            let pre_miss = &dp[pat_idx + 1][line_idx];
            let mut match_miss_score = pre_miss.match_score.score;
            let mut miss_miss_score = pre_miss.miss_score.score;
            if pat_idx < num_pattern_chars - 1 {
                match_miss_score -= skip_penalty(pat_idx, pat_ch, Action::Match);
                miss_miss_score -= skip_penalty(pat_idx, pat_ch, Action::Miss);
            }

            let miss_score = if match_miss_score > miss_miss_score {
                ScoreInfo {
                    score: match_miss_score,
                    last_action: Action::Match,
                }
            } else {
                ScoreInfo {
                    score: miss_miss_score,
                    last_action: Action::Miss,
                }
            };

            let pre_match = &dp[pat_idx][line_idx];
            let match_match_score =
                if allow_match(pat_idx, pat_ch, line_idx, line_ch, Action::Match) {
                    pre_match.match_score.score
                        + match_bonus(pat_idx, pat_ch, line_idx, line_ch, Action::Match)
                } else {
                    AWFUL_SCORE
                };

            let miss_match_score = if allow_match(pat_idx, pat_ch, line_idx, line_ch, Action::Miss)
            {
                pre_match.miss_score.score
                    + match_bonus(pat_idx, pat_ch, line_idx, line_ch, Action::Miss)
            } else {
                AWFUL_SCORE
            };

            let match_score = if match_match_score > miss_match_score {
                ScoreInfo {
                    score: match_match_score,
                    last_action: Action::Match,
                }
            } else {
                ScoreInfo {
                    score: miss_match_score,
                    last_action: Action::Miss,
                }
            };

            dp[pat_idx + 1][line_idx + 1] = Score {
                miss_score,
                match_score,
            };
        }
    }

    let mut indices_reverse = Vec::new();
    let mut row = num_pattern_chars;
    let mut col = num_line_chars;

    let cell = dp[row][col];
    let (mut last_action, score) = if cell.match_score.score > cell.miss_score.score {
        (cell.match_score.last_action, cell.match_score.score)
    } else {
        (cell.miss_score.last_action, cell.miss_score.score)
    };

    if score <= AWFUL_SCORE {
        return None;
    }

    // search backwards for the matched indices
    while row > 0 || col > 0 {
        if last_action == Action::Match {
            indices_reverse.push(col-1);
        }

        let cell = &dp[row][col];
        if last_action == Action::Match {
            last_action = cell.match_score.last_action;
            row -=1;
            col -= 1;
        } else {
            last_action = cell.miss_score.last_action;
            col -= 1;
        }
    }

    indices_reverse.reverse();
    Some((score as i64, indices_reverse))
}

fn skip_penalty(_ch_idx: usize, _ch: char, last_action: Action) -> i32 {
    if last_action == Action::Match {
        // Non-consecutive match.
        return 2;
    } else {
        return 0;
    }
}

fn allow_match(
    pat_idx: usize,
    pat_ch: char,
    line_idx: usize,
    line_ch: char,
    last_action: Action,
) -> bool {
    pat_ch.to_ascii_lowercase() == line_ch.to_ascii_lowercase()
}

fn match_bonus(
    pat_idx: usize,
    pat_ch: char,
    line_idx: usize,
    line_ch: char,
    last_action: Action,
) -> i32 {
    20
}

fn print_dp(line: &str, pattern: &str, dp: Vec<Vec<Score>>) {
    let num_line_chars = line.chars().count();
    let num_pattern_chars = pattern.chars().count();

    print!("\t");
    for (idx, ch) in line.chars().enumerate() {
        print!("\t\t{}/{}", idx + 1, ch);
    }

    for row in 0..(num_pattern_chars + 1) {
        print!("\n{}\t", row);
        for col in 0..(num_line_chars + 1) {
            let cell = &dp[row][col];
            print!(
                "({},{})/({},{})\t",
                cell.miss_score.score,
                if cell.miss_score.last_action == Action::Miss {
                    'X'
                } else {
                    'O'
                },
                cell.match_score.score,
                if cell.match_score.last_action == Action::Miss {
                    'X'
                } else {
                    'O'
                }
            );
        }
    }
}

pub mod legacy;

use rand::seq::SliceRandom;

use std::f64::consts::E;
use std::io::stdin;
use rand::prelude::*;

fn select_from_probabilities(probabilities: &Vec<f64>) -> (usize, f64) {
    let mut rng = thread_rng();
    let mut cumulative_prob = 0.0;
    let total_prob = probabilities.iter().sum::<f64>();

    let random_value = rng.gen_range(0.0..total_prob);

    for (i, prob) in probabilities.iter().enumerate() {
        cumulative_prob += *prob;
        if random_value < cumulative_prob {
            return (i, *prob);
        }
    }

    // If we reach here, we shouldn't have.
    panic!("Could not select a value from the provided probabilities.");
}

fn softmax(x: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; x.len()];
    let sum: f64 = x.iter().map(|&v| E.powf(v)).sum();

    for i in 0..x.len() {
        result[i] = E.powf(x[i]) / sum;
    }

    result
}

fn init() -> (Vec<[usize; 3]>, Vec<[usize; 3]>, Vec<[usize; 7]>) {
    // 初始化蜂巢图
    let comb = vec![[0, 0, 0]; 20];

    // 初始化发牌
    let mut card_list = Vec::new();
    for _ in 0..2 {
        for i in &[3, 4, 8] {
            for j in &[1, 5, 9] {
                for k in &[2, 6, 7] {
                    card_list.push([*i, *j, *k]);
                }
            }
        }
        card_list.push([10, 10, 10]);
    }
    let mut ok = false;
    card_list.shuffle(&mut rand::thread_rng());
    for card in &card_list[..20] {
        if *card == [10, 10, 10] {
            ok = true;
            break;
        }
    }
    if !ok {
        for i in 0..20 {
            card_list.swap(i, i + 20);
        }
    }

    // 初始化每一“行”。lines中的每个元素意思为：该行长度 该行类别（左上右下 上下 右上左下） 后方是该行经过的格子
    let mut lines = Vec::new();
    lines.push([3, 0, 8, 13, 17, 0, 0]);
    lines.push([4, 0, 4, 9, 14, 18, 0]);
    lines.push([5, 0, 1, 5, 10, 15, 19]);
    lines.push([4, 0, 2, 6, 11, 16, 0]);
    lines.push([3, 0, 3, 7, 12, 0, 0]);

    lines.push([3, 1, 1, 2, 3, 0, 0]);
    lines.push([4, 1, 4, 5, 6, 7, 0]);
    lines.push([5, 1, 8, 9, 10, 11, 12]);
    lines.push([4, 1, 13, 14, 15, 16, 0]);
    lines.push([3, 1, 17, 18, 19, 0, 0]);

    lines.push([3, 2, 1, 4, 8, 0, 0]);
    lines.push([4, 2, 2, 5, 9, 13, 0]);
    lines.push([5, 2, 3, 6, 10, 14, 17]);
    lines.push([4, 2, 7, 11, 15, 18, 0]);
    lines.push([3, 2, 12, 16, 19, 0, 0]);

    (comb, card_list, lines)
}

fn row_status(
    comb: &Vec<[usize; 3]>,
    line: &[usize; 7],
) -> (String, usize, usize, f64, usize, usize) {
    // 该行状态：full partial empty damaged
    let status;
    // 该行长度
    let length = line[0];
    // 该行种类
    let type_num = line[1];
    // 该行已经填了几个
    let mut filled = 0;
    // 该行对应的数字
    let mut num = 0;
    // 该行得分
    let mut score = 0;
    for i in 2..2 + length {
        let now = comb[line[i] as usize][type_num as usize];
        if now != 0 {
            filled += 1;
        }
        if now != 0 && now != 10 {
            // 损坏行
            if num != 0 && num != now {
                status = String::from("broken");
                score = 0;
                return (status, length, type_num, score as f64, num, filled);
            }
            // 该行数字
            num = now;
        }
    }

    // 已经填满的行
    if filled == length {
        status = String::from("full");
        score = num * length;
    }
    // 全空的行
    else if filled == 0 {
        status = String::from("empty");
        score = 10 * length;
    }
    // 填满部分的行
    else {
        status = String::from("partial");
        score = num * length;
        if num == 0 {
            score = 10 * length;
            num = 10;
        }
    }

    return (status, length, type_num, score as f64, num, filled);
}

use std::f64;

use std::vec::Vec;

fn exp_score(comb: &Vec<[usize; 3]>, lines: &Vec<[usize; 7]>, vars: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    let mut block_count: f64 = 0.0;
    for i in 0..20 {
        if comb[i] != [0, 0, 0] {
            block_count += 1.;
        }
    }

    let mut last_num: usize = 0;
    let mut last_score: f64 = 0.0;
    let mut _selected: Vec<usize> = vec![0; 11];
    let mut desired: Vec<f64> = vec![0.0; 10];
    let mut waiting: Vec<f64> = vec![0.0; 10];
    let mut decide: Vec<[f64; 2]> = vec![[0.0, 0.0]; 20];
    let mut needs: Vec<i32> = vec![0; 10];
    let mut line_needs: Vec<i32> = vec![0; 15];

    // 计算0方块的分数
    if comb[0] == [0, 0, 0] {
        sum += vars[10];
    } else {
        sum += (comb[0][0] + comb[0][1] + comb[0][2]) as f64;
    }

    let mut adj = 0.1;
    // 对于每一行
    for i in 0..15 {
        let (status, length, _type, score, num, filled) = row_status(comb, &lines[i]);
        let scale = vars[length - filled];

        sum += scale * score;

        // 越后期，已连通的价值越高。
        if status != "full" {
            sum -= scale * (1.0 - (0.993 as f64).powf(block_count)) * score;
        }

        // 尽可能使得游戏开局没有相邻元素，变量var[7]。
        if block_count < 20.0 {
            if status == "partial" {
                if num == last_num && num != 0 && num != 10 {
                    sum -= last_score * vars[7] * adj;
                    sum -= (last_num as f64 / 2.0).sqrt() * adj;
                    adj += 0.7;
                }
                last_num = num;
                last_score = scale * score;
            } else {
                last_num = 0;
                last_score = 0.0;
            }
        }
        // 尽可能使得游戏开局不破坏行。
        if block_count < 10.0 {
            if status == "broken" {
                sum -= (num as f64).sqrt();
            }
        }
        // 尽可能使得游戏开局不一次开太多行，变量var[6]。
        if num != 0 && num != 10 && status == "partial" {
            desired[num] += (length - filled) as f64;
            waiting[num] += scale * score;
        }
        // 降低交错点的期望得分，变量var[8], var[9]。
        if status == "partial" {
            for j in 2..2 + length {
                if comb[lines[i][j]] == [0, 0, 0] {
                    decide[lines[i][j]][0] += 1.;
                    decide[lines[i][j]][1] += scale * score;
                }
            }
        }
        // 计算每个数字有多少行
        if num != 0 && num != 10 {
            needs[num] = needs[num] + 1;
            // 记录该行需要多少方块
        }
        line_needs[i] = line_needs[i] + 1;
    }
    let mut scale;
    for i in 1..10 {
        scale = (desired[i] * vars[6] / 10.0).powf(2.);
        if desired[i] < 5.0 || needs[i] < 3 {
            scale = 0.0;
        }
        sum = sum - scale * waiting[i];
    }
    // 降低交点牌得分概率
    scale = (block_count / 20.0).powf(2.);
    let mut times = 0.4;
    for i in 0..20 {
        if comb[i] == [10, 10, 10] {
            times = 1.
        }
    }
    scale = scale * times;
    for i in 0..20 {
        if decide[i][0] == 2. {
            sum = sum - scale * vars[8] * decide[i][1];
        }
        if decide[i][0] == 3. {
            sum = sum - scale * vars[9] * decide[i][1];
        }
    }
    return sum;
}

fn step<'a>(
    comb: &'a mut Vec<[usize; 3]>,
    lines: &'a Vec<[usize; 7]>,
    vars: &'a Vec<f64>,
    now: [usize; 3],
    log: bool,
) -> (&'a mut Vec<[usize; 3]>, usize) {
    let mut mx = -1.;
    let mut put = 0;
    let mut exp = vec![];
    for i in 0..20 {
        if comb[i] == [0, 0, 0] {
            comb[i] = now;
            let score = exp_score(comb, lines, vars);
            if score > mx {
                if score >= mx + 0.4 {
                    mx = score;
                    put = i;
                }else if rand::thread_rng().gen_bool((score - mx) / 0.4 * 0.3 + 0.7) {
                    mx = score;
                    put = i;
                }
            }else if score > mx - 0.4 {
                if !rand::thread_rng().gen_bool((mx - score) / 0.4 * 0.3 + 0.7) {
                    mx = score;
                    put = i;
                }
            }
            exp.push(score);
            comb[i] = [0, 0, 0];
        } else {
            exp.push(-1.)
        }
    }
    //put = select_from_probabilities(so&softmax(&exp)).0;
    comb[put] = now;
    if log {
        let mut map = exp
            .iter()
            .enumerate()
            .map(|(k, v)| (k, v))
            .collect::<Vec<(usize, &f64)>>();
        map.sort_by(|(k1, v1), (k2, v2)| v2.partial_cmp(v1).unwrap());
        for i in 0..3 {
            println!("建议位置: {} 得分: {}", map[i].0, map[i].1);
        }
    }
    return (comb, put);
}

fn step_eval<'a>(
    comb: &'a mut Vec<[usize; 3]>,
    lines: &'a Vec<[usize; 7]>,
    vars: &'a Vec<f64>,
    now: [usize; 3],
    log: bool,
) -> (&'a mut Vec<[usize; 3]>, usize) {
    let mut mx = -1.;
    let mut put = 0;
    let mut exp = vec![];
    for i in 0..20 {
        if comb[i] == [0, 0, 0] {
            comb[i] = now;
            let score = exp_score(comb, lines, vars);
            if score > mx {
                mx = score;
                put = i;
            }
            exp.push(score);
            comb[i] = [0, 0, 0];
        } else {
            exp.push(-1.)
        }
    }
    //put = select_from_probabilities(so&softmax(&exp)).0;
    if log {
        let mut map = exp
            .iter()
            .enumerate()
            .map(|(k, v)| (k, v))
            .collect::<Vec<(usize, &f64)>>();
        map.sort_by(|(k1, v1), (k2, v2)| v2.partial_cmp(v1).unwrap());
        for i in 0..3 {
            println!("建议位置: {} 得分: {}", map[i].0, map[i].1);
        }
    }
    return (comb, put);
}

fn eval_play(vars: &Vec<f64>) {
    let (mut comb, _, lines) = init();
    for i in 0..20 {
        let mut buf = String::new();
        println!("input card");
        while stdin().read_line(&mut buf).err().is_none() {
            let card: [usize; 3];
            if buf.trim().len() == 6 {
                card = [10; 3];
            }else {
                let mut chars = buf.chars();
                card = [chars.next().unwrap().to_digit(10).unwrap().try_into().unwrap(), chars.next().unwrap().to_digit(10).unwrap().try_into().unwrap(), chars.next().unwrap().to_digit(10).unwrap().try_into().unwrap()]
            }
            let (_, _) = step_eval(&mut comb, &lines, vars, card, true);
            let mut buf_put = String::new();
            while stdin().read_line(&mut buf_put).err().is_none() {
                let p = buf_put.trim().parse::<usize>().unwrap();
                comb[p] = card;
                break;
            }
            break;
        }
        /*println!(
            "card: {},{},{} put: {}",
            card_list[i][0], card_list[i][1], card_list[i][2], put
        );*/
    }
}

fn self_play_compare(vars: &Vec<f64>, vars2: &Vec<f64>) -> (f64, f64) {
    let (mut comb, card_list, lines) = init();
    let score_legacy = {
        let mut comb = comb.clone();
        for i in 0..20 {
            let (_, put) = legacy::step(&mut comb, &lines, vars2, card_list[i], false);
            /*println!(
                "card: {},{},{} put: {}",
                card_list[i][0], card_list[i][1], card_list[i][2], put
            );*/
        }
        legacy::exp_score(&mut comb, &lines, vars)
    };
    for i in 0..20 {
        let (_, put) = step(&mut comb, &lines, vars, card_list[i], false);
        /*println!(
            "card: {},{},{} put: {}",
            card_list[i][0], card_list[i][1], card_list[i][2], put
        );*/
    }
    let score = exp_score(&mut comb, &lines, vars);
    return (score, score_legacy);
}

fn self_play(vars: &Vec<f64>) -> f64 {
    let (mut comb, card_list, lines) = init();
    for i in 0..20 {
        let (_, put) = step(&mut comb, &lines, vars, card_list[i], false);
        /*println!(
            "card: {},{},{} put: {}",
            card_list[i][0], card_list[i][1], card_list[i][2], put
        );*/
    }
    let score = exp_score(&mut comb, &lines, vars);
    return score;
}

fn test(vars: &Vec<f64>, times: usize) -> f64 {
    let mut highest = 0.;
    let mut lowest = 3000.;
    let mut score = 0.;
    let mut score_legacy = 0.;
    let mut p_score = 0.;
    for i in 0..times {
        let sc = self_play(vars);
        score += sc;
        p_score += sc;
        if sc > highest {
            highest = sc;
        }
        if sc < lowest {
            lowest = sc;
        }
        if (i + 1) % 10000 == 0 {
            println!(
                "Evaluated {} games, Part avg is {}, Total avg is {}, high: {}, low: {}",
                i + 1,
                p_score / 10000.,
                score / (i as f64 + 1.),
                highest,
                lowest
            );
            p_score = 0.;
        }
    }
    println!(
        "-------------------------------\nEvaluated {} games, Avg score is {}, high: {}, low: {}",
        times,
        score / times as f64,
        highest,
        lowest
    );
    return score / times as f64;
}

fn evaluate(vars: &Vec<f64>, vars2: &Vec<f64>, times: usize) -> f64 {
    let mut highest = 0.;
    let mut lowest = 3000.;
    let mut score = 0.;
    let mut score_legacy = 0.;
    let mut p_score = 0.;
    for i in 0..times {
        let (sc, sc_legacy) = self_play_compare(vars, vars2);
        score += sc;
        score_legacy += sc_legacy;
        p_score += sc;
        if sc > highest {
            highest = sc;
        }
        if sc < lowest {
            lowest = sc;
        }
        if (i + 1) % 10000 == 0 {
            println!(
                "Evaluated {} games, Part avg is {}, Total avg is {}, Legacy avg is {}, high: {}, low: {}",
                i + 1,
                p_score / 10000.,
                score / (i as f64 + 1.),
                score_legacy / (i as f64 + 1.),
                highest,
                lowest
            );
            p_score = 0.;
        }
    }
    println!(
        "-------------------------------\nEvaluated {} games, Avg score is {}(Legacy {}), high: {}, low: {}",
        times,
        score / times as f64,
        score_legacy / times as f64,
        highest,
        lowest
    );
    return score / times as f64;
}

fn main() {
    let vars = vec![
        1.00, 0.721, 0.3993, 0.1947, 0.069, 0.0312, 0.75, 0.008, 0.08465, 0.08164, 18.,
    ];
    
    let vars2 = vec![
        1.00, 0.721, 0.3993, 0.1947, 0.069, 0.0312, 0.75, 0.008, 0.08465, 0.08164, 18.,
    ];
    //eval_play(&vars);
    evaluate(&vars, &vars2, 100000);
    //test(&vars, 1000000);
    
    
}

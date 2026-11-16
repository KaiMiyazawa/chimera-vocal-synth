//! Research probe (pillar 2): F0 just after an obstruent, with nothing in the plan asking for it.
//!
//! Each item is a long lead /a/ (0.5 s, so the two-mass folds' trim loop settles at the asked F0),
//! then a consonant and a vowel. The plan holds F0 flat (no stress, accent 0, flat intonation);
//! roughness, drift and vibrato are 0. Sources: the pulse (F0 is set, so it should stay flat: the
//! control), the folds with their trim loop, and the folds with the loop frozen 0.1 s before the
//! consonant (at the operating point the loop reached), so nothing pulls F0 back.
//!
//! F0 is read cycle by cycle from successive peaks (cycle_check's reader, validated on the pulse
//! after the consonant: within ±1 Hz from the first voiced cycle to 60 ms in): of the pulse's area,
//! and of the folds' upper-mass displacement, which carries no abduction offset. Reported: the
//! mean change of F0 (Hz) in windows after the voicing onset (the start of the first cycle after
//! the consonant's release, the end of its segment; `delay_ms` is how long after the release),
//! against the mean F0 of the 100 ms that end 60 ms before the consonant, as mean and range over
//! ten start offsets (0-9 ms). Before the consonant the reader is not within ±1 Hz (cycle shape
//! changes), so nothing is reported there.
//! Run: `cargo run --release -q -p chimera --example microprosody_probe > out.tsv`
use chimera::phonetics::Intonation;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const F0: f64 = 110.0;
const WINDOWS: [(f64, f64); 5] = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.06)];

/// (time of the cycle's midpoint, F0) from successive peaks of `y` (cycle_check's reader).
fn cycles_from_peaks(y: &[f64], half: usize, min_prominence: f64) -> Vec<(f64, f64)> {
    let n = y.len();
    let mut peaks: Vec<f64> = vec![];
    let mut last_peak_i: Option<usize> = None;
    for i in half..n.saturating_sub(half) {
        let w = &y[i - half..=i + half];
        if w.iter().all(|&v| v <= y[i]) && w.iter().filter(|&&v| v == y[i]).count() == 1 {
            let lo = last_peak_i.map_or(&y[i.saturating_sub(4 * half)..i], |j| &y[j..i]).iter().cloned().fold(f64::MAX, f64::min);
            if y[i] - lo >= min_prominence {
                let (a, b, c) = (y[i - 1], y[i], y[i + 1]);
                let den = a - 2.0 * b + c;
                let off = if den.abs() > 1e-18 { 0.5 * (a - c) / den } else { 0.0 };
                peaks.push((i as f64 + off) / FS);
                last_peak_i = Some(i);
            }
        }
    }
    peaks.windows(2).filter(|w| w[1] - w[0] < 0.015).map(|w| (0.5 * (w[0] + w[1]), 1.0 / (w[1] - w[0]))).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Src {
    Pulse,
    Folds,
    Frozen,
}

/// Mean F0 change per window after the release, and the first voiced cycle's delay (ms).
fn measure(ipa: &str, aspirated: Option<bool>, src: Src, offset: f64) -> (Vec<Option<f64>>, f64) {
    let mut input = tokenize_ipa(ipa);
    input[0].dur = Some(0.5);
    for s in input.iter_mut() {
        s.stress = Some(0);
        if ["p", "t", "k"].contains(&s.ipa.as_str()) {
            s.aspirated = aspirated;
        }
    }
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05 + offset, f0: Some(F0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
    let c = plan.segments[1].clone();
    let mut v = HumanVoice::with_defaults(FS);
    let model = if src == Src::Pulse { GlottisModel::Parametric } else { GlottisModel::TwoMass };
    v.set_source(&SourcePatch { glottis: Some(model), roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.05)) as usize;
    let freeze = ((c.t0 - 0.1) * FS) as usize;
    let mut out = [0.0f32; 1];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        if src == Src::Frozen && i == freeze {
            v.folds.trim_loop = false;
        }
        v.render(&mut out);
        y.push(if src == Src::Pulse { v.glottis.area } else { v.folds.displacement() });
    }
    let (half, prom) = ((0.0025 * FS) as usize, if src == Src::Pulse { 0.003 } else { 0.002 });
    let cyc = cycles_from_peaks(&y, half, prom);
    let mean = |a: f64, b: f64| {
        let s: Vec<f64> = cyc.iter().filter(|x| x.0 >= a && x.0 < b).map(|x| x.1).collect();
        (!s.is_empty()).then(|| s.iter().sum::<f64>() / s.len() as f64)
    };
    let reference = mean(c.t0 - 0.16, c.t0 - 0.06).unwrap_or(f64::NAN);
    // windows run from the voicing onset: the start of the first cycle after the release
    let first = cyc.iter().find(|x| x.0 >= c.t1 - 0.005).copied();
    let delay = first.map_or(f64::NAN, |x| (x.0 - 0.5 / x.1 - c.t1) * 1000.0);
    let onset = first.map_or(f64::NAN, |x| x.0 - 0.5 / x.1);
    (WINDOWS.iter().map(|&(a, b)| mean(onset + a, onset + b).map(|f| f - reference)).collect(), delay)
}

fn main() {
    // (item, consonant shown, aspirated stops?)
    let items: [(&str, &str, Option<bool>); 16] = [
        ("apa", "p˭", Some(false)),
        ("ata", "t˭", Some(false)),
        ("aka", "k˭", Some(false)),
        ("apa", "pʰ", Some(true)),
        ("ata", "tʰ", Some(true)),
        ("aka", "kʰ", Some(true)),
        ("aba", "b", None),
        ("ada", "d", None),
        ("aga", "g", None),
        ("asa", "s", None),
        ("aɕi", "ɕ(i)", None),
        ("açi", "ç(i)", None),
        ("aɸɯ", "ɸ(ɯ)", None),
        ("aha", "h", None),
        ("aza", "z", None),
        ("ama", "m", None),
    ];
    let rev = std::process::Command::new("git").args(["describe", "--always", "--dirty"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    println!("# microprosody_probe at {}", rev.trim());
    println!("# male, F0 {F0} Hz flat, roughness/drift/vibrato 0, lead /a/ 0.5 s, frozen loop 0.1 s before the consonant; 10 start offsets; mean [min,max] Hz vs the 100 ms ending 60 ms before the consonant");
    println!("source\titem\tC\tdelay_ms\tonset+0-10ms\t10-20ms\t20-30ms\t30-40ms\t40-60ms");
    let jobs: Vec<(Src, &str, &str, Option<bool>)> = [Src::Pulse, Src::Folds, Src::Frozen].iter().flat_map(|&s| items.iter().map(move |&(w, c, a)| (s, w, c, a))).collect();
    let lines: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&(src, w, c, asp)| {
                sc.spawn(move || {
                    let runs: Vec<(Vec<Option<f64>>, f64)> = (0..10).map(|o| measure(w, asp, src, o as f64 * 0.001)).collect();
                    let cell = |k: usize| {
                        let xs: Vec<f64> = runs.iter().filter_map(|r| r.0[k]).collect();
                        if xs.is_empty() {
                            return "-".to_string();
                        }
                        let m = xs.iter().sum::<f64>() / xs.len() as f64;
                        let (lo, hi) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
                        format!("{m:+.1} [{lo:+.1},{hi:+.1}]{}", if xs.len() < runs.len() { format!(" n={}", xs.len()) } else { String::new() })
                    };
                    let delay = runs.iter().map(|r| r.1).filter(|d| d.is_finite()).sum::<f64>() / runs.len() as f64;
                    let name = match src {
                        Src::Pulse => "pulse",
                        Src::Folds => "folds",
                        Src::Frozen => "folds-frozen",
                    };
                    format!("{name}\t{w}\t{c}\t{delay:.1}\t{}\t{}\t{}\t{}\t{}", cell(0), cell(1), cell(2), cell(3), cell(4))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for l in lines {
        println!("{l}");
    }
}

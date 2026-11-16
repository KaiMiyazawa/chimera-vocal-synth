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
//! `--early`: the stiffening timed to beat the first cycle (see `early_table`). `--cubic` gives
//! the folds the cubic springs and contact of Ishizaka & Flanagan 1972 (100 and 500 cm⁻²).
//! Run: `cargo run --release -q -p chimera --example microprosody_probe [-- --stiffness | --early] [--cubic] > out.tsv`
use chimera::phonetics::Intonation;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
/// the folds' cubic springs and contact (η, cm⁻²) for every run (`--cubic`)
static CUBIC: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
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
/// Stiffening (research): `amount` of the tension factor, from the consonant's oral closure (its
/// segment start) to its release (its end), decaying over `tau` after it; or, `glottal`, on the
/// consonant's own glottal-opening gesture with the same decay. None: no stiffening.
#[derive(Clone, Copy)]
struct Stiff {
    amount: f64,
    tau: f64,
    glottal: bool,
    /// `early_table`: from 30 ms before the oral closure, held `hold` past the release, then let
    /// go over a ramp of `tau` (instead of the closure-to-release span with a 3·tau ramp)
    early: Option<f64>,
}

fn measure(ipa: &str, aspirated: Option<bool>, src: Src, offset: f64) -> (Vec<Option<f64>>, f64) {
    let (w, d, _) = measure_with(ipa, aspirated, src, offset, None);
    (w, d)
}

/// As `measure`, with stiffening; also returns the mean stiffening input in each window.
fn measure_with(ipa: &str, aspirated: Option<bool>, src: Src, offset: f64, stiff: Option<Stiff>) -> (Vec<Option<f64>>, f64, Vec<f64>) {
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
    (v.folds.spring_cubic, v.folds.contact_cubic) = *CUBIC.get().unwrap_or(&(0.0, 0.0));
    v.schedule(&plan.gestures);
    let mut track: Vec<(chimera::control::score::Gesture, f64)> = vec![];
    if let Some(st) = stiff {
        use chimera::control::score::{Gesture, Targets};
        if let Some(hold) = st.early {
            track.push((Gesture::new(c.t0, c.t1 + hold, 0.03, st.tau, 1.0, Targets::default()), st.amount));
        } else if st.glottal {
            for g in plan.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| ["aspiration", "voiceless", "h"].contains(&l.as_str())) && g.t0 < c.t1 && g.t1 > c.t0) {
                let mut x = g.clone();
                x.ramp_out = 3.0 * st.tau;
                track.push((x, st.amount));
            }
        } else {
            track.push((Gesture::new(c.t0, c.t1, 0.03, 3.0 * st.tau, 1.0, Targets::default()), st.amount));
        }
        v.stiffness_track = track.clone();
        v.folds.stiffness_keeps_gap = true;
    }
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
    let input = |a: f64, b: f64| {
        let k = 20;
        (0..k).map(|j| {
            let t = a + (b - a) * (j as f64 + 0.5) / k as f64;
            track.iter().map(|(g, s)| chimera::control::score::activation(g, t) * s).fold(0.0, f64::max)
        }).sum::<f64>() / k as f64
    };
    (WINDOWS.iter().map(|&(a, b)| mean(onset + a, onset + b).map(|f| f - reference)).collect(), delay, WINDOWS.iter().map(|&(a, b)| if onset.is_finite() { input(onset + a, onset + b) } else { f64::NAN }).collect())
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
    if std::env::args().any(|a| a == "--cubic") {
        CUBIC.set((100.0, 500.0)).ok();
    }
    if std::env::args().any(|a| a == "--early") {
        early_table(&items);
        return;
    }
    if std::env::args().any(|a| a == "--stiffness") {
        stiffness_table(&items);
        return;
    }
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

/// Stiffening sweep on the folds with the loop frozen. Each cell: the F0 change against the same
/// item without stiffening, less what the input alone predicts (the reference F0 times the mean
/// stiffening in the window): what the mechanics add beyond the input. Mean over ten offsets.
fn stiffness_table(items: &[(&str, &str, Option<bool>)]) {
    println!("# stiffening, folds with the loop frozen; cells: dF0 vs no stiffening (Hz) | residual beyond the input (Hz); windows from the voicing onset");
    println!("anchor\tamount\ttau_ms\titem\tC\tdelay_ms\t0-10ms\t10-20ms\t20-30ms\t30-40ms\t40-60ms");
    let mut jobs = vec![];
    for glottal in [false, true] {
        for amount in [0.1, 0.2] {
            for tau in [0.01, 0.02, 0.04] {
                for &(w, c, a) in items.iter().filter(|x| !["b", "d", "g", "z", "m"].contains(&x.1)) {
                    jobs.push((glottal, amount, tau, w, c, a));
                }
            }
        }
    }
    let lines: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs.iter().map(|&(glottal, amount, tau, w, c, a)| sc.spawn(move || {
            let mut cells = vec![(0.0, 0.0, 0usize); WINDOWS.len()];
            let mut delay = 0.0;
            for o in 0..10 {
                let off = o as f64 * 0.001;
                let (base, _, _) = measure_with(w, a, Src::Frozen, off, None);
                let (st, d, inp) = measure_with(w, a, Src::Frozen, off, Some(Stiff { amount, tau, glottal, early: None }));
                delay += d / 10.0;
                for k in 0..WINDOWS.len() {
                    if let (Some(x), Some(y)) = (st[k], base[k]) {
                        let df = x - y;
                        cells[k].0 += df;
                        cells[k].1 += df - F0 * inp[k];
                        cells[k].2 += 1;
                    }
                }
            }
            let cell = |k: usize| if cells[k].2 == 0 { "-".to_string() } else { format!("{:+.1}|{:+.1}", cells[k].0 / cells[k].2 as f64, cells[k].1 / cells[k].2 as f64) };
            format!("{}\t{amount}\t{:.0}\t{w}\t{c}\t{delay:.1}\t{}\t{}\t{}\t{}\t{}", if glottal { "glottal" } else { "oral" }, tau * 1000.0, cell(0), cell(1), cell(2), cell(3), cell(4))
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for l in lines {
        println!("{l}");
    }
}

/// The human microprosody after a voiceless obstruent, averaged over the probe's windows:
/// 11.1 Hz · exp(−95.9 t) (Birkholz & Zhang 2020, Eq. 4; German, Praat F0 in 10 ms steps over the
/// vowel's first 80 ms, so the time constant is about the sampling step: a loose target).
fn human_window(a: f64, b: f64) -> f64 {
    11.1 * ((-95.9 * a).exp() - (-95.9 * b).exp()) / (95.9 * (b - a))
}

/// Stiffening timed to beat the slow first cycle. The folds (loop frozen) are stiffened by
/// `amount` of their tension factor from 30 ms before the oral closure, held `hold` past the
/// release (the voicing onset comes 34-73 ms after it), then let go over `tau`. Each cell is the
/// F0 change against the 100 ms reference before the consonant, less the pulse's change in the
/// same window (the plan's own F0 movement, -6 to -7 Hz in the vowel): the outcome to compare with
/// the human curve, which is measured against the vowel's own course. Per row: rmse against the human window means, and whether the
/// shape is the human one (largest in the first window, at least +5 Hz there, below +2 Hz by
/// 30-40 ms). Mean over ten start offsets.
fn early_table(items: &[(&str, &str, Option<bool>)]) {
    let rev = std::process::Command::new("git").args(["describe", "--always", "--dirty"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    println!("# microprosody_probe --early at {}; cubic {:?}", rev.trim(), CUBIC.get());
    println!("# folds, loop frozen; stiffening from closure - 30 ms, held past the release, let go over tau; cells: F0 change vs the reference, less the pulse's (Hz)");
    print!("# human (Birkholz & Zhang 2020, Eq. 4) per window:");
    for &(a, b) in WINDOWS.iter() {
        print!(" {:+.1}", human_window(a, b));
    }
    println!();
    println!("amount\thold_ms\ttau_ms\titem\tC\tdelay_ms\t0-10ms\t10-20ms\t20-30ms\t30-40ms\t40-60ms\trmse_vs_human\thuman_shape");
    let chosen: Vec<&(&str, &str, Option<bool>)> = items.iter().filter(|x| ["p˭", "t˭", "k˭", "kʰ", "s", "h"].contains(&x.1)).collect();
    // the pulse's change per window, per item and offset (the plan's own F0 movement)
    let pulse: std::collections::HashMap<(String, usize), Vec<Option<f64>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = chosen
            .iter()
            .flat_map(|&&(w, c, a)| (0..10).map(move |o| (w, c, a, o)))
            .map(|(w, c, a, o)| sc.spawn(move || ((format!("{w}{c}"), o), measure_with(w, a, Src::Pulse, o as f64 * 0.001, None).0)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let pulse = &pulse;
    let mut jobs = vec![];
    for amount in [0.0, 0.1, 0.2, 0.3, 0.4] {
        for hold in [0.0, 0.02, 0.04, 0.06, 0.08] {
            for tau in [0.005, 0.01, 0.02, 0.04] {
                if amount == 0.0 && (hold > 0.0 || tau > 0.005) {
                    continue;
                }
                for &&(w, c, a) in &chosen {
                    jobs.push((amount, hold, tau, w, c, a));
                }
            }
        }
    }
    let lines: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&(amount, hold, tau, w, c, a)| {
                sc.spawn(move || {
                    let mut cells = vec![(0.0, 0usize); WINDOWS.len()];
                    let mut delay = 0.0;
                    for o in 0..10 {
                        let st = (amount > 0.0).then_some(Stiff { amount, tau, glottal: false, early: Some(hold) });
                        let (r, d, _) = measure_with(w, a, Src::Frozen, o as f64 * 0.001, st);
                        delay += d / 10.0;
                        let pr = &pulse[&(format!("{w}{c}"), o)];
                        for k in 0..WINDOWS.len() {
                            if let (Some(x), Some(p)) = (r[k], pr[k]) {
                                cells[k].0 += x - p;
                                cells[k].1 += 1;
                            }
                        }
                    }
                    let m: Vec<Option<f64>> = cells.iter().map(|c| (c.1 > 0).then(|| c.0 / c.1 as f64)).collect();
                    let human: Vec<f64> = WINDOWS.iter().map(|&(a, b)| human_window(a, b)).collect();
                    let pairs: Vec<(f64, f64)> = m.iter().zip(&human).filter_map(|(x, h)| x.map(|x| (x, *h))).collect();
                    let rmse = (pairs.iter().map(|(x, h)| (x - h) * (x - h)).sum::<f64>() / pairs.len().max(1) as f64).sqrt();
                    let shape = match (m[0], m[3]) {
                        (Some(first), Some(late)) => first >= 5.0 && late < 2.0 && m.iter().skip(1).all(|x| x.is_none_or(|x| x < first)),
                        _ => false,
                    };
                    let cell = |k: usize| m[k].map_or("-".to_string(), |x| format!("{x:+.1}"));
                    format!(
                        "{amount}\t{:.0}\t{:.0}\t{w}\t{c}\t{delay:.1}\t{}\t{}\t{}\t{}\t{}\t{rmse:.1}\t{}",
                        hold * 1000.0,
                        tau * 1000.0,
                        cell(0),
                        cell(1),
                        cell(2),
                        cell(3),
                        cell(4),
                        if shape { "yes" } else { "no" }
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for l in lines {
        println!("{l}");
    }
}

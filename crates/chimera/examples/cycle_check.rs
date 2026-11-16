//! Validates the cycle reader the microprosody probe uses, on the pulse, whose cycles are known:
//! its phase wraps once per cycle, and it is asked a flat F0. The reader finds each cycle's peak of
//! the glottal area (the largest value within ±2.5 ms, refined by a parabola through it and its two
//! neighbours) and takes F0 from successive peaks; a cycle counts if its peak stands at least
//! 0.003 cm² above the lowest area between it and the previous peak.
//!
//! Pass (set before running): over every cycle from the first voiced one after the consonant to
//! 60 ms after it, and in the 60 ms before the consonant, the reader's F0 is within ±1 Hz of the
//! true F0 (from the phase wraps), and it misses no cycle. Run over aCa words, F0 110 and 160 Hz,
//! and ten start offsets (0-9 ms).
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::Intonation;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

const FS: f64 = 48000.0;

/// (time of the cycle's midpoint, F0) from successive peaks of `y`: each peak is the largest value
/// within ±`half` samples, standing at least `min_prominence` above the lowest value since the
/// previous peak, placed between samples by a parabola through it and its two neighbours. Meant for
/// a signal with no slow offset (the folds' mass displacement); an offset that ramps within a cycle
/// drags the peak (tried and rejected: subtracting the line through the troughs over-corrects where
/// the ramp curves).
pub fn cycles_from_peaks(y: &[f64], half: usize, min_prominence: f64) -> Vec<(f64, f64)> {
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

fn main() {
  for offset_free in [false, true] {
    let mut worst: f64 = 0.0;
    let mut missed = 0usize;
    let mut checked = 0usize;
    let mut bad: Vec<String> = vec![];
    let mut errs: Vec<(bool, f64, f64)> = vec![];
    for f0 in [110.0, 160.0] {
        for w in ["apa", "ata", "aka", "asa", "afa", "aha", "aba", "ada", "aza", "ama"] {
            for off in 0..10 {
                let mut input = tokenize_ipa(w);
                for s in input.iter_mut() {
                    s.stress = Some(0);
                }
                let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05 + off as f64 * 0.001, f0: Some(f0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
                let mut v = HumanVoice::with_defaults(FS);
                v.set_source(&SourcePatch { roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
                v.schedule(&plan.gestures);
                let n = (FS * (plan.end + 0.1)) as usize;
                let mut out = [0.0f32; 1];
                let (mut area, mut wraps, mut voiced) = (Vec::with_capacity(n), vec![], vec![]);
                let mut bare = Vec::with_capacity(n);
                let mut last_phase = 0.0;
                for i in 0..n {
                    v.render(&mut out);
                    area.push(v.glottis.area);
                    bare.push(v.glottis.area - v.glottis.static_abduction());
                    let ph = v.glottis.phase();
                    if ph < last_phase {
                        // the wrap between the two samples, by linear interpolation of the phase
                        let frac = (1.0 - last_phase) / (ph + 1.0 - last_phase);
                        wraps.push((i as f64 - 1.0 + frac) / FS);
                        voiced.push(v.glottis.voicing);
                    }
                    last_phase = ph;
                }
                let truth: Vec<(f64, f64)> = wraps.windows(2).zip(voiced.windows(2)).filter(|(_, v)| v[0] > 0.5 && v[1] > 0.5).map(|(w, _)| (0.5 * (w[0] + w[1]), 1.0 / (w[1] - w[0]))).collect();
                let got = cycles_from_peaks(if offset_free { &bare } else { &area }, (0.0025 * FS) as usize, 0.003);
                let c = &plan.segments[1];
                let in_window = |t: f64| (t > c.t0 - 0.06 && t < c.t0) || (t > c.t1 && t < c.t1 + 0.06);
                for &(t, f) in truth.iter().filter(|x| in_window(x.0)) {
                    checked += 1;
                    match got.iter().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).filter(|g| (g.0 - t).abs() < 0.5 / f) {
                        Some(g) => {
                            let e = (g.1 - f).abs();
                            errs.push((t >= c.t1, if t < c.t0 { (c.t0 - t) * 1000.0 } else { (t - c.t1) * 1000.0 }, g.1 - f));
                            worst = worst.max(e);
                            if e > 1.0 {
                                let rel = if t < c.t0 { t - c.t0 } else { t - c.t1 };
                                bad.push(format!("{w} f0 {f0} off {off}: {:+.1} ms from the consonant {}, error {:+.2} Hz", rel * 1000.0, if t < c.t0 { "(before)" } else { "(after)" }, g.1 - f));
                            }
                        }
                        None => missed += 1,
                    }
                }
            }
        }
    }
    println!("signal: {}", if offset_free { "area less its static opening (offset-free)" } else { "raw area (with the abduction offset)" });
    println!("cycles over 1 Hz: {}", bad.len());
    for b in bad.iter().take(6) {
        println!("  {b}");
    }
    for (after, name) in [(false, "before the consonant"), (true, "after it")] {
        for (lo, hi) in [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 60.0)] {
            let e: Vec<f64> = errs.iter().filter(|x| x.0 == after && x.1 >= lo && x.1 < hi).map(|x| x.2).collect();
            if !e.is_empty() {
                let worst = e.iter().cloned().fold(0.0, |m: f64, x| m.max(x.abs()));
                let mean = e.iter().sum::<f64>() / e.len() as f64;
                println!("  {name} {lo:.0}-{hi:.0} ms: {} cycles, mean error {mean:+.2} Hz, worst {worst:.2} Hz", e.len());
            }
        }
    }
    println!("cycles checked {checked}, missed {missed}, worst F0 error {worst:.2} Hz -> {}", if missed == 0 && worst <= 1.0 { "PASS" } else { "FAIL" });
  }
}

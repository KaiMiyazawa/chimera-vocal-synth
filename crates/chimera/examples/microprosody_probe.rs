//! Research probe (pillar 2): does the F0 of a vowel dip next to a voiced obstruent and rise after a
//! voiceless one (microprosody) when nothing in the plan asks for it?
//!
//! The plan holds F0 flat (no stress, `accent` 0, flat intonation), so any F0 movement around the
//! consonant comes from the source. Two sources: the parametric pulse (F0 is set directly, so it
//! should stay flat) and the two-mass folds (F0 comes out of the folds' mechanics under the
//! transglottal pressure the airway computes, with a slow loop trimming tension toward the asked
//! F0).
//!
//! F0 is read cycle by cycle from the glottal area (its oscillating part, upward zero crossings). For each word `aCa`
//! it reports, relative to the F0 at the middle of the second vowel (its last third): F0 at
//! 5, 10, 20, 30 and 40 ms after the second vowel's voicing onset (the first frame from the
//! consonant's release on that is periodic), and the lowest F0 in the last 40 ms of the first vowel
//! (the dip into a voiced obstruent).
//! Run: `cargo run --release -q -p chimera --example microprosody_probe`
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::Intonation;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const F0: f64 = 110.0;

/// F0 cycle by cycle from the glottal area; each cycle's F0 is assigned to its midpoint.
fn track(ipa: &str, model: GlottisModel) -> (Vec<(f64, f64)>, chimera::phonetics::Plan) {
    let mut input = tokenize_ipa(ipa);
    for s in input.iter_mut() {
        s.stress = Some(0);
    }
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(model), ..Default::default() });
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = [0.0f32; 1];
    let mut area = Vec::with_capacity(n);
    for _ in 0..n {
        v.render(&mut out);
        area.push(if model == GlottisModel::TwoMass { v.folds.area } else { v.glottis.area });
    }
    // the oscillating part: the area less its 5 ms moving average; a cycle runs from one upward
    // zero crossing to the next (2.5–15 ms apart) and counts if it swings by at least 0.01 cm²
    let half = (0.0025 * FS) as usize;
    let mut sum = 0.0;
    let mut acc = vec![0.0; n + 1];
    for i in 0..n {
        sum += area[i];
        acc[i + 1] = sum;
    }
    let ac: Vec<f64> = (0..n).map(|i| {
        let (lo, hi) = (i.saturating_sub(half), (i + half + 1).min(n));
        area[i] - (acc[hi] - acc[lo]) / (hi - lo) as f64
    }).collect();
    let ups: Vec<usize> = (1..n).filter(|&i| ac[i - 1] < 0.0 && ac[i] >= 0.0).collect();
    let cycles = ups
        .windows(2)
        .filter_map(|w| {
            let per = (w[1] - w[0]) as f64 / FS;
            let seg = &ac[w[0]..w[1]];
            let swing = seg.iter().cloned().fold(f64::MIN, f64::max) - seg.iter().cloned().fold(f64::MAX, f64::min);
            (per >= 0.0025 && per <= 0.015 && swing >= 0.01).then(|| (0.5 * (w[0] + w[1]) as f64 / FS, 1.0 / per))
        })
        .collect();
    (cycles, plan)
}

fn main() {
    println!("model\tword\tC\tdip_v1\tfirst\t+5ms\t+10ms\t+20ms\t+30ms\t+40ms\tref_hz\tonset_ms");
    for (model, name) in [(GlottisModel::Parametric, "pulse"), (GlottisModel::TwoMass, "folds")] {
        for (w, c) in [("apa", "p"), ("aba", "b"), ("ata", "t"), ("ada", "d"), ("aka", "k"), ("aga", "g"), ("asa", "s"), ("aza", "z"), ("afa", "f"), ("ava", "v"), ("ama", "m"), ("aha", "h")] {
            let (f, plan) = track(w, model);
            let (v1, cs, v2) = (&plan.segments[0], &plan.segments[1], &plan.segments[2]);
            // the cycle whose midpoint is nearest, if within 6 ms
            let at = |t: f64| f.iter().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).filter(|x| (x.0 - t).abs() < 0.006).map(|x| x.1).unwrap_or(0.0);
            let tail: Vec<f64> = f.iter().filter(|x| x.0 > v2.t1 - (v2.t1 - v2.t0) / 3.0 && x.0 < v2.t1 && x.1 > 0.0).map(|x| x.1).collect();
            let reference = if tail.is_empty() { f64::NAN } else { tail.iter().sum::<f64>() / tail.len() as f64 };
            // the first cycle after the consonant's release (its segment end, less 10 ms)
            let onset = f.iter().find(|x| x.0 >= cs.t1 - 0.01).map(|x| x.0).unwrap_or(f64::NAN);
            let dip = f.iter().filter(|x| x.0 > v1.t1 - 0.04 && x.0 <= v1.t1 + 0.01 && x.1 > 0.0).map(|x| x.1).fold(f64::NAN, f64::min);
            let rel = |hz: f64| if hz > 0.0 { format!("{:+.1}", hz - reference) } else { "-".to_string() };
            let first = f.iter().find(|x| x.0 >= onset).map(|x| x.1).unwrap_or(0.0);
            println!("{name}\t{w}\t{c}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.0}", rel(dip), rel(first), rel(at(onset + 0.005)), rel(at(onset + 0.01)), rel(at(onset + 0.02)), rel(at(onset + 0.03)), rel(at(onset + 0.04)), reference, (onset - cs.t1) * 1000.0);
        }
    }
}

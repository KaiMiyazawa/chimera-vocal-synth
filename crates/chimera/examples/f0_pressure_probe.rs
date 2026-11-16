//! Research probe (pillar 2): how much does F0 move with the pressure across the folds?
//!
//! A held /a/ at a flat asked F0; the lung pressure steps from 8 cmH₂O to P at 0.8 s. F0 is read
//! cycle by cycle from the glottal area (oscillating part, upward zero crossings), averaged over
//! windows: before the step (0.6–0.8 s), just after it (+5 to +30 ms, before a slow pitch loop can
//! answer), and settled (+300 to +500 ms). The fast change per cmH₂O of lung pressure is the folds'
//! own sensitivity; the settled change is what the pitch loop leaves. Speech measurements and
//! HLsyn put the sensitivity near 3 Hz per cmH₂O of transglottal pressure (Hanson & Stevens 2002,
//! Eq. 33).
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::Intonation;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;

fn cycles(area: &[f64]) -> Vec<(f64, f64)> {
    let n = area.len();
    let half = (0.0025 * FS) as usize;
    let mut acc = vec![0.0; n + 1];
    for i in 0..n {
        acc[i + 1] = acc[i] + area[i];
    }
    let ac: Vec<f64> = (0..n).map(|i| {
        let (lo, hi) = (i.saturating_sub(half), (i + half + 1).min(n));
        area[i] - (acc[hi] - acc[lo]) / (hi - lo) as f64
    }).collect();
    let ups: Vec<usize> = (1..n).filter(|&i| ac[i - 1] < 0.0 && ac[i] >= 0.0).collect();
    ups.windows(2).filter_map(|w| {
        let per = (w[1] - w[0]) as f64 / FS;
        let s = &ac[w[0]..w[1]];
        let swing = s.iter().cloned().fold(f64::MIN, f64::max) - s.iter().cloned().fold(f64::MAX, f64::min);
        (per >= 0.0025 && per <= 0.015 && swing >= 0.005).then(|| (0.5 * (w[0] + w[1]) as f64 / FS, 1.0 / per))
    }).collect()
}

fn main() {
    println!("model\tf0_asked\tp_after\tf0_before\tfast_dHz\tsettled_dHz\tfast_Hz_per_cmH2O\tsettled_Hz_per_cmH2O");
    for (model, name) in [(GlottisModel::Parametric, "pulse"), (GlottisModel::TwoMass, "folds")] {
        for f0 in [110.0, 160.0] {
            for p_after in [4.0, 6.0, 10.0, 12.0, 16.0] {
                let mut input = tokenize_ipa("a");
                input[0].dur = Some(1.6);
                input[0].stress = Some(0);
                let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(f0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
                let mut v = HumanVoice::with_defaults(FS);
                v.set_source(&SourcePatch { glottis: Some(model), pressure: Some(8.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
                v.schedule(&plan.gestures);
                let n = (FS * 1.5) as usize;
                let step = (FS * 0.8) as usize;
                let mut out = [0.0f32; 1];
                let mut area = Vec::with_capacity(n);
                for i in 0..n {
                    if i == step {
                        v.set_source(&SourcePatch { pressure: Some(p_after), ..Default::default() });
                    }
                    v.render(&mut out);
                    area.push(if model == GlottisModel::TwoMass { v.folds.area } else { v.glottis.area });
                }
                let c = cycles(&area);
                let mean = |a: f64, b: f64| {
                    let s: Vec<f64> = c.iter().filter(|x| x.0 >= a && x.0 < b).map(|x| x.1).collect();
                    if s.is_empty() { f64::NAN } else { s.iter().sum::<f64>() / s.len() as f64 }
                };
                let (before, fast, settled) = (mean(0.6, 0.8), mean(0.805, 0.83), mean(1.1, 1.3));
                let dp = p_after - 8.0;
                println!("{name}\t{f0}\t{p_after}\t{before:.1}\t{:+.1}\t{:+.1}\t{:+.2}\t{:+.2}", fast - before, settled - before, (fast - before) / dp, (settled - before) / dp);
            }
        }
    }
}

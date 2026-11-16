//! Research probe (pillar 2): how much does F0 move with the pressure across the folds?
//!
//! A held /a/ at a flat asked F0 (roughness, drift and vibrato 0), lung pressure 8 cmH₂O. For the
//! folds the trim loop settles and is frozen at 0.6 s, so what follows is their mechanics alone. At
//! 0.8 s the lung pressure steps to P. F0 and the transglottal pressure (last tracheal node minus
//! first pharyngeal node) are averaged over the cycles of 0.65–0.8 s (before) and 1.0–1.3 s
//! (after). The sensitivity is the F0 change over the transglottal pressure change actually
//! measured, not over the lung-pressure step. Cycles of the folds are read from successive peaks of
//! the upper-mass displacement (cycle_check's reader); the pulse, whose F0 is given, is read from
//! its own phase wrapping and is the control (0 by construction).
//! HLsyn's default is 3 Hz per cmH₂O (Hanson & Stevens 2002, Eq. 33); a measured human value is
//! still to be read (Titze 1989).
//! With `--no-lower-push` the folds run without the subglottal push on the shut lower mass (a
//! diagnosis only, see onset_probe).
//! Run: `cargo run --release -q -p chimera --example f0_pressure_probe [-- --no-lower-push]`
use chimera::phonetics::Intonation;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::physics::CM_H2O;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;

/// Peak times of `y` (cycle_check's reader).
fn peaks(y: &[f64], half: usize, min_prominence: f64) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![];
    let mut last: Option<usize> = None;
    for i in half..n.saturating_sub(half) {
        let w = &y[i - half..=i + half];
        if w.iter().all(|&v| v <= y[i]) && w.iter().filter(|&&v| v == y[i]).count() == 1 {
            let lo = last.map_or(&y[i.saturating_sub(4 * half)..i], |j| &y[j..i]).iter().cloned().fold(f64::MAX, f64::min);
            if y[i] - lo >= min_prominence {
                let (a, b, c) = (y[i - 1], y[i], y[i + 1]);
                let den = a - 2.0 * b + c;
                let off = if den.abs() > 1e-18 { 0.5 * (a - c) / den } else { 0.0 };
                out.push((i as f64 + off) / FS);
                last = Some(i);
            }
        }
    }
    out
}

fn run(model: GlottisModel, f0: f64, p_after: f64, lower_push: bool) -> (f64, f64, f64, f64) {
    let mut input = tokenize_ipa("ɑ");
    input[0].dur = Some(1.6);
    input[0].stress = Some(0);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(f0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(model), pressure: Some(8.0), roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
    v.folds.closed_push_lower = lower_push;
    v.schedule(&plan.gestures);
    let n = (FS * 1.35) as usize;
    let (freeze, step) = ((FS * 0.6) as usize, (FS * 0.8) as usize);
    let (sub, supra) = (v.airway.trachea.last_node(), v.airway.pharynx.first_node());
    let mut out = [0.0f32; 1];
    let (mut x, mut dp) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        if i == freeze {
            v.folds.trim_loop = false;
        }
        if i == step {
            v.set_source(&SourcePatch { pressure: Some(p_after), ..Default::default() });
        }
        v.render(&mut out);
        x.push(if model == GlottisModel::TwoMass { v.folds.displacement() } else { v.glottis.phase() });
        dp.push((v.airway.net.p[sub] - v.airway.net.p[supra]) / CM_H2O);
    }
    let t: Vec<f64> = if model == GlottisModel::TwoMass {
        peaks(&x, (0.0025 * FS) as usize, 0.002)
    } else {
        // the pulse's phase wraps from 1 to 0 once per cycle
        (1..n).filter(|&i| x[i] < x[i - 1] - 0.5).map(|i| i as f64 / FS).collect()
    };
    let f = |a: f64, b: f64| {
        let s: Vec<f64> = t.windows(2).filter(|w| w[0] >= a && w[1] < b).map(|w| 1.0 / (w[1] - w[0])).collect();
        s.iter().sum::<f64>() / s.len().max(1) as f64
    };
    let p = |a: f64, b: f64| dp[(a * FS) as usize..(b * FS) as usize].iter().sum::<f64>() / ((b - a) * FS);
    (f(0.65, 0.8), f(1.0, 1.3), p(0.65, 0.8), p(1.0, 1.3))
}

fn main() {
    let lower_push = !std::env::args().any(|a| a == "--no-lower-push");
    println!("# loop frozen at 0.6 s; lung pressure 8 -> P cmH2O at 0.8 s; before 0.65-0.8 s, after 1.0-1.3 s; lower push {}", if lower_push { "on" } else { "off" });
    println!("model\tf0_asked\tp_lung_after\tf0_before\tf0_after\tdF0\tptg_before\tptg_after\tdPtg\tHz_per_cmH2O_transglottal");
    let mut jobs = vec![];
    for (model, name) in [(GlottisModel::Parametric, "pulse"), (GlottisModel::TwoMass, "folds")] {
        for f0 in [110.0, 160.0] {
            for p_after in [4.0, 6.0, 10.0, 12.0, 16.0] {
                jobs.push((model, name, f0, p_after));
            }
        }
    }
    let lines: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&(m, name, f0, pa)| {
                sc.spawn(move || {
                    let (fb, fa, pb, pa2) = run(m, f0, pa, lower_push);
                    format!("{name}\t{f0}\t{pa}\t{fb:.1}\t{fa:.1}\t{:+.2}\t{pb:.2}\t{pa2:.2}\t{:+.2}\t{:+.3}", fa - fb, pa2 - pb, (fa - fb) / (pa2 - pb))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for l in lines {
        println!("{l}");
    }
}

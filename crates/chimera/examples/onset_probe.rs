//! Research probe (pillar 2): why are the two-mass folds' first cycles after a pause slow?
//!
//! A held /a/ at 110 Hz (flat; roughness, drift and vibrato 0). The trim loop settles for 0.8 s and
//! is then frozen, so the folds keep that operating point. At 1.0 s voicing is interrupted for
//! 60 ms in one of these ways, and resumes:
//! - `abduct A`: an abduction gesture of A cm² (as a voiceless consonant's), released after 60 ms;
//! - `breath`: a stiff breath-0 gesture (the score's breath is a weighted mean) and back;
//! - `stop`: a glottal closure (the folds held shut) and back;
//! - `hold A`: abduction A held from 1.0 to 1.3 s while the folds keep vibrating (A small enough):
//!   reports the steady F0, open quotient and pressure under that opening (cycles 1.15–1.3 s).
//!
//! Cycles are read from successive peaks of the upper-mass displacement (cycle_check's reader). For
//! the first eight cycles after voicing resumes it reports each cycle's F0 against the settled F0
//! (the cycles 35–130 ms before the interruption, clear of its onset ramp), its swing (the
//! displacement's peak-to-peak, against the settled swing), the mean transglottal pressure (against
//! the settled one), the abduction above the settled value, and the folds' open quotient. The
//! settled F0 is printed beside the lower natural frequency of the masses and springs alone at the
//! settled tension, to show how far the flow and the closures lift F0 above it.
//!
//! The folds' mechanics can be taken apart through research inputs, one set per argument
//! `kc:zc:push` (default `3:1:both`): the contact stiffness, the fraction of the contact damping,
//! and which shut masses the subglottal pressure pushes (both, lower, upper, none).
//! Run: `cargo run --release -q -p chimera --example onset_probe -- [kc:zc:push ...]`
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::phonetics::Intonation;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::physics::CM_H2O;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const T_CUT: f64 = 1.0;
const GAP: f64 = 0.06;
/// lower natural frequency (Hz) of the folds' two masses and springs alone (no flow, no contact)
/// at Q = 1, from the engine's constants (M1 M2 K1 K2 KC); it scales with Q
const LINEAR_HZ_AT_Q1: f64 = 120.42;

/// Peaks of `y` (cycle_check's reader): indices, refined times.
fn peaks(y: &[f64], half: usize, min_prominence: f64) -> Vec<(usize, f64)> {
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
                out.push((i, (i as f64 + off) / FS));
                last = Some(i);
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
enum Cut {
    Abduct(f64),
    Breath,
    Stop,
    Hold(f64),
    /// abduction held with the breath set to this value (the score's own is 1) over the same span
    HoldP(f64, f64),
}

fn run(cut: Cut, kc: f64, zc: f64, push: (bool, bool)) -> String {
    let mut input = tokenize_ipa("ɑ");
    input[0].dur = Some(1.6);
    input[0].stress = Some(0);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
    v.folds.contact_stiffness = kc;
    v.folds.contact_damping = zc;
    (v.folds.closed_push_lower, v.folds.closed_push_upper) = push;
    v.schedule(&plan.gestures);
    match cut {
        Cut::Abduct(a) => v.schedule(&[Gesture::new(T_CUT, T_CUT + GAP, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a)).label("probe")]),
        Cut::Stop => v.schedule(&[Gesture::new(T_CUT, T_CUT + GAP, 0.015, 0.015, 8.0, Targets::default().with(ControlKey::CLOSURE, 1.0)).label("probe")]),
        Cut::Breath => v.schedule(&[Gesture::new(T_CUT, T_CUT + GAP, 0.02, 0.02, 60.0, Targets::default().with(ControlKey::BREATH, 0.0)).label("probe")]),
        Cut::HoldP(a, b) => v.schedule(&[
            Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a)).label("probe"),
            Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 60.0, Targets::default().with(ControlKey::BREATH, b)).label("probe"),
        ]),
        Cut::Hold(a) => v.schedule(&[Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a)).label("probe")]),
    }
    let n = (FS * 1.4) as usize;
    let freeze = (0.8 * FS) as usize;
    let mut out = [0.0f32; 1];
    let mut q_ref = 0.0;
    let (mut x, mut dp, mut ab, mut oq) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let (sub, supra) = (v.airway.trachea.last_node(), v.airway.pharynx.first_node());
    for i in 0..n {
        if i == freeze {
            v.folds.trim_loop = false;
        }
        v.render(&mut out);
        x.push(v.folds.displacement());
        dp.push((v.airway.net.p[sub] - v.airway.net.p[supra]) / CM_H2O);
        ab.push(v.glottis.params.abduction);
        if i == (0.9 * FS) as usize {
            q_ref = v.folds.tension();
        }
        oq.push(v.folds.open_quotient);
    }
    let pk = peaks(&x, (0.0025 * FS) as usize, 0.002);
    // cycles: (mid time, F0, swing, mean transglottal pressure, mean abduction, open quotient)
    type C = (f64, f64, f64, f64, f64, f64);
    let cyc: Vec<C> = pk
        .windows(2)
        .filter(|w| w[1].1 - w[0].1 < 0.015)
        .map(|w| {
            let (i, j) = (w[0].0, w[1].0);
            let s = &x[i..j];
            let swing = s.iter().cloned().fold(f64::MIN, f64::max) - s.iter().cloned().fold(f64::MAX, f64::min);
            let mean = |v: &[f64]| v[i..j].iter().sum::<f64>() / (j - i) as f64;
            // the folds' open quotient is updated when their own cycle detector fires: read it a
            // little after this cycle's end
            let q = oq[(j + (j - i) / 2).min(oq.len() - 1)];
            (0.5 * (w[0].1 + w[1].1), 1.0 / (w[1].1 - w[0].1), swing, mean(&dp), mean(&ab), q)
        })
        .collect();
    let avg = |cs: &[&C], f: fn(&C) -> f64| cs.iter().map(|c| f(c)).sum::<f64>() / cs.len().max(1) as f64;
    let settled: Vec<&C> = cyc.iter().filter(|c| c.0 > T_CUT - 0.13 && c.0 < T_CUT - 0.035).collect();
    let (f_ref, sw_ref, p_ref, ab_ref, oq_ref) = (avg(&settled, |c| c.1), avg(&settled, |c| c.2), avg(&settled, |c| c.3), avg(&settled, |c| c.4), avg(&settled, |c| c.5));
    let held_name = match cut {
        Cut::Hold(a) => Some(format!("hold {a:.2}")),
        Cut::HoldP(a, b) => Some(format!("hold {a:.2} breath {b:.1}")),
        _ => None,
    };
    if let Some(hn) = held_name {
        let held: Vec<&C> = cyc.iter().filter(|c| c.0 > T_CUT + 0.15 && c.0 < T_CUT + 0.3).collect();
        return format!(
            "kc{kc} zc{zc} push-{}\t{hn}\t{f_ref:.1} (linear {:.1})\tn{}\t{:+.1}Hz s{:.2} p{:.2} dab{:.3} oq {:.2}->{:.2}",
            pn(push),
            LINEAR_HZ_AT_Q1 * q_ref,
            held.len(),
            avg(&held, |c| c.1) - f_ref,
            avg(&held, |c| c.2) / sw_ref,
            avg(&held, |c| c.3) / p_ref,
            avg(&held, |c| c.4) - ab_ref,
            oq_ref,
            avg(&held, |c| c.5)
        );
    }
    let after: Vec<&C> = cyc.iter().filter(|c| c.0 > T_CUT + GAP * 0.5).take(8).collect();
    let name = match cut {
        Cut::Abduct(a) => format!("abduct {a:.1}"),
        Cut::Breath => "breath".into(),
        Cut::Stop => "stop".into(),
        Cut::Hold(_) | Cut::HoldP(..) => unreachable!(),
    };
    let mut line = format!("kc{kc} zc{zc} push-{}\t{name}\t{f_ref:.1} (linear {:.1})\t{:.1}", pn(push), LINEAR_HZ_AT_Q1 * q_ref, after.first().map_or(f64::NAN, |c| (c.0 - T_CUT - GAP) * 1000.0));
    for c in &after {
        line += &format!("\t{:+.1}Hz s{:.2} p{:.2} dab{:.3} oq{:.2}", c.1 - f_ref, c.2 / sw_ref, c.3 / p_ref, c.4 - ab_ref, c.5);
    }
    line
}

/// which shut masses are pushed by the subglottal pressure
fn pn(p: (bool, bool)) -> &'static str {
    match p {
        (true, true) => "both",
        (true, false) => "lower",
        (false, true) => "upper",
        (false, false) => "none",
    }
}

fn main() {
    println!("# folds, loop frozen at 0.8 s, 110 Hz, voicing cut for {:.0} ms at {T_CUT} s; per cycle after it resumes: dF0 vs settled, swing / settled (s), transglottal pressure / settled (p), abduction above settled cm² (dab), open quotient (oq)", GAP * 1000.0);
    println!("kc\tcut\tsettled_hz\tfirst_cycle_ms_after\tcycle1\tcycle2\tcycle3\tcycle4\tcycle5\tcycle6\tcycle7\tcycle8");
    // (contact stiffness, fraction of the contact damping, which shut masses are pushed) sets to run, as
    // kc:zc:push (default 3:1:both)
    let sets: Vec<(f64, f64, (bool, bool))> = std::env::args()
        .skip(1)
        .filter_map(|a| {
            let f: Vec<&str> = a.split(':').collect();
            Some((f.first()?.parse().ok()?, f.get(1).map_or(Some(1.0), |z| z.parse().ok())?, match f.get(2).copied().unwrap_or("both") { "both" => (true, true), "lower" => (true, false), "upper" => (false, true), "none" => (false, false), _ => return None }))
        })
        .collect();
    let sets = if sets.is_empty() { vec![(3.0, 1.0, (true, true))] } else { sets };
    let cuts = [Cut::Abduct(0.1), Cut::Abduct(0.2), Cut::Abduct(0.3), Cut::Breath, Cut::Stop, Cut::Hold(0.0), Cut::Hold(0.02), Cut::Hold(0.04), Cut::Hold(0.06), Cut::Hold(0.1), Cut::HoldP(0.0, 0.7), Cut::HoldP(0.0, 1.3), Cut::HoldP(0.04, 0.7), Cut::HoldP(0.04, 1.3), Cut::HoldP(0.1, 0.7), Cut::HoldP(0.1, 1.3)];
    // one set at a time, its cuts in parallel
    let mut lines = vec![];
    for &(k, z, p) in &sets {
        lines.extend(std::thread::scope(|sc| {
            let hs: Vec<_> = cuts.iter().map(|&c| sc.spawn(move || run(c, k, z, p))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<String>>()
        }));
    }
    for l in lines {
        println!("{l}");
    }
}

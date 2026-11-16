//! Research probe (pillar 2): why are the two-mass folds' first cycles after a pause slow?
//!
//! A held /a/ at 110 Hz (flat; roughness, drift and vibrato 0). The trim loop settles for 0.8 s and
//! is then frozen, so the folds keep that operating point. At 1.0 s voicing is interrupted in one
//! of these ways (the gesture's 60 ms plus its ramps: from 0.98 s to `cut end`):
//! - `abduct A`: an abduction gesture of A cm² (as a voiceless consonant's), with no oral closure;
//! - `breath`: a stiff breath-0 gesture (the score's breath is a weighted mean);
//! - `stop`: a glottal closure (the folds held shut);
//! - `hold A` (`breath B`): abduction A (and breath B) held from 1.0 to 1.3 s while the folds may
//!   keep vibrating: the steady state under that opening (cycles 1.15–1.3 s).
//!
//! Cycles are read from successive peaks of the upper-mass displacement (`peaks`). Per cycle: F0
//! against the settled F0 (cycles 0.87–0.965 s), swing (displacement peak-to-peak) and mean
//! transglottal pressure against the settled ones, abduction above the settled one (`dab`, cm²),
//! the open quotient read from the folds' own channel area (share of the cycle with the narrower
//! channel above 1e-4 cm²; `oq`), whether the cycle closes at all (`c`/`-`), and the closed-phase
//! push on the lower mass integrated over the cycle against the settled one (`imp`). Cycles whose
//! middle falls inside the interruption are counted (`during`); the eight reported are those after
//! `cut end`, each marked `r` once the abduction is back within 0.005 cm² and the pressure within
//! 5% of the settled values (only those are "voicing resumed"), `.` before.
//!
//! The folds' mechanics can be taken apart, one set per argument `kc:zc:push` (default
//! `3:1:both`): the contact stiffness, the fraction of the contact damping, and which shut masses
//! the subglottal pressure pushes (both, lower, upper, none). By default a set is in force from the
//! start, so the loop settles each set to 110 Hz at its own tension. With `--q-fixed` every run
//! settles with the default mechanics and the set is switched in when the loop freezes (0.8 s):
//! the tension is the same for all sets, and `pre_hz` (0.7–0.8 s) against `settled_hz` is the set's
//! own effect on F0 at that tension. With `--push-off A,B` the push on the shut lower mass is left
//! out only from A to B s (the first cycles after the interruption, everything before identical).
//! Run: `cargo run --release -q -p chimera --example onset_probe -- [--q-fixed] [--push-off A,B] [kc:zc:push ...]`
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::phonetics::Intonation;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::physics::CM_H2O;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

#[path = "common/peaks.rs"]
mod peaks;
use peaks::peaks;

const FS: f64 = 48000.0;
const T_FREEZE: f64 = 0.8;
const T_CUT: f64 = 1.0;
const GAP: f64 = 0.06;
/// lower natural frequency (Hz) of the folds' two masses and springs alone (no flow, no contact)
/// at Q = 1, from the engine's constants (M1 M2 K1 K2 KC); it scales with Q
const LINEAR_HZ_AT_Q1: f64 = 120.42;

#[derive(Clone, Copy)]
enum Cut {
    Abduct(f64),
    Breath,
    Stop,
    Hold(f64),
    /// abduction held with the breath set to this value (the score's own is 1) over the same span
    HoldP(f64, f64),
}

#[derive(Clone, Copy)]
struct Set {
    kc: f64,
    zc: f64,
    push: (bool, bool),
}

#[derive(Clone, Copy)]
struct Opts {
    q_fixed: bool,
    push_off: Option<(f64, f64)>,
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

fn apply(v: &mut HumanVoice, s: Set) {
    v.folds.contact_stiffness = s.kc;
    v.folds.contact_damping = s.zc;
    (v.folds.closed_push_lower, v.folds.closed_push_upper) = s.push;
}

/// one cycle: (mid time, F0, swing, transglottal pressure, abduction, open quotient, closes, push impulse)
type C = (f64, f64, f64, f64, f64, f64, bool, f64);

fn run(cut: Cut, set: Set, o: Opts) -> String {
    let mut input = tokenize_ipa("ɑ");
    input[0].dur = Some(1.6);
    input[0].stress = Some(0);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
    if !o.q_fixed {
        apply(&mut v, set);
    }
    v.schedule(&plan.gestures);
    let probe = match cut {
        Cut::Abduct(a) => vec![Gesture::new(T_CUT, T_CUT + GAP, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a))],
        Cut::Stop => vec![Gesture::new(T_CUT, T_CUT + GAP, 0.015, 0.015, 8.0, Targets::default().with(ControlKey::CLOSURE, 1.0))],
        Cut::Breath => vec![Gesture::new(T_CUT, T_CUT + GAP, 0.02, 0.02, 60.0, Targets::default().with(ControlKey::BREATH, 0.0))],
        Cut::HoldP(a, b) => vec![
            Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a)),
            Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 60.0, Targets::default().with(ControlKey::BREATH, b)),
        ],
        Cut::Hold(a) => vec![Gesture::new(T_CUT, T_CUT + 0.3, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, a))],
    };
    let cut_end = probe.iter().map(|g| g.t1 + g.ramp_out).fold(0.0, f64::max);
    let cut_start = probe.iter().map(|g| g.t0 - g.ramp_in).fold(f64::MAX, f64::min);
    v.schedule(&probe.iter().map(|g| g.clone().label("probe")).collect::<Vec<_>>());
    let n = (FS * 1.4) as usize;
    let freeze = (T_FREEZE * FS) as usize;
    let mut out = [0.0f32; 1];
    let mut q_ref = 0.0;
    let [mut x, mut dp, mut ab, mut ch, mut push] = std::array::from_fn(|_| Vec::with_capacity(n));
    let (sub, supra) = (v.airway.trachea.last_node(), v.airway.pharynx.first_node());
    for i in 0..n {
        let t = i as f64 / FS;
        if i == freeze {
            v.folds.trim_loop = false;
            if o.q_fixed {
                apply(&mut v, set);
            }
        }
        if let Some((a, b)) = o.push_off {
            v.folds.closed_push_lower = if t >= a && t < b { false } else { set.push.0 };
        }
        v.render(&mut out);
        x.push(v.folds.displacement());
        dp.push((v.airway.net.p[sub] - v.airway.net.p[supra]) / CM_H2O);
        ab.push(v.glottis.params.abduction);
        ch.push(v.folds.channel_area());
        push.push(v.folds.closed_push_force());
        if i == (0.9 * FS) as usize {
            q_ref = v.folds.tension();
        }
    }
    let pk = peaks(&x, FS, (0.0025 * FS) as usize, 0.002);
    let cyc: Vec<C> = pk
        .windows(2)
        // a gap longer than 30 ms is the silence, not a cycle
        .filter(|w| w[1].1 - w[0].1 < 0.03)
        .map(|w| {
            let (i, j) = (w[0].0, w[1].0);
            let s = &x[i..j];
            let swing = s.iter().cloned().fold(f64::MIN, f64::max) - s.iter().cloned().fold(f64::MAX, f64::min);
            let mean = |v: &[f64]| v[i..j].iter().sum::<f64>() / (j - i) as f64;
            let open = ch[i..j].iter().filter(|&&a| a > 1e-4).count() as f64 / (j - i) as f64;
            let closes = ch[i..j].iter().any(|&a| a <= 1e-4);
            let imp = push[i..j].iter().sum::<f64>() / FS;
            (0.5 * (w[0].1 + w[1].1), 1.0 / (w[1].1 - w[0].1), swing, mean(&dp), mean(&ab), open, closes, imp)
        })
        .collect();
    let avg = |cs: &[&C], f: fn(&C) -> f64| if cs.is_empty() { f64::NAN } else { cs.iter().map(|c| f(c)).sum::<f64>() / cs.len() as f64 };
    let pre: Vec<&C> = cyc.iter().filter(|c| c.0 > 0.7 && c.0 < T_FREEZE - 0.005).collect();
    let settled: Vec<&C> = cyc.iter().filter(|c| c.0 > 0.87 && c.0 < 0.965).collect();
    let f_pre = avg(&pre, |c| c.1);
    let (f_ref, sw_ref, p_ref, ab_ref, oq_ref, imp_ref) =
        (avg(&settled, |c| c.1), avg(&settled, |c| c.2), avg(&settled, |c| c.3), avg(&settled, |c| c.4), avg(&settled, |c| c.5), avg(&settled, |c| c.7));
    let head = format!("kc{} zc{} push-{}\t", set.kc, set.zc, pn(set.push));
    let lin = LINEAR_HZ_AT_Q1 * q_ref;
    let held_name = match cut {
        Cut::Hold(a) => Some(format!("hold {a:.2}")),
        Cut::HoldP(a, b) => Some(format!("hold {a:.2} breath {b:.1}")),
        _ => None,
    };
    if let Some(hn) = held_name {
        let held: Vec<&C> = cyc.iter().filter(|c| c.0 > T_CUT + 0.15 && c.0 < T_CUT + 0.3).collect();
        let closing = held.iter().filter(|c| c.6).count();
        return format!(
            "{head}{hn}\t{f_pre:.1}\t{f_ref:.1}\t{lin:.1}\tn{} closing{}\t{:+.1}Hz s{:.2} p{:.2} dab{:.3} oq {oq_ref:.2}->{:.2} imp{:.2}",
            held.len(),
            closing,
            avg(&held, |c| c.1) - f_ref,
            avg(&held, |c| c.2) / sw_ref,
            avg(&held, |c| c.3) / p_ref,
            avg(&held, |c| c.4) - ab_ref,
            avg(&held, |c| c.5),
            avg(&held, |c| c.7) / imp_ref,
        );
    }
    let during = cyc.iter().filter(|c| c.0 > cut_start && c.0 < cut_end).count();
    let after: Vec<&C> = cyc.iter().filter(|c| c.0 > cut_end).take(8).collect();
    let name = match cut {
        Cut::Abduct(a) => format!("abduct {a:.1}"),
        Cut::Breath => "breath".into(),
        Cut::Stop => "stop".into(),
        Cut::Hold(_) | Cut::HoldP(..) => unreachable!(),
    };
    let mut line = format!("{head}{name}\t{f_pre:.1}\t{f_ref:.1}\t{lin:.1}\t{cut_end:.3}\t{during}\t{:.1}", after.first().map_or(f64::NAN, |c| (c.0 - cut_end) * 1000.0));
    for c in &after {
        let back = (c.4 - ab_ref).abs() < 0.005 && (c.3 / p_ref - 1.0).abs() < 0.05;
        line += &format!(
            "\t{}{:+.1}Hz s{:.2} p{:.2} dab{:.3} oq{:.2} {} imp{:.2}",
            if back { "r" } else { "." },
            c.1 - f_ref,
            c.2 / sw_ref,
            c.3 / p_ref,
            c.4 - ab_ref,
            c.5,
            if c.6 { "c" } else { "-" },
            c.7 / imp_ref
        );
    }
    line
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let q_fixed = args.iter().any(|a| a == "--q-fixed");
    let push_off = args.iter().position(|a| a == "--push-off").and_then(|i| args.get(i + 1)).and_then(|s| {
        let (a, b) = s.split_once(',')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let o = Opts { q_fixed, push_off };
    let sets: Vec<Set> = args
        .iter()
        .filter_map(|a| {
            let f: Vec<&str> = a.split(':').collect();
            if f.len() < 2 {
                return None;
            }
            let push = match f.get(2).copied().unwrap_or("both") {
                "both" => (true, true),
                "lower" => (true, false),
                "upper" => (false, true),
                "none" => (false, false),
                _ => return None,
            };
            Some(Set { kc: f[0].parse().ok()?, zc: f[1].parse().ok()?, push })
        })
        .collect();
    let sets = if sets.is_empty() { vec![Set { kc: 3.0, zc: 1.0, push: (true, true) }] } else { sets };
    let rev = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let dirty = std::process::Command::new("git").args(["status", "--porcelain", "--untracked-files=no"]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false);
    println!("# onset_probe at {}{}; args: {}", rev.trim(), if dirty { " (uncommitted changes)" } else { "" }, args.join(" "));
    println!("# folds, 110 Hz, loop frozen at {T_FREEZE} s{}; interruption from 0.98 s to cut_end", if q_fixed { ", the set switched in then (same tension for all sets)" } else { ", each set in force from the start (tension settled per set)" });
    println!("# per cycle after cut_end: r = abduction and pressure back (voicing resumed), . = not yet; dF0 vs settled, s = swing, p = transglottal pressure, dab = abduction above settled (cm2), oq = open share of the channel, c/- = closes or not, imp = closed-phase push impulse, all against settled");
    println!("set\tcut\tpre_hz\tsettled_hz\tlinear_hz\tcut_end_s\tcycles_during\tfirst_cycle_ms_after\tcycle1\tcycle2\tcycle3\tcycle4\tcycle5\tcycle6\tcycle7\tcycle8");
    let cuts = [
        Cut::Abduct(0.1),
        Cut::Abduct(0.2),
        Cut::Abduct(0.3),
        Cut::Breath,
        Cut::Stop,
        Cut::Hold(0.0),
        Cut::Hold(0.02),
        Cut::Hold(0.04),
        Cut::Hold(0.06),
        Cut::Hold(0.1),
        Cut::HoldP(0.0, 0.7),
        Cut::HoldP(0.0, 1.3),
        Cut::HoldP(0.04, 0.7),
        Cut::HoldP(0.04, 1.3),
    ];
    // one set at a time, its cuts in parallel
    let mut lines = vec![];
    for &s in &sets {
        lines.extend(std::thread::scope(|sc| {
            let hs: Vec<_> = cuts.iter().map(|&c| sc.spawn(move || run(c, s, o))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<String>>()
        }));
    }
    for l in lines {
        println!("{l}");
    }
}

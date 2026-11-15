//! Research probe: when does a vowel between voiceless consonants lose its voicing, and through
//! which part of the model?
//!
//! Each word is said in a frame, `e` + word + `to`, with no stress. The target is the word's first
//! vowel; its duration is set directly. Conditions, crossed:
//! - `vot`: the word's voiceless stops unaspirated (`˭`) or aspirated with a nominal voice onset
//!   time (`Segment::vot`; the engine's ramps add to it, so it is a setting, not a measurement).
//! - `larynx`: `overlap` (only the flanking consonants' own openings); `peak` (those openings
//!   reorganised into one opening peaking at the vowel's midpoint, twice the larger of them: the
//!   reading of Fujimoto et al. 2002 in Shaw & Kawahara 2018); `own` (the vowel's own opening,
//!   `Segment::devoiced`, over the whole vowel: closer to Tsuchida 2001's [+spread glottis] on the
//!   vowel), at 0.6 cm², and swept on its own block.
//! - `tongue`: `target` or `none` (the vowel's tongue keys removed; jaw and lips kept).
//!
//! Measured over the whole target vowel:
//! - `voiced15`, `voiced30`, `voiced45`: share of 20 ms frames (5 ms hop, centred in the vowel)
//!   in which the output is periodic (YIN aperiodicity below 0.15, 0.30, 0.45; 100–300 Hz) and
//!   louder than −35 dB re the utterance's loudest frame. The main measure; the threshold is
//!   shown because YIN's usual 0.15 counts a voiced vowel's edges (the closure approaching) as
//!   aperiodic, so a fully voiced vowel scores well below 1. Compare with the voiced controls.
//! - `rms_db`: the vowel's level re the loudest frame.
//! - `gate`: the glottis model's voicing gate (0..1), and its factors before smoothing:
//!   `p_gate` (transglottal pressure over the phonation threshold) and `a_gate` (folds close
//!   enough: 1 below 0.08 cm² static abduction, 0 above 0.22). The model's rule, not the sound.
//! - `abd`: mean static abduction the gate reads (cm²). `closed`: share of the vowel with the tract
//!   shut somewhere (narrowest area < 0.01 cm²). `tb_y`: mean tongue-body height.
//!
//! Output: tab-separated, with `#` lines first (commit, settings).
//! Run: `cargo run --release -q -p chimera --example devoicing_probe > probe.tsv`
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::ArticKey;
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, is_vowel, plan_utterance};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;
const F0: f64 = 120.0;
const OWN_DEFAULT: f64 = 0.6;
const TONGUE: [ArticKey; 8] = [ArticKey::TbX, ArticKey::TbY, ArticKey::TtX, ArticKey::TtY, ArticKey::TtCurl, ArticKey::Root, ArticKey::Dorsum, ArticKey::Lateral];
const STOPS: [&str; 3] = ["p", "t", "k"];

#[derive(Clone, Copy)]
enum Larynx {
    Overlap,
    Peak,
    Own(f64),
}

impl Larynx {
    fn name(self) -> String {
        match self {
            Larynx::Overlap => "overlap".into(),
            Larynx::Peak => "peak".into(),
            Larynx::Own(a) => format!("own{a:.2}"),
        }
    }
}

struct Row {
    dur_ms: f64,
    voiced: [f64; 3],
    rms_db: f64,
    gate: f64,
    p_gate: f64,
    a_gate: f64,
    abd: f64,
    closed: f64,
    tb_y: f64,
}

fn is_label(g: &Gesture, names: &[&str]) -> bool {
    g.label.as_ref().is_some_and(|l| names.contains(&l.as_str()))
}

/// `vot`: None unaspirated, Some(s) aspirated with that nominal voice onset time.
fn probe(word: &str, dur: f64, vot: Option<f64>, larynx: Larynx, tongue_none: bool) -> Row {
    let mut input = tokenize_ipa(&format!("e{word}to"));
    let k = (0..input.len()).filter(|&i| is_vowel(&input[i].ipa, "male".into())).nth(1).expect("a target vowel");
    input[k].dur = Some(dur);
    for s in input.iter_mut().filter(|s| STOPS.contains(&s.ipa.as_str())) {
        s.aspirated = Some(vot.is_some());
        s.vot = vot;
    }
    if let Larynx::Own(_) = larynx {
        input[k].devoiced = Some(true);
    }
    let mut plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), ..Default::default() });
    assert_eq!(plan.segments.len(), input.len(), "{word}: the plan split a segment");
    assert_eq!(plan.segments[k].ipa, input[k].ipa, "{word}: measuring the wrong vowel");
    let (t0, t1) = (plan.segments[k].t0, plan.segments[k].t1);
    assert!((t1 - t0 - dur).abs() < 1e-9, "{word}: the vowel is {:.3} s, not {dur}", t1 - t0);
    match larynx {
        Larynx::Overlap => {}
        Larynx::Own(a) => {
            for g in plan.gestures.iter_mut().filter(|g| is_label(g, &["devoiced"])) {
                g.targets = Targets::default().with(ControlKey::ABDUCTION, a);
            }
        }
        Larynx::Peak => {
            // the flanking voiceless consonants' openings, gathered into one peak at the vowel's
            // midpoint, spanning from the first one's onset to the last one's end
            let (c0, c1) = (&plan.segments[k - 1], &plan.segments[k + 1]);
            let flank = |g: &Gesture| is_label(g, &["aspiration", "voiceless"]) && ((g.t0 < c0.t1 && g.t1 > c0.t0) || (g.t0 < c1.t1 && g.t1 > c1.t0));
            let theirs: Vec<Gesture> = plan.gestures.iter().filter(|g| flank(g)).cloned().collect();
            if !theirs.is_empty() {
                let mid = 0.5 * (t0 + t1);
                let from = theirs.iter().map(|g| g.t0 - g.ramp_in).fold(f64::MAX, f64::min).min(t0);
                let to = theirs.iter().map(|g| g.end()).fold(f64::MIN, f64::max).max(t1);
                let size = 2.0 * theirs.iter().filter_map(|g| g.targets.get(ControlKey::ABDUCTION)).fold(0.0, f64::max);
                plan.gestures.retain(|g| !flank(g));
                plan.gestures.push(Gesture::new(mid, mid, mid - from, to - mid, 6.0, Targets::default().with(ControlKey::ABDUCTION, size)).label("peak"));
            }
        }
    }
    if tongue_none {
        let v = input[k].ipa.clone();
        for g in plan.gestures.iter_mut().filter(|g| is_label(g, &[v.as_str()]) && (g.t0 - t0).abs() < 1e-9) {
            for key in TONGUE {
                g.targets.remove(key);
            }
        }
    }
    let mut v = HumanVoice::with_defaults(FS);
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let (mut gate, mut p_gate, mut a_gate, mut abd, mut closed, mut tb_y) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        gate.push(v.glottis.voicing);
        p_gate.push(v.glottis.pressure_gate);
        a_gate.push(v.glottis.adduct_gate);
        abd.push(v.glottis.static_abduction());
        let f = v.airway.frame.as_ref().expect("the airway publishes a frame every block");
        closed.push(if f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min) < 0.01 { 1.0 } else { 0.0 });
        tb_y.push(v.artic.x.get(ArticKey::TbY));
    }
    let block = |t: f64| ((t * FS) as usize / BLOCK).min(gate.len() - 1);
    let (a, b) = (block(t0), block(t1));
    let mean = |x: &[f64]| x[a..=b].iter().sum::<f64>() / (b - a + 1) as f64;
    // output: 20 ms frames, 5 ms hop
    let (win, hop) = ((0.02 * FS) as usize, (0.005 * FS) as usize);
    let frames: Vec<(f64, f64, f64)> = (0..n.saturating_sub(win))
        .step_by(hop)
        .map(|s| {
            let x: Vec<f64> = out[s..s + win].iter().map(|&y| y as f64).collect();
            let rms = (x.iter().map(|y| y * y).sum::<f64>() / win as f64).sqrt();
            let p = yin_pitch(&x, FS, YinOpts { min_f0: Some(100.0), max_f0: Some(300.0), threshold: None });
            ((s + win / 2) as f64 / FS, rms, p.aperiodicity)
        })
        .collect();
    let loudest = frames.iter().map(|f| f.1).fold(1e-12, f64::max);
    let inside: Vec<&(f64, f64, f64)> = frames.iter().filter(|f| f.0 >= t0 && f.0 <= t1).collect();
    let db = |r: f64| 20.0 * (r / loudest).max(1e-9).log10();
    let share = |thr: f64| inside.iter().filter(|f| f.2 < thr && db(f.1) > -35.0).count() as f64 / inside.len().max(1) as f64;
    let voiced = [share(0.15), share(0.30), share(0.45)];
    let rms_db = db((inside.iter().map(|f| f.1 * f.1).sum::<f64>() / inside.len().max(1) as f64).sqrt());
    Row { dur_ms: (t1 - t0) * 1000.0, voiced, rms_db, gate: mean(&gate), p_gate: mean(&p_gate), a_gate: mean(&a_gate), abd: mean(&abd), closed: mean(&closed), tb_y: mean(&tb_y) }
}

fn main() {
    // (word, group, context): the target is the first vowel of the word
    let words = [
        ("kita", "main", "stop_stop /i/"),
        ("kɯta", "main", "stop_stop /ɯ/"),
        ("ɕita", "main", "fric_stop /i/"),
        ("ɕɯta", "main", "fric_stop /ɯ/ (ɕ)"),
        ("sɯta", "main", "fric_stop /ɯ/ (s)"),
        ("ɸɯta", "main", "fric_stop /ɯ/ (ɸ)"),
        ("kɯɕa", "main", "stop_fric /ɯ/"),
        ("kida", "control", "voiced C2"),
        ("gida", "control", "voiced both"),
        ("ɕiɕa", "check", "sibilant_sibilant /i/"),
        ("sɯsa", "check", "sibilant_sibilant /ɯ/"),
    ];
    let durs: Vec<f64> = (4..=16).map(|d| d as f64 * 0.01).collect();
    let vots = [None, Some(0.03), Some(0.05), Some(0.07)];
    let rev = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let dirty = std::process::Command::new("git").args(["status", "--porcelain", "--untracked-files=no"]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false);
    println!("# devoicing_probe at {}{}", rev.trim(), if dirty { " (uncommitted changes)" } else { "" });
    println!("# male, f0 {F0} Hz, frame e_X_to, no stress, fs {FS}, own opening {OWN_DEFAULT} cm2 (swept in the own-sweep block), voicing gate closes at 0.08-0.22 cm2 static abduction");
    println!("block\tword\tgroup\tcontext\tvot\tlarynx\ttongue\tvowel_ms\tvoiced15\tvoiced30\tvoiced45\trms_db\tgate\tp_gate\ta_gate\tabd\tclosed\ttb_y");
    let rows: Vec<Vec<String>> = std::thread::scope(|sc| {
        let handles: Vec<_> = words
            .iter()
            .map(|&(w, grp, ctx)| {
                let durs = &durs;
                sc.spawn(move || {
                    let mut lines = vec![];
                    let mut emit = |block: &str, vot: Option<f64>, l: Larynx, none: bool, d: f64| {
                        let r = probe(w, d, vot, l, none);
                        let vot_s = vot.map_or("unasp".to_string(), |s| format!("{:.0}ms", s * 1000.0));
                        lines.push(format!(
                            "{block}\t{w}\t{grp}\t{ctx}\t{vot_s}\t{}\t{}\t{:.0}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                            l.name(),
                            if none { "none" } else { "target" },
                            r.dur_ms, r.voiced[0], r.voiced[1], r.voiced[2], r.rms_db, r.gate, r.p_gate, r.a_gate, r.abd, r.closed, r.tb_y
                        ));
                    };
                    for vot in vots {
                        for l in [Larynx::Overlap, Larynx::Peak, Larynx::Own(OWN_DEFAULT)] {
                            for none in [false, true] {
                                for &d in durs {
                                    emit("main", vot, l, none, d);
                                }
                            }
                        }
                    }
                    // the vowel's own opening, swept, at an ordinary length
                    for a in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6] {
                        for &d in &[0.06, 0.082, 0.149] {
                            emit("own-sweep", None, Larynx::Own(a), false, d);
                        }
                    }
                    lines
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for lines in rows {
        for l in lines {
            println!("{l}");
        }
    }
}

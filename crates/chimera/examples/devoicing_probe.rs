//! Research probe: when does a vowel between voiceless consonants lose its voicing, and through
//! which part of the model?
//!
//! Each word is said in a frame, `e` + word + `to`, with no stress. The target is the word's first
//! vowel; its nominal duration is set directly (40–200 ms). Conditions, crossed:
//! - `glottis`: `cutoff` (the engine's default: an opening of 0.08–0.22 cm² stops the folds),
//!   `threshold` (the research option: a phonation threshold rising with the opening), or `folds`
//!   (the self-oscillating two-mass folds, pitch loop on: whether they vibrate follows from their
//!   mechanics; `gate` is then their voicing measure and `p_gate` is not defined, 0). Chosen with
//!   `PROBE_GLOTTIS`, a comma list (default `cutoff,threshold`, the runs before the folds).
//! - `vot`: the word's voiceless stops unaspirated (`˭`) or aspirated with a nominal voice onset
//!   time (`Segment::vot`; the engine's ramps add to it, so it is a setting, not a measurement).
//! - `adv`: the following voiceless consonant's glottal opening starts this much earlier (ms):
//!   how early must it start for the flanking openings to reach the vowel (the overlap account)?
//! - `larynx`: `overlap` (the flanking openings as planned), or `peakA`: the flanking openings
//!   gathered into one raised-cosine peak of A cm² at the vowel's midpoint (strength 2, so the
//!   opening follows the activation and is a peak, not a plateau), from the first one's onset to
//!   the last one's end: Fujimoto et al. 2002 as Shaw & Kawahara 2018 read it, a stop-stop datum,
//!   so only for stop-initial words (kɯɕa is an extrapolation).
//! - `tongue`: `target` or `none` (the vowel's tongue keys removed; jaw and lips kept).
//!
//! The vowel is measured over its open interval: from the first consonant's release (its dominant
//! gesture's activation below one half) to the second's closing (above one half), excluding any
//! time the tract is shut (narrowest area below 0.01 cm²). Measures there:
//! - `open_ms`: that interval's length; the axis to compare with acoustic vowel durations.
//! - `voiced`: share of 20 ms frames (5 ms hop) centred in it in which the output is periodic
//!   (YIN aperiodicity below 0.30, 100–300 Hz) and louder than −35 dB re the loudest frame; empty
//!   below 20 ms. Normalise by the voiced controls before comparing words.
//! - `gate`, `p_gate`: the glottis model's voicing gate and its pressure factor (the model's rule).
//! - `ps`, `pg`, `dp` (cmH₂O): mean subglottal pressure (trachea's last node), supraglottal
//!   pressure (pharynx's first node, the oral pressure behind a constriction), and their difference.
//! - `abd`: mean static abduction the gate reads (cm²). `tb_y`: mean tongue-body height.
//!
//! Output: tab-separated, with `#` lines first (commit, settings).
//! Run: `cargo run --release -q -p chimera --example devoicing_probe > probe.tsv`
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::ArticKey;
use chimera::control::score::{ControlKey, Gesture, Targets, activation};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, is_vowel, plan_utterance};
use chimera::physics::CM_H2O;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const BLOCK: usize = 64;
const F0: f64 = 120.0;
const TONGUE: [ArticKey; 8] = [ArticKey::TbX, ArticKey::TbY, ArticKey::TtX, ArticKey::TtY, ArticKey::TtCurl, ArticKey::Root, ArticKey::Dorsum, ArticKey::Lateral];
const STOPS: [&str; 3] = ["p", "t", "k"];
const OPENINGS: [&str; 2] = ["aspiration", "voiceless"];

#[derive(Clone, Copy)]
struct Cond {
    threshold: bool,
    folds: bool,
    vot: Option<f64>,
    adv: f64,
    peak: Option<f64>,
    tongue_none: bool,
    dur: f64,
}

#[derive(Default)]
struct Row {
    open_ms: f64,
    voiced: Option<f64>,
    gate: f64,
    p_gate: f64,
    ps: f64,
    pg: f64,
    abd: f64,
    tb_y: f64,
}

fn is_label(g: &Gesture, names: &[&str]) -> bool {
    g.label.as_ref().is_some_and(|l| names.contains(&l.as_str()))
}

/// The strongest gesture labelled with this segment's symbol that overlaps it: the consonant's
/// dominant (constriction-forming) gesture.
fn dominant<'a>(gs: &'a [Gesture], ipa: &str, t0: f64, t1: f64) -> Option<&'a Gesture> {
    gs.iter().filter(|g| is_label(g, &[ipa]) && g.t0 < t1 && g.t1 > t0).max_by(|a, b| a.strength.total_cmp(&b.strength))
}

fn probe(word: &str, c: Cond) -> Option<Row> {
    let mut input = tokenize_ipa(&format!("e{word}to"));
    let k = (0..input.len()).filter(|&i| is_vowel(&input[i].ipa, "male".into())).nth(1)?;
    input[k].dur = Some(c.dur);
    for s in input.iter_mut().filter(|s| STOPS.contains(&s.ipa.as_str())) {
        s.aspirated = Some(c.vot.is_some());
        s.vot = c.vot;
    }
    let mut plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), ..Default::default() });
    assert_eq!(plan.segments.len(), input.len(), "{word}: the plan split a segment");
    assert_eq!(plan.segments[k].ipa, input[k].ipa, "{word}: measuring the wrong vowel");
    let (t0, t1) = (plan.segments[k].t0, plan.segments[k].t1);
    let (c0, c1) = (plan.segments[k - 1].clone(), plan.segments[k + 1].clone());
    let flank = |g: &Gesture, s: &chimera::phonetics::PlannedSegment| is_label(g, &OPENINGS) && g.t0 < s.t1 && g.t1 > s.t0;
    // the following consonant's opening starts earlier
    if c.adv > 0.0 {
        for g in plan.gestures.iter_mut().filter(|g| flank(g, &c1)) {
            g.t0 -= c.adv;
        }
    }
    if let Some(size) = c.peak {
        let theirs: Vec<Gesture> = plan.gestures.iter().filter(|g| flank(g, &c0) || flank(g, &c1)).cloned().collect();
        if !theirs.is_empty() {
            let mid = 0.5 * (t0 + t1);
            let from = theirs.iter().map(|g| g.t0 - g.ramp_in).fold(f64::MAX, f64::min).min(t0);
            let to = theirs.iter().map(|g| g.end()).fold(f64::MIN, f64::max).max(t1);
            plan.gestures.retain(|g| !(flank(g, &c0) || flank(g, &c1)));
            plan.gestures.push(Gesture::new(mid, mid, mid - from, to - mid, 2.0, Targets::default().with(ControlKey::ABDUCTION, size)).label("peak"));
        }
    }
    if c.tongue_none {
        let v = input[k].ipa.clone();
        for g in plan.gestures.iter_mut().filter(|g| is_label(g, &[v.as_str()]) && (g.t0 - t0).abs() < 1e-9) {
            for key in TONGUE {
                g.targets.remove(key);
            }
        }
    }
    let g0 = dominant(&plan.gestures, &c0.ipa, c0.t0, c0.t1).cloned();
    let g1 = dominant(&plan.gestures, &c1.ipa, c1.t0, c1.t1).cloned();
    let mut v = HumanVoice::with_defaults(FS);
    v.set_opening_threshold(c.threshold);
    if c.folds {
        v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    }
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let (sub, supra) = (v.airway.trachea.last_node(), v.airway.pharynx.first_node());
    struct B {
        t: f64,
        open: bool,
        gate: f64,
        p_gate: f64,
        ps: f64,
        pg: f64,
        abd: f64,
        tb_y: f64,
    }
    let mut blocks: Vec<B> = vec![];
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        let t = (i + BLOCK / 2) as f64 / FS;
        let f = v.airway.frame.as_ref().expect("the airway publishes a frame every block");
        let shut = f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min) < 0.01;
        let released = g0.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        let not_closing = g1.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        blocks.push(B { t, open: released && not_closing && !shut, gate: v.voicing(), p_gate: if c.folds { 0.0 } else { v.glottis.pressure_gate }, ps: v.airway.net.p[sub] / CM_H2O, pg: v.airway.net.p[supra] / CM_H2O, abd: v.glottis.static_abduction(), tb_y: v.artic.x.get(ArticKey::TbY) });
    }
    // the open run around the vowel: the longest run of open blocks that overlaps [t0, t1]
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < blocks.len() {
        if blocks[i].open {
            let s = i;
            while i < blocks.len() && blocks[i].open {
                i += 1;
            }
            let (bs, be) = (blocks[s].t, blocks[i - 1].t);
            if be >= t0 && bs <= t1 && best.is_none_or(|(a, b)| i - s > b - a) {
                best = Some((s, i));
            }
        } else {
            i += 1;
        }
    }
    let Some((a, b)) = best else { return Some(Row::default()) };
    let w = &blocks[a..b];
    let mean = |f: &dyn Fn(&B) -> f64| w.iter().map(f).sum::<f64>() / w.len() as f64;
    let (o0, o1) = (blocks[a].t - 0.5 * BLOCK as f64 / FS, blocks[b - 1].t + 0.5 * BLOCK as f64 / FS);
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
    let db = |r: f64| 20.0 * (r / loudest).max(1e-9).log10();
    let inside: Vec<&(f64, f64, f64)> = frames.iter().filter(|f| f.0 >= o0 && f.0 <= o1).collect();
    let voiced = (o1 - o0 >= 0.02 && !inside.is_empty()).then(|| inside.iter().filter(|f| f.2 < 0.3 && db(f.1) > -35.0).count() as f64 / inside.len() as f64);
    Some(Row { open_ms: (o1 - o0) * 1000.0, voiced, gate: mean(&|b| b.gate), p_gate: mean(&|b| b.p_gate), ps: mean(&|b| b.ps), pg: mean(&|b| b.pg), abd: mean(&|b| b.abd), tb_y: mean(&|b| b.tb_y) })
}

fn main() {
    // (word, group): the target is the first vowel of the word
    let words = [
        ("kita", "main"),
        ("kɯta", "main"),
        ("ɕita", "main"),
        ("ɕɯta", "main"),
        ("sɯta", "main"),
        ("ɸɯta", "main"),
        ("kɯɕa", "main"),
        ("kida", "control"),
        ("gida", "control"),
        ("gita", "control"),
        ("gɯɕa", "control"),
        ("kata", "control"),
        ("sata", "control"),
        ("ɕiɕa", "check"),
        ("sɯsa", "check"),
    ];
    // PROBE_QUICK=1: two words, a few lengths (a check before the full run)
    let quick = std::env::var("PROBE_QUICK").is_ok_and(|v| v == "1");
    let words: Vec<(&str, &str)> = words.into_iter().filter(|(w, _)| !quick || ["kita", "gida"].contains(w)).collect();
    let peaked = ["kita", "kɯta", "kɯɕa"];
    let durs: Vec<f64> = if quick { vec![0.06, 0.1, 0.16] } else { (4..=20).map(|d| d as f64 * 0.01).collect() };
    let vots = [None, Some(0.03), Some(0.05)];
    let glottises: Vec<String> = std::env::var("PROBE_GLOTTIS").unwrap_or_else(|_| "cutoff,threshold".into()).split(',').map(|g| g.trim().to_string()).collect();
    assert!(glottises.iter().all(|g| ["cutoff", "threshold", "folds"].contains(&g.as_str())), "PROBE_GLOTTIS: cutoff, threshold, folds");
    let rev = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let dirty = std::process::Command::new("git").args(["status", "--porcelain", "--untracked-files=no"]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false);
    println!("# devoicing_probe at {}{}", rev.trim(), if dirty { " (uncommitted changes)" } else { "" });
    println!("# male, f0 {F0} Hz, Breath 8 cmH2O, frame e_X_to, no stress, fs {FS}; glottis cutoff = engine default, threshold = research option (8 cmH2O/cm2 above 0.045 cm2), folds = two-mass; peak strength 2");
    println!("block\tword\tgroup\tglottis\tvot\tadv_ms\tlarynx\ttongue\tnominal_ms\topen_ms\tvoiced\tgate\tp_gate\tps\tpg\tdp\tabd\ttb_y");
    let rows: Vec<Vec<String>> = std::thread::scope(|sc| {
        let handles: Vec<_> = words
            .iter()
            .flat_map(|&(w, grp)| glottises.iter().map(move |g| (w, grp, g.as_str())))
            .map(|(w, grp, glottis)| {
                let (threshold, folds) = (glottis == "threshold", glottis == "folds");
                let durs = &durs;
                sc.spawn(move || {
                    let mut lines = vec![];
                    let mut emit = |block: &str, c: Cond| {
                        let r = probe(w, c).expect("a target vowel");
                        let vot_s = c.vot.map_or("unasp".to_string(), |s| format!("{:.0}ms", s * 1000.0));
                        let lar = c.peak.map_or("overlap".to_string(), |a| format!("peak{a:.2}"));
                        let voiced = r.voiced.map_or(String::new(), |x| format!("{x:.3}"));
                        lines.push(format!(
                            "{block}\t{w}\t{grp}\t{}\t{vot_s}\t{:.0}\t{lar}\t{}\t{:.0}\t{:.1}\t{voiced}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}",
                            glottis,
                            c.adv * 1000.0,
                            if c.tongue_none { "none" } else { "target" },
                            c.dur * 1000.0, r.open_ms, r.gate, r.p_gate, r.ps, r.pg, r.ps - r.pg, r.abd, r.tb_y
                        ));
                    };
                    for vot in vots {
                        for adv in [0.0, 0.02, 0.04, 0.06] {
                            for tongue_none in [false, true] {
                                for &dur in durs {
                                    emit("main", Cond { threshold, folds, vot, adv, peak: None, tongue_none, dur });
                                }
                            }
                        }
                        if peaked.contains(&w) {
                            for size in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] {
                                for &dur in durs {
                                    emit("peak", Cond { threshold, folds, vot, adv: 0.0, peak: Some(size), tongue_none: false, dur });
                                }
                            }
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

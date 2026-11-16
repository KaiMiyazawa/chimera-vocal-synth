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
//! - `lead_ms`, `tail_ms`: from the open interval's start to the first voiced frame's centre, and
//!   from the last voiced frame's centre to its end: the two edges of the voiced stretch, apart.
//! - folds only (blank otherwise): the share of the interval's 20 ms frames in which the folds
//!   close (`f_close`: the narrower channel shuts at least once), swing without closing (`f_swing`:
//!   upper-mass displacement peak-to-peak 0.004 cm or more), or stay still (`f_still`). With the
//!   folds `gate` and `p_gate` are blank (their voicing measure is a held swing, not a decision).
//!
//! Settings by environment (all recorded in the `#` lines): `PROBE_GLOTTIS` (above);
//! `PROBE_WORDS` (a comma list, default all); `PROBE_YIN_MIN` (YIN's lowest F0, Hz, default 100);
//! `PROBE_FOLDS_LOOP` (`on`, default, or `frozen`: the folds' trim is settled once on a held /e/
//! at the probe's F0 and kept, the loop off); `PROBE_ABDUCT_REST` (the folds' `abduct_rest`,
//! default the engine's 0.5; 0 makes the rest area equal the abduction asked for);
//! `PROBE_LUNG_R` (factor on the lungs' series resistance, default 1); `PROBE_TRACE` (`word,ms`:
//! for that word and nominal length, unaspirated, frame by frame around the release, to stderr);
//! `PROBE_LOCK_RELEASE=1` (the first consonant's glottal openings move with where its release
//! falls, found in a first pass: the openings are timed to the release, not the plan).
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
    /// the first consonant's openings move by this much (s); set by `PROBE_LOCK_RELEASE`
    shift: f64,
}

#[derive(Default)]
struct Row {
    /// the open interval's start minus the planned vowel's start (s)
    start: f64,
    lead_ms: Option<f64>,
    tail_ms: Option<f64>,
    states: Option<(f64, f64, f64)>,
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

/// With `PROBE_LOCK_RELEASE` the first consonant's openings are moved by where its release fell
/// in a first pass (the open interval's start against the planned vowel's start), so that they
/// keep the same timing against the release whatever the vowel; otherwise one pass.
fn probe(word: &str, c: Cond) -> Option<Row> {
    let first = probe_once(word, c)?;
    if !settings().lock_release || first.open_ms == 0.0 {
        return Some(first);
    }
    let mut r = probe_once(word, Cond { shift: first.start, ..c })?;
    r.start = first.start;
    Some(r)
}

fn probe_once(word: &str, c: Cond) -> Option<Row> {
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
    if c.shift != 0.0 {
        for g in plan.gestures.iter_mut().filter(|g| flank(g, &c0)) {
            g.t0 += c.shift;
            g.t1 += c.shift;
        }
    }
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
    let st = settings();
    if c.folds {
        v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
        if let Some(abr) = st.abduct_rest {
            v.folds.abduct_rest = abr;
        }
        if let Some(trim) = st.frozen_trim {
            v.folds.set_trim(trim);
            v.folds.trim_loop = false;
        }
    }
    if st.lung_r != 1.0 {
        let lp = v.airway.lung_port;
        let r = v.airway.net.port_extra_resistance(lp);
        v.airway.net.set_port_extra_resistance(lp, r * st.lung_r);
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
        amin: f64,
        chmin: f64,
    }
    let mut blocks: Vec<B> = vec![];
    // the folds' channel and swing, sample by sample (folds only)
    let (mut ch, mut xs) = (vec![], vec![]);
    for i in (0..n).step_by(BLOCK) {
        if c.folds {
            for j in i..(i + BLOCK).min(n) {
                v.render(&mut out[j..j + 1]);
                ch.push(v.folds.channel_area());
                xs.push(v.folds.displacement());
            }
        } else {
            v.render(&mut out[i..(i + BLOCK).min(n)]);
        }
        let t = (i + BLOCK / 2) as f64 / FS;
        let f = v.airway.frame.as_ref().expect("the airway publishes a frame every block");
        let amin = f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min);
        let shut = amin < 0.01;
        let chmin = if c.folds { ch[i..].iter().cloned().fold(f64::MAX, f64::min) } else { f64::NAN };
        let released = g0.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        let not_closing = g1.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        blocks.push(B { t, open: released && not_closing && !shut, gate: v.voicing(), p_gate: v.glottis.pressure_gate, ps: v.airway.net.p[sub] / CM_H2O, pg: v.airway.net.p[supra] / CM_H2O, abd: if c.folds { v.glottis.params.abduction } else { v.glottis.static_abduction() }, tb_y: v.artic.x.get(ArticKey::TbY), amin, chmin });
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
    let frames: Vec<(f64, f64, f64, usize)> = (0..n.saturating_sub(win))
        .step_by(hop)
        .map(|s| {
            let x: Vec<f64> = out[s..s + win].iter().map(|&y| y as f64).collect();
            let rms = (x.iter().map(|y| y * y).sum::<f64>() / win as f64).sqrt();
            let p = yin_pitch(&x, FS, YinOpts { min_f0: Some(st.yin_min), max_f0: Some(300.0), threshold: None });
            ((s + win / 2) as f64 / FS, rms, p.aperiodicity, s)
        })
        .collect();
    let loudest = frames.iter().map(|f| f.1).fold(1e-12, f64::max);
    let db = |r: f64| 20.0 * (r / loudest).max(1e-9).log10();
    let inside: Vec<&(f64, f64, f64, usize)> = frames.iter().filter(|f| f.0 >= o0 && f.0 <= o1).collect();
    let is_voiced = |f: &(f64, f64, f64, usize)| f.2 < 0.3 && db(f.1) > -35.0;
    let voiced = (o1 - o0 >= 0.02 && !inside.is_empty()).then(|| inside.iter().filter(|f| is_voiced(f)).count() as f64 / inside.len() as f64);
    let lead_ms = inside.iter().find(|f| is_voiced(f)).map(|f| (f.0 - o0) * 1000.0);
    let tail_ms = inside.iter().rev().find(|f| is_voiced(f)).map(|f| (o1 - f.0) * 1000.0);
    let states = (c.folds && !inside.is_empty()).then(|| {
        let (mut close, mut swing) = (0usize, 0usize);
        for f in &inside {
            let (a, b) = (f.3, (f.3 + win).min(ch.len()));
            if ch[a..b].iter().any(|&x| x <= 1e-4) {
                close += 1;
            } else if xs[a..b].iter().cloned().fold(f64::MIN, f64::max) - xs[a..b].iter().cloned().fold(f64::MAX, f64::min) >= 0.004 {
                swing += 1;
            }
        }
        let k = inside.len() as f64;
        (close as f64 / k, swing as f64 / k, (inside.len() - close - swing) as f64 / k)
    });
    // PROBE_TRACE=word,nominal_ms: frame by frame around the release (unaspirated, no advance,
    // overlap, tongue target), to stderr
    if let Some((tw, td)) = &st.trace
        && tw == word && (c.dur * 1000.0 - td).abs() < 0.5 && c.vot.is_none() && c.adv == 0.0 && c.peak.is_none() && !c.tongue_none
    {
        let gl = if c.folds { "folds" } else if c.threshold { "threshold" } else { "cutoff" };
        eprintln!("# trace {word} {td} ms {gl}: open {:.1} ms, from {:.1} ms after the planned vowel's start to {:.1} ms after its end; loudest frame rms {:.4}", (o1 - o0) * 1000.0, (o0 - t0) * 1000.0, (o1 - t1) * 1000.0, loudest);
        eprintln!("kind\tglottis\tword\tt_ms\tap\tdb\tvoiced\tamin\tps\tpg\tdp\tabd\tchmin");
        for f in frames.iter().filter(|f| f.0 >= o0 - 0.03 && f.0 <= o0 + 0.09) {
            eprintln!("frame\t{gl}\t{word}\t{:.1}\t{:.3}\t{:.1}\t{}", (f.0 - o0) * 1000.0, f.2, db(f.1), is_voiced(f) as u8);
        }
        for b in blocks.iter().filter(|b| b.t >= o0 - 0.03 && b.t <= o0 + 0.09) {
            eprintln!("block\t{gl}\t{word}\t{:.1}\t\t\t\t{:.4}\t{:.2}\t{:.2}\t{:.2}\t{:.4}\t{:.5}", (b.t - o0) * 1000.0, b.amin, b.ps, b.pg, b.ps - b.pg, b.abd, b.chmin);
        }
    }
    Some(Row { start: o0 - t0, lead_ms, tail_ms, states, open_ms: (o1 - o0) * 1000.0, voiced, gate: mean(&|b| b.gate), p_gate: mean(&|b| b.p_gate), ps: mean(&|b| b.ps), pg: mean(&|b| b.pg), abd: mean(&|b| b.abd), tb_y: mean(&|b| b.tb_y) })
}

/// Settings read from the environment (see the module comment).
struct Settings {
    words: Option<Vec<String>>,
    yin_min: f64,
    frozen_trim: Option<f64>,
    abduct_rest: Option<f64>,
    lung_r: f64,
    trace: Option<(String, f64)>,
    lock_release: bool,
}

static SETTINGS: std::sync::OnceLock<Settings> = std::sync::OnceLock::new();

fn settings() -> &'static Settings {
    SETTINGS.get_or_init(|| {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let num = |k: &str| env(k).map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("{k}: a number")));
        let frozen = match env("PROBE_FOLDS_LOOP").as_deref() {
            None | Some("on") => false,
            Some("frozen") => true,
            Some(o) => panic!("PROBE_FOLDS_LOOP: on or frozen, not {o}"),
        };
        Settings {
            words: env("PROBE_WORDS").map(|v| v.split(',').map(|w| w.trim().to_string()).collect()),
            yin_min: num("PROBE_YIN_MIN").unwrap_or(100.0),
            frozen_trim: frozen.then(settled_trim),
            abduct_rest: num("PROBE_ABDUCT_REST"),
            lung_r: num("PROBE_LUNG_R").unwrap_or(1.0),
            lock_release: env("PROBE_LOCK_RELEASE").is_some_and(|v| v == "1"),
            trace: env("PROBE_TRACE").map(|v| {
                let (w, d) = v.split_once(',').expect("PROBE_TRACE: word,nominal_ms");
                (w.trim().to_string(), d.trim().parse().expect("PROBE_TRACE: nominal ms, a number"))
            }),
        }
    })
}

/// The folds' trim after the loop has settled on a held /e/ at the probe's F0 (frame's first vowel).
fn settled_trim() -> f64 {
    let mut input = tokenize_ipa("e");
    input[0].dur = Some(0.8);
    input[0].stress = Some(0);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    if let Some(abr) = std::env::var("PROBE_ABDUCT_REST").ok().and_then(|x| x.parse().ok()) {
        v.folds.abduct_rest = abr;
    }
    v.schedule(&plan.gestures);
    let mut out = vec![0.0f32; (0.7 * FS) as usize];
    v.render(&mut out);
    v.folds.trim()
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
        ("gada", "control"),
        ("gɯda", "control"),
        ("gita", "control"),
        ("gɯɕa", "control"),
        ("kata", "control"),
        ("sata", "control"),
        ("ɕiɕa", "check"),
        ("sɯsa", "check"),
    ];
    // PROBE_QUICK=1: two words, a few lengths (a check before the full run)
    let quick = std::env::var("PROBE_QUICK").is_ok_and(|v| v == "1");
    let st = settings();
    let words: Vec<(&str, &str)> = words.into_iter().filter(|(w, _)| !quick || ["kita", "gida"].contains(w)).filter(|(w, _)| st.words.as_ref().is_none_or(|ws| ws.iter().any(|x| x == w))).collect();
    let peaked = ["kita", "kɯta", "kɯɕa"];
    // with PROBE_TRACE only the traced condition is run
    let traced = st.trace.as_ref().map(|t| t.1 / 1000.0);
    let durs: Vec<f64> = if let Some(d) = traced { vec![d] } else if quick { vec![0.06, 0.1, 0.16] } else { (4..=20).map(|d| d as f64 * 0.01).collect() };
    let vots: Vec<Option<f64>> = if traced.is_some() { vec![None] } else { vec![None, Some(0.03), Some(0.05)] };
    let glottises: Vec<String> = std::env::var("PROBE_GLOTTIS").unwrap_or_else(|_| "cutoff,threshold".into()).split(',').map(|g| g.trim().to_string()).collect();
    assert!(glottises.iter().all(|g| ["cutoff", "threshold", "folds"].contains(&g.as_str())), "PROBE_GLOTTIS: cutoff, threshold, folds");
    let rev = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let dirty = std::process::Command::new("git").args(["status", "--porcelain", "--untracked-files=no"]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false);
    println!("# devoicing_probe at {}{}", rev.trim(), if dirty { " (uncommitted changes)" } else { "" });
    println!("# male, f0 {F0} Hz, Breath 8 cmH2O, frame e_X_to, no stress, fs {FS}; glottis cutoff = engine default, threshold = research option (8 cmH2O/cm2 above 0.045 cm2), folds = two-mass; peak strength 2");
    println!(
        "# run: glottis {}; words {}; YIN min {} Hz; folds loop {}; abduct_rest {}; lung resistance x{}; openings locked to the release {}",
        glottises.join(","),
        st.words.as_ref().map_or("all".to_string(), |w| w.join(",")),
        st.yin_min,
        st.frozen_trim.map_or("on".to_string(), |t| format!("frozen (trim {t:.4})")),
        st.abduct_rest.map_or("engine".to_string(), |a| a.to_string()),
        st.lung_r,
        st.lock_release
    );
    println!("block\tword\tgroup\tglottis\tvot\tadv_ms\tlarynx\ttongue\tnominal_ms\topen_ms\tvoiced\tgate\tp_gate\tps\tpg\tdp\tabd\ttb_y\tlead_ms\ttail_ms\tf_close\tf_swing\tf_still");
    let rows: Vec<Vec<String>> = std::thread::scope(|sc| {
        let handles: Vec<_> = words
            .iter()
            .flat_map(|&(w, grp)| glottises.iter().map(move |g| (w, grp, g.as_str())))
            .map(|(w, grp, glottis)| {
                let (threshold, folds) = (glottis == "threshold", glottis == "folds");
                let (durs, vots) = (&durs, &vots);
                sc.spawn(move || {
                    let mut lines = vec![];
                    let mut emit = |block: &str, c: Cond| {
                        let r = probe(w, c).expect("a target vowel");
                        let vot_s = c.vot.map_or("unasp".to_string(), |s| format!("{:.0}ms", s * 1000.0));
                        let lar = c.peak.map_or("overlap".to_string(), |a| format!("peak{a:.2}"));
                        let voiced = r.voiced.map_or(String::new(), |x| format!("{x:.3}"));
                        let opt = |x: Option<f64>, d: usize| x.map_or(String::new(), |x| format!("{x:.d$}"));
                        let (gate, p_gate) = if folds { (String::new(), String::new()) } else { (format!("{:.3}", r.gate), format!("{:.3}", r.p_gate)) };
                        let states = r.states.map_or("\t\t".to_string(), |(a, b, c)| format!("{a:.3}\t{b:.3}\t{c:.3}"));
                        lines.push(format!(
                            "{block}\t{w}\t{grp}\t{}\t{vot_s}\t{:.0}\t{lar}\t{}\t{:.0}\t{:.1}\t{voiced}\t{gate}\t{p_gate}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{}\t{}\t{states}",
                            glottis,
                            c.adv * 1000.0,
                            if c.tongue_none { "none" } else { "target" },
                            c.dur * 1000.0, r.open_ms, r.ps, r.pg, r.ps - r.pg, r.abd, r.tb_y, opt(r.lead_ms, 1), opt(r.tail_ms, 1)
                        ));
                    };
                    let traced = traced.is_some();
                    for &vot in vots {
                        for adv in if traced { vec![0.0] } else { vec![0.0, 0.02, 0.04, 0.06] } {
                            for tongue_none in if traced { vec![false] } else { vec![false, true] } {
                                for &dur in durs {
                                    emit("main", Cond { threshold, folds, vot, adv, peak: None, tongue_none, dur, shift: 0.0 });
                                }
                            }
                        }
                        if peaked.contains(&w) && !traced {
                            for size in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] {
                                for &dur in durs {
                                    emit("peak", Cond { threshold, folds, vot, adv: 0.0, peak: Some(size), tongue_none: false, dur, shift: 0.0 });
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

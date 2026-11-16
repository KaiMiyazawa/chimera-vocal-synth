//! Research probe: does the vowel's raised tongue keep frication going through a devoiced vowel,
//! and does that depend on the consonant before it?
//!
//! Shaw & Kawahara (2021) proposed, without testing it, that a devoiced high vowel keeps its
//! tongue raise after [ɸ] because the raise narrows the tract, so turbulence (frication) goes on
//! through the voiceless vowel; after a lingual fricative the tongue is near there anyway. Here the
//! target vowel of `e` + word + `to` is marked devoiced (`Segment::devoiced`: a glottal opening over
//! the vowel) or not, with its tongue keys kept (`target`) or removed (`none`; jaw and lips kept),
//! and the frication the model makes over the vowel's open interval is read off.
//!
//! The open interval is the same as in `devoicing_probe`: from the first consonant's release (its
//! dominant gesture's activation below one half) to the second's closing, excluding any time the
//! tract is shut (narrowest area below 0.01 cm²). Measures there:
//! - `open_ms`, `voiced` (share of 20 ms frames, 5 ms hop, periodic by YIN below 0.30 at 60–300 Hz
//!   and above −35 dB re the loudest frame).
//! - `amin`: mean narrowest supraglottal area (cm²); `noisy`: share of blocks in which the
//!   narrowest point makes a frication source (area below the engine's fade-out, 0.9 cm² at this
//!   body scale); now the engine's own weight (1 − smoothstep(0.35, 0.9, area)) averaged.
//! - `n_glot`, `n_constr`, `n_teeth`: mean amplitude of the three turbulence sources the airway
//!   registers (aspiration at the glottis, the narrowest constriction, the teeth obstacle), in the
//!   network's units; `c_constr`, `c_teeth`: the same over the first consonant's planned middle half
//!   (the fricative itself; zero for a stop).
//! - `hole`: share of blocks in which the engine's narrowest point should make a source (its weight
//!   above zero) but none was registered; `pos`: mean index of that point (0 is the oral tube's
//!   first section, negative the pharynx); `ag_area`: mean glottal branch area (cm²).
//! - `hf_abs_db`, `lo_abs_db`: absolute output energy above 2 kHz and in 1–2.6 kHz (dB).
//! - `hf_db`: output energy above 2 kHz over the open interval, dB re the loudest 20 ms frame's
//!   energy above 2 kHz in the whole utterance; `hf_rel_db`: the same re the first consonant's
//!   planned middle half (blank for a stop).
//!
//! The devoiced vowel's glottal opening is the planner's (0.6 cm²) or, with `PROBE_AG` (a comma
//! list of cm²), each of those in turn: with the glottis open the flow is limited by the glottis and
//! the constriction in series, so a constriction's frication should peak when its area is near the
//! glottis's (column `ag`, 0 for the planner's value, blank for the voiced vowel).
//! - `noise_end_ms`: after a fricative, from the open interval's start to the first block whose
//!   summed turbulence amplitude is below half its mean over the fricative's middle half (the
//!   interval's length if it never is): how long the aperiodic energy is prolonged into the vowel.
//! - `tb_y`: mean tongue-body height over the interval.
//!
//! `PROBE_TB_DY` (a comma list): the vowel's tongue-body height target moved by each amount (the
//! tongue kept), to sweep the lingual constriction across the other one; `none` is always run.
//! `PROBE_WORDS`, `PROBE_DURS` (ms), `PROBE_VOICED_ONLY=1` (no devoiced rows), `PROBE_JUNCTION=1`
//! (the airway's `junction_noise` research option: no dropped source just past the velum).
//! Settings by environment: `PROBE_GLOTTIS` (`cutoff`, `folds` or both, default both; the folds
//! run with their pitch loop settled on a held /e/ and frozen, as `devoicing_probe` does with
//! `PROBE_FOLDS_LOOP=frozen`).
//! Run: `cargo run --release -q -p chimera --example frication_probe > frication.tsv`
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::ArticKey;
use chimera::control::score::{ControlKey, Gesture, Targets, activation};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, is_vowel, plan_utterance};
use chimera::math::smoothstep;
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const BLOCK: usize = 64;
const F0: f64 = 120.0;
const TONGUE: [ArticKey; 8] = [ArticKey::TbX, ArticKey::TbY, ArticKey::TtX, ArticKey::TtY, ArticKey::TtCurl, ArticKey::Root, ArticKey::Dorsum, ArticKey::Lateral];
/// the engine's frication fade-out area (cm²) at the default body scale (`update_noise`)
const NOISY_AREA: f64 = 0.9;

fn is_label(g: &Gesture, names: &[&str]) -> bool {
    g.label.as_ref().is_some_and(|l| names.contains(&l.as_str()))
}

fn dominant<'a>(gs: &'a [Gesture], ipa: &str, t0: f64, t1: f64) -> Option<&'a Gesture> {
    gs.iter().filter(|g| is_label(g, &[ipa]) && g.t0 < t1 && g.t1 > t0).max_by(|a, b| a.strength.total_cmp(&b.strength))
}

/// Energy above about 2 kHz: a second-order high-pass (RBJ biquad, Q 0.707).
fn highpass(x: &[f32]) -> Vec<f64> {
    highpass_at(x, 2000.0)
}

fn highpass_at(x: &[f32], fc: f64) -> Vec<f64> {
    let w0 = 2.0 * std::f64::consts::PI * fc / FS;
    let (c, s) = (w0.cos(), w0.sin());
    let alpha = s / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = ((1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0);
    let (a1, a2) = (-2.0 * c / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    x.iter()
        .map(|&v| {
            let v = v as f64;
            let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            (x2, x1, y2, y1) = (x1, v, y1, y);
            y
        })
        .collect()
}

struct Row {
    open_ms: f64,
    voiced: Option<f64>,
    amin: f64,
    noisy: f64,
    n_glot: f64,
    n_constr: f64,
    n_teeth: f64,
    c_constr: f64,
    c_teeth: f64,
    hf_db: f64,
    hf_rel_db: Option<f64>,
    hole: f64,
    pos: f64,
    ag: f64,
    hf_abs_db: f64,
    lo_abs_db: f64,
    noise_end_ms: Option<f64>,
    tb_y: f64,
}

fn probe(word: &str, folds: Option<f64>, devoiced: Option<f64>, tongue_none: bool, tb_dy: f64, dur: f64) -> Option<Row> {
    let mut input = tokenize_ipa(&format!("e{word}to"));
    let k = (0..input.len()).filter(|&i| is_vowel(&input[i].ipa, "male".into())).nth(1)?;
    let vowel_ipa = input[k].ipa.clone();
    input[k].dur = Some(dur);
    input[k].devoiced = Some(devoiced.is_some());
    let mut plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), ..Default::default() });
    assert_eq!(plan.segments.len(), input.len(), "{word}: the plan split a segment");
    assert_eq!(plan.segments[k].ipa, vowel_ipa, "{word}: measuring the wrong vowel");
    let (t0, t1) = (plan.segments[k].t0, plan.segments[k].t1);
    let (c0, c1) = (plan.segments[k - 1].clone(), plan.segments[k + 1].clone());
    // the devoiced vowel's glottal opening, if not the planner's
    if let Some(ag) = devoiced.filter(|&a| a > 0.0) {
        for g in plan.gestures.iter_mut().filter(|g| is_label(g, &["devoiced"])) {
            g.targets = Targets::default().with(ControlKey::ABDUCTION, ag);
        }
    }
    if tb_dy != 0.0 {
        let v = input[k].ipa.clone();
        let mut hit = 0;
        for g in plan.gestures.iter_mut().filter(|g| is_label(g, &[v.as_str()]) && (g.t0 - t0).abs() < 1e-9) {
            if let Some(y) = g.targets.get(ArticKey::TbY) {
                g.targets.set(ArticKey::TbY, y + tb_dy);
                hit += 1;
            }
        }
        assert!(hit > 0, "{word}: no tongue-body height on the vowel to move");
    }
    if tongue_none {
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
    v.airway.junction_noise = std::env::var("PROBE_JUNCTION").is_ok_and(|x| x == "1");
    if let Some(trim) = folds {
        v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
        v.folds.set_trim(trim);
        v.folds.trim_loop = false;
    }
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    #[derive(Clone, Copy)]
    struct B {
        t: f64,
        open: bool,
        amin: f64,
        lv: [f64; 3],
        tb_y: f64,
        /// the engine's own choice: narrowest point's index (oral index n_ph = 0; negative is
        /// pharynx), its area, its source weight, whether a source was registered, glottal area
        pos: f64,
        w: f64,
        hole: bool,
        ag: f64,
    }
    let mut blocks: Vec<B> = vec![];
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        let t = (i + BLOCK / 2) as f64 / FS;
        let f = v.airway.frame.as_ref().expect("the airway publishes a frame every block");
        let amin = f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min);
        let released = g0.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        let not_closing = g1.as_ref().is_none_or(|g| activation(g, t) < 0.5);
        // sources in registration order: glottis, then the narrowest point, then the teeth
        let src = v.airway.net.noise_sources();
        let mut lv = [0.0; 3];
        for (j, s) in src.iter().enumerate().take(3) {
            lv[j] = s.level;
        }
        let a = &v.airway;
        let (mut min_a, mut min_i) = (f64::INFINITY, -1isize);
        for j in 0..a.n_ph + a.n_or {
            let ar = if j < a.n_ph { a.ph_min[j] } else { a.or_min[j - a.n_ph] };
            if ar > 0.0 && ar < min_a {
                (min_a, min_i) = (ar, j as isize);
            }
        }
        let w = if min_i >= 0 { 1.0 - smoothstep(0.35, NOISY_AREA, min_a) } else { 0.0 };
        let hole = w > 0.0 && src.len() < 2;
        let ag = a.net.branch_area[a.glottis_port];
        blocks.push(B { t, open: released && not_closing && amin >= 0.01, amin, lv, pos: (min_i - a.n_ph as isize) as f64, w, hole, ag, tb_y: v.artic.x.get(ArticKey::TbY) });
    }
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < blocks.len() {
        if blocks[i].open {
            let s = i;
            while i < blocks.len() && blocks[i].open {
                i += 1;
            }
            if blocks[i - 1].t >= t0 && blocks[s].t <= t1 && best.is_none_or(|(a, b)| i - s > b - a) {
                best = Some((s, i));
            }
        } else {
            i += 1;
        }
    }
    let (a, b) = best.unwrap_or_else(|| panic!("{word}: no open interval"));
    let tb_y_mean = |ws: &[B]| ws.iter().map(|b| b.tb_y).sum::<f64>() / ws.len().max(1) as f64;
    let w = &blocks[a..b];
    let mean = |ws: &[B], f: &dyn Fn(&B) -> f64| if ws.is_empty() { 0.0 } else { ws.iter().map(f).sum::<f64>() / ws.len() as f64 };
    let (o0, o1) = (blocks[a].t - 0.5 * BLOCK as f64 / FS, blocks[b - 1].t + 0.5 * BLOCK as f64 / FS);
    let (m0, m1) = (c0.t0 + 0.25 * (c0.t1 - c0.t0), c0.t0 + 0.75 * (c0.t1 - c0.t0));
    let cmid: Vec<B> = blocks.iter().filter(|b| b.t >= m0 && b.t <= m1).map(|b| B { lv: b.lv, ..*b }).collect();
    let fric = c0.ipa != "k" && c0.ipa != "t" && c0.ipa != "p";
    // voicing, as in devoicing_probe
    let (win, hop) = ((0.02 * FS) as usize, (0.005 * FS) as usize);
    let hp = highpass(&out);
    let frames: Vec<(f64, f64, f64, f64)> = (0..n.saturating_sub(win))
        .step_by(hop)
        .map(|s| {
            let x: Vec<f64> = out[s..s + win].iter().map(|&y| y as f64).collect();
            let rms = (x.iter().map(|y| y * y).sum::<f64>() / win as f64).sqrt();
            let hfe = hp[s..s + win].iter().map(|y| y * y).sum::<f64>() / win as f64;
            let p = yin_pitch(&x, FS, YinOpts { min_f0: Some(60.0), max_f0: Some(300.0), threshold: None });
            ((s + win / 2) as f64 / FS, rms, p.aperiodicity, hfe)
        })
        .collect();
    let loudest = frames.iter().map(|f| f.1).fold(1e-12, f64::max);
    let hf_max = frames.iter().map(|f| f.3).fold(1e-30, f64::max);
    let inside: Vec<_> = frames.iter().filter(|f| f.0 >= o0 && f.0 <= o1).collect();
    let voiced = (o1 - o0 >= 0.02 && !inside.is_empty()).then(|| inside.iter().filter(|f| f.2 < 0.3 && 20.0 * (f.1 / loudest).max(1e-9).log10() > -35.0).count() as f64 / inside.len() as f64);
    let hf_in: f64 = hp[(o0 * FS) as usize..((o1 * FS) as usize).min(n)].iter().map(|y| y * y).sum::<f64>() / ((o1 - o0) * FS).max(1.0);
    let hf_c: f64 = hp[(m0 * FS) as usize..((m1 * FS) as usize).min(n)].iter().map(|y| y * y).sum::<f64>() / ((m1 - m0) * FS).max(1.0);
    let db = |x: f64| 10.0 * x.max(1e-30).log10();
    // 1–2.6 kHz: energy above 1 kHz minus energy above 2.6 kHz (second-order edges)
    let (h1, h26) = (highpass_at(&out, 1000.0), highpass_at(&out, 2600.0));
    let seg = |h: &[f64]| h[(o0 * FS) as usize..((o1 * FS) as usize).min(n)].iter().map(|y| y * y).sum::<f64>() / ((o1 - o0) * FS).max(1.0);
    let lo_in = (seg(&h1) - seg(&h26)).max(1e-30);
    Some(Row {
        open_ms: (o1 - o0) * 1000.0,
        voiced,
        amin: mean(w, &|b| b.amin),
        noisy: mean(w, &|b| b.w),
        hole: mean(w, &|b| b.hole as u8 as f64),
        pos: mean(w, &|b| b.pos),
        ag: mean(w, &|b| b.ag),
        noise_end_ms: fric.then(|| {
            let c = mean(&cmid, &|b| b.lv.iter().sum());
            w.iter().find(|b| b.lv.iter().sum::<f64>() < 0.5 * c).map_or((o1 - o0) * 1000.0, |b| (b.t - o0) * 1000.0)
        }),
        tb_y: tb_y_mean(w),
        hf_abs_db: db(hf_in),
        lo_abs_db: db(lo_in),
        n_glot: mean(w, &|b| b.lv[0]),
        n_constr: mean(w, &|b| b.lv[1]),
        n_teeth: mean(w, &|b| b.lv[2]),
        c_constr: mean(&cmid, &|b| b.lv[1]),
        c_teeth: mean(&cmid, &|b| b.lv[2]),
        hf_db: db(hf_in) - db(hf_max),
        hf_rel_db: fric.then(|| db(hf_in) - db(hf_c)),
    })
}

/// The folds' trim after the loop has settled on a held /e/ at the probe's F0.
fn settled_trim() -> f64 {
    let mut input = tokenize_ipa("e");
    input[0].dur = Some(0.8);
    input[0].stress = Some(0);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(F0), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    v.schedule(&plan.gestures);
    let mut out = vec![0.0f32; (0.7 * FS) as usize];
    v.render(&mut out);
    v.folds.trim()
}

fn main() {
    let words: Vec<String> = std::env::var("PROBE_WORDS").ok().filter(|v| !v.is_empty()).map_or(["ɸɯta", "ɕɯta", "sɯta", "kɯta", "ɕita", "kita", "çita"].map(String::from).to_vec(), |v| v.split(',').map(|w| w.trim().to_string()).collect());
    let dys: Vec<f64> = std::env::var("PROBE_TB_DY").ok().filter(|v| !v.is_empty()).map_or(vec![0.0], |v| v.split(',').map(|x| x.trim().parse().expect("PROBE_TB_DY: numbers")).collect());
    let no_ag = std::env::var("PROBE_VOICED_ONLY").is_ok_and(|v| v == "1");
    let ags: Vec<f64> = std::env::var("PROBE_AG").ok().filter(|v| !v.is_empty()).map_or(vec![0.0], |v| v.split(',').map(|x| x.trim().parse().expect("PROBE_AG: numbers")).collect());
    let durs: Vec<f64> = std::env::var("PROBE_DURS").ok().filter(|v| !v.is_empty()).map_or(vec![0.06, 0.08, 0.1, 0.12], |v| v.split(',').map(|x| x.trim().parse::<f64>().expect("PROBE_DURS: ms") / 1000.0).collect());
    let glottises: Vec<String> = std::env::var("PROBE_GLOTTIS").unwrap_or_else(|_| "cutoff,folds".into()).split(',').map(|g| g.trim().to_string()).collect();
    assert!(glottises.iter().all(|g| ["cutoff", "folds"].contains(&g.as_str())), "PROBE_GLOTTIS: cutoff, folds");
    let trim = glottises.iter().any(|g| g == "folds").then(settled_trim);
    let rev = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let dirty = std::process::Command::new("git").args(["status", "--porcelain", "--", "crates"]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false);
    println!("# frication_probe at {}{}", rev.trim(), if dirty { " (uncommitted changes)" } else { "" });
    println!("# male, f0 {F0} Hz, Breath 8 cmH2O, frame e_X_to, no stress, fs {FS}; folds loop frozen{}; junction noise {}", trim.map_or(String::new(), |t| format!(" (trim {t:.4})")), std::env::var("PROBE_JUNCTION").is_ok_and(|x| x == "1"));
    println!("word\tglottis\tdevoiced\tag\ttongue\ttb_dy\tnominal_ms\topen_ms\tvoiced\tamin\tnoisy\tn_glot\tn_constr\tn_teeth\tc_constr\tc_teeth\thf_db\thf_rel_db\thole\tpos\tag_area\thf_abs_db\tlo_abs_db\tnoise_end_ms\ttb_y");
    let rows: Vec<Vec<String>> = std::thread::scope(|sc| {
        let hs: Vec<_> = words
            .iter()
            .flat_map(|w| glottises.iter().map(move |g| (w.as_str(), g.clone())))
            .map(|(w, g)| {
                let (ags, durs, dys) = (&ags, &durs, &dys);
                sc.spawn(move || {
                    let folds = if g == "folds" { trim } else { None };
                    let mut lines = vec![];
                    let conds: Vec<Option<f64>> = std::iter::once(None).chain(ags.iter().filter(|_| !no_ag).map(|&a| Some(a))).collect();
                    let tongues: Vec<(bool, f64)> = std::iter::once((true, 0.0)).chain(dys.iter().map(|&y| (false, y))).collect();
                    for &devoiced in &conds {
                        for &(tongue_none, dy) in &tongues {
                            for &d in durs {
                                let r = probe(w, folds, devoiced, tongue_none, dy, d).expect("a target vowel");
                                lines.push(format!(
                                    "{w}\t{g}\t{}\t{}\t{}\t{dy}\t{:.0}\t{:.1}\t{}\t{:.3}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{}\t{:.2}\t{:.1}\t{:.3}\t{:.1}\t{:.1}\t{}\t{:.3}",
                                    devoiced.is_some() as u8,
                                    devoiced.map_or(String::new(), |a| format!("{a:.2}")),
                                    if tongue_none { "none" } else { "target" },
                                    d * 1000.0,
                                    r.open_ms,
                                    r.voiced.map_or(String::new(), |x| format!("{x:.3}")),
                                    r.amin,
                                    r.noisy,
                                    r.n_glot,
                                    r.n_constr,
                                    r.n_teeth,
                                    r.c_constr,
                                    r.c_teeth,
                                    r.hf_db,
                                    r.hf_rel_db.map_or(String::new(), |x| format!("{x:.1}")),
                                    r.hole,
                                    r.pos,
                                    r.ag,
                                    r.hf_abs_db,
                                    r.lo_abs_db,
                                    r.noise_end_ms.map_or(String::new(), |x| format!("{x:.1}")),
                                    r.tb_y
                                ));
                            }
                        }
                    }
                    lines
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for l in rows.into_iter().flatten() {
        println!("{l}");
    }
}

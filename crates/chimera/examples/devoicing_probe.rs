//! Research probe: when does a vowel between voiceless consonants lose its voicing?
//!
//! Two axes, set on the target vowel (the second vowel of each word, so it has a vowel with a
//! posture on both sides). Larynx: `overlap` (only what the neighbours' glottal-abduction gestures
//! leave) or `active` (the vowel's own opening, `Segment::devoiced`). Tongue: `target` or `none`
//! (`Segment::targetless`).
//!
//! Measured over the middle half of the target vowel (the transitions into and out of the
//! consonants left out):
//! - `gate`: the glottis model's own voicing gate (0..1). It is the engine's rule (folds close
//!   enough, pressure high enough), so it says what the model decided, not what the sound is.
//! - `aper`: YIN's aperiodicity of the output in that window (0 periodic .. 1 noise); empty when
//!   the window is too short to hold two periods at 80 Hz.
//! - `narrow`: the tract's narrowest area there (cm²), for the tongue's height.
//!
//! The voicing gate starts to fall at a static abduction of 0.08 cm² and is closed at 0.22.
//! Run: `cargo run --release -q -p chimera --example devoicing_probe`
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::phonetics::Segment;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, is_vowel, plan_utterance};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;
const MIN_F0: f64 = 80.0;

struct Row {
    dur_ms: f64,
    gate: f64,
    aper: Option<f64>,
    abd_max: f64,
    narrow: f64,
}

/// The index of the second vowel of the input (the target).
fn target(input: &[Segment]) -> Option<usize> {
    (0..input.len()).filter(|&k| is_vowel(&input[k].ipa, "male".into())).nth(1)
}

fn probe(ipa: &str, rate: f64, active: bool, targetless: bool) -> Option<Row> {
    let mut input = tokenize_ipa(ipa);
    let k = target(&input)?;
    input[k].devoiced = active.then_some(true);
    input[k].targetless = targetless.then_some(true);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(120.0), rate: Some(rate), ..Default::default() });
    // no diphthongs in these words, so the plan's segments line up with the input's
    assert_eq!(plan.segments.len(), input.len(), "{ipa}: the plan split a segment");
    assert_eq!(plan.segments[k].ipa, input[k].ipa, "{ipa}: measuring the wrong vowel");
    let mut v = HumanVoice::with_defaults(FS);
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let (mut gate, mut abd, mut narrow) = (vec![], vec![], vec![]);
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        gate.push(v.glottis.voicing);
        abd.push(v.glottis.params.abduction);
        let f = v.airway.frame.as_ref().expect("the airway publishes a frame every block");
        narrow.push(f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min));
    }
    let (t0, t1) = (plan.segments[k].t0, plan.segments[k].t1);
    let (m0, m1) = (t0 + 0.25 * (t1 - t0), t1 - 0.25 * (t1 - t0));
    let frame = |t: f64| ((t * FS) as usize / BLOCK).min(gate.len() - 1);
    let (a, b) = (frame(m0), frame(m1));
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let samples: Vec<f64> = out[(m0 * FS) as usize..(m1 * FS) as usize].iter().map(|&x| x as f64).collect();
    let aper = (samples.len() as f64 > 2.0 * FS / MIN_F0).then(|| yin_pitch(&samples, FS, YinOpts { min_f0: Some(MIN_F0), max_f0: Some(400.0), threshold: None }).aperiodicity);
    Some(Row {
        dur_ms: (t1 - t0) * 1000.0,
        gate: mean(&gate[a..=b]),
        aper,
        abd_max: abd[a..=b].iter().cloned().fold(0.0, f64::max),
        narrow: narrow[a..=b].iter().cloned().fold(f64::MAX, f64::min),
    })
}

fn main() {
    // the target vowel between two voiceless consonants (devoicing context) or with a voiced
    // neighbour (control); stops unaspirated (closer to Japanese than English's rule), the target
    // stressed
    let words = [
        ("aˈk˭it˭a", "stop_stop /i/"),
        ("aˈk˭ɯt˭a", "stop_stop /ɯ/"),
        ("aˈɕit˭a", "fric_stop /i/"),
        ("aˈsɯt˭a", "fric_stop /ɯ/"),
        ("aˈɸɯt˭a", "fric_stop /ɯ/ (ɸ)"),
        ("aˈɕiɕa", "fric_fric /i/"),
        ("aˈsɯsa", "fric_fric /ɯ/"),
        ("aˈk˭at˭a", "stop_stop /a/ (low)"),
        ("aˈk˭ida", "control voiced C2"),
        ("aˈgida", "control voiced both"),
    ];
    println!("word\tcontext\tlarynx\ttongue\trate\tvowel_ms\tgate\taper\tabd_max\tnarrow_cm2");
    for (w, ctx) in words {
        for (active, targetless) in [(false, false), (false, true), (true, false), (true, true)] {
            for rate in [0.7, 1.0, 1.4, 1.8] {
                let (l, t) = (if active { "active" } else { "overlap" }, if targetless { "none" } else { "target" });
                match probe(w, rate, active, targetless) {
                    Some(r) => {
                        let aper = r.aper.map_or(String::new(), |x| format!("{x:.2}"));
                        println!("{w}\t{ctx}\t{l}\t{t}\t{rate}\t{:.0}\t{:.2}\t{aper}\t{:.3}\t{:.3}", r.dur_ms, r.gate, r.abd_max, r.narrow);
                    }
                    None => println!("{w}\t{ctx}\t{l}\t{t}\t{rate}\tno target vowel"),
                }
            }
        }
    }
}

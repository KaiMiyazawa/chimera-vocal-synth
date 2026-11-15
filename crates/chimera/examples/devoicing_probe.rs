//! Research probe: when does a vowel between voiceless consonants lose its voicing?
//!
//! Two axes, set on the first inner vowel of each word. Larynx: `overlap` (only what the
//! neighbours' glottal-abduction gestures leave) or `active` (the vowel's own opening,
//! `Segment::devoiced`). Tongue: `target` or `none` (`Segment::targetless`).
//!
//! For each word, condition and speaking rate it renders the plan and reports, over that vowel,
//! the smoothed voicing gate (0..1), the glottal abduction the score asked for, and the tract's
//! narrowest area.  Run: `cargo run --release -q -p chimera --example devoicing_probe`
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;

struct Row {
    dur_ms: f64,
    voiced_ms: f64,
    voicing_mean: f64,
    abd_min: f64,
    abd_max: f64,
    narrow_min: f64,
}

/// `active`: the vowel has its own glottal opening (the laryngeal axis: overlap only / active).
/// `targetless`: the vowel has no tongue target (the lingual axis: target / none).
fn probe(ipa: &str, rate: f64, active: bool, targetless: bool) -> Option<Row> {
    let mut v = HumanVoice::with_defaults(FS);
    let mut input = tokenize_ipa(ipa);
    let k0 = (1..input.len().saturating_sub(1)).find(|&k| chimera::phonetics::planner::is_vowel(&input[k].ipa, "male".into()))?;
    input[k0].devoiced = active.then_some(true);
    input[k0].targetless = targetless.then_some(true);
    let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(120.0), rate: Some(rate), ..Default::default() });
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let (mut voicing, mut abd, mut narrow) = (vec![], vec![], vec![]);
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        voicing.push(v.glottis.voicing);
        abd.push(v.glottis.params.abduction);
        let f = v.airway.frame.as_ref().unwrap();
        narrow.push(f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min));
    }
    let segs = &plan.segments;
    let k = (0..segs.len()).find(|&k| segs[k].kind == "vowel" && k > 0 && k + 1 < segs.len())?;
    let frame = |t: f64| ((t * FS) as usize / BLOCK).min(voicing.len() - 1);
    let (a, b) = (frame(segs[k].t0), frame(segs[k].t1));
    let vs = &voicing[a..=b];
    let dt_ms = BLOCK as f64 / FS * 1000.0;
    Some(Row {
        dur_ms: (segs[k].t1 - segs[k].t0) * 1000.0,
        voiced_ms: vs.iter().filter(|&&x| x > 0.5).count() as f64 * dt_ms,
        voicing_mean: vs.iter().sum::<f64>() / vs.len() as f64,
        abd_min: abd[a..=b].iter().cloned().fold(f64::MAX, f64::min),
        abd_max: abd[a..=b].iter().cloned().fold(0.0, f64::max),
        narrow_min: narrow[a..=b].iter().cloned().fold(f64::MAX, f64::min),
    })
}

fn main() {
    // first inner vowel: between two voiceless consonants (devoicing context) or with a voiced
    // neighbour (control). Stops unaspirated (closer to Japanese than English's rule).
    let words = [
        ("ˈk˭it˭a", "stop_stop /i/"),
        ("ˈk˭ɯt˭a", "stop_stop /ɯ/"),
        ("ˈɕit˭a", "fric_stop /i/"),
        ("ˈsɯt˭a", "fric_stop /ɯ/"),
        ("ˈɸɯt˭a", "fric_stop /ɯ/ (ɸ)"),
        ("ˈɕiɕa", "fric_fric /i/"),
        ("ˈsɯsa", "fric_fric /ɯ/"),
        ("ˈk˭at˭a", "stop_stop /a/ (low)"),
        ("ˈk˭ida", "control voiced C2"),
        ("ˈgida", "control voiced both"),
    ];
    println!("word\tcontext\tlarynx\ttongue\trate\tvowel_ms\tvoiced_ms\tvoicing_mean\tabd_min\tabd_max\tnarrow_min_cm2");
    for (w, ctx) in words {
        for (active, targetless) in [(false, false), (false, true), (true, false), (true, true)] {
            for rate in [0.7, 1.0, 1.4, 1.8] {
                let (l, t) = (if active { "active" } else { "overlap" }, if targetless { "none" } else { "target" });
                match probe(w, rate, active, targetless) {
                    Some(r) => println!("{w}\t{ctx}\t{l}\t{t}\t{rate}\t{:.0}\t{:.0}\t{:.2}\t{:.3}\t{:.3}\t{:.3}", r.dur_ms, r.voiced_ms, r.voicing_mean, r.abd_min, r.abd_max, r.narrow_min),
                    None => println!("{w}\t{ctx}\t{l}\t{t}\t{rate}\tno inner vowel"),
                }
            }
        }
    }
}

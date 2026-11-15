//! A devoiced vowel (IPA ring below) is the front end's call: the folds are held apart through it
//! by a gesture of its own. A targetless vowel has no tongue gesture. Unmarked speech is unchanged.
use chimera::control::score::ControlKey;
use chimera::phonetics::ipa::{skipped_ipa, tokenize_ipa};
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::{Plan, Segment};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;

fn plan(segs: &[Segment]) -> Plan {
    plan_utterance(segs, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(120.0), ..Default::default() })
}

/// Mean of the voicing gate (0..1) over the segment at `k`.
fn voicing(p: &Plan, k: usize) -> f64 {
    let mut v = HumanVoice::with_defaults(FS);
    v.schedule(&p.gestures);
    let n = (FS * (p.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let mut gate = vec![];
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        gate.push(v.glottis.voicing);
    }
    let frame = |t: f64| ((t * FS) as usize / BLOCK).min(gate.len() - 1);
    let s = &gate[frame(p.segments[k].t0)..=frame(p.segments[k].t1)];
    s.iter().sum::<f64>() / s.len() as f64
}

#[test]
fn the_ring_below_marks_a_vowel_devoiced() {
    let d: Vec<Option<bool>> = tokenize_ipa("ki\u{325}ta kiː\u{325}ta kita").into_iter().filter(|s| s.ipa == "i").map(|s| s.devoiced).collect();
    assert_eq!(d, [Some(true), Some(true), None]);
    // read on a vowel, so no longer reported there; still unread on a consonant
    let r = skipped_ipa("ki\u{325}ta n\u{325}a");
    assert_eq!(r.ignored_marks, vec!["n\u{325}".to_string()]);
}

#[test]
fn a_devoiced_vowel_adds_one_gesture_and_changes_no_other() {
    let (plain, marked) = (plan(&tokenize_ipa("ˈk˭it˭a")), plan(&tokenize_ipa("ˈk˭i\u{325}t˭a")));
    let own: Vec<_> = marked.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "devoiced")).collect();
    assert_eq!(own.len(), 1);
    assert_eq!(own[0].targets.get(ControlKey::ABDUCTION), Some(0.6));
    let rest = |p: &Plan| serde_json::to_string(&p.gestures.iter().filter(|g| g.label.as_ref().is_none_or(|l| l.as_str() != "devoiced")).collect::<Vec<_>>()).unwrap();
    assert_eq!(rest(&plain), rest(&marked));
    assert_eq!(plain.segments, marked.segments);
}

#[test]
fn a_devoiced_vowel_is_voiceless_where_the_plain_one_is_voiced() {
    // between unaspirated stops the neighbours' openings alone leave the vowel mostly voiced
    let plain = voicing(&plan(&tokenize_ipa("ˈk˭it˭a")), 1);
    let marked = voicing(&plan(&tokenize_ipa("ˈk˭i\u{325}t˭a")), 1);
    assert!(plain > 0.5, "plain /i/ between stops: voicing {plain:.2}");
    assert!(marked < 0.05, "devoiced /i/ between stops: voicing {marked:.2}");
}

#[test]
fn a_targetless_vowel_has_no_tongue_gesture_and_keeps_its_time() {
    let mut segs = tokenize_ipa("ˈsɯsa");
    let plain = plan(&segs);
    segs[1].targetless = Some(true);
    let bare = plan(&segs);
    let has = |p: &Plan| p.gestures.iter().any(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "ɯ"));
    assert!(has(&plain) && !has(&bare));
    assert_eq!(plain.segments, bare.segments);
    assert_eq!(plain.gestures.len(), bare.gestures.len() + 1);
}

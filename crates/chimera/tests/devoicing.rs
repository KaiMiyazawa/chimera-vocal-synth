//! A devoiced vowel (IPA ring below) is the front end's call: the folds are held apart through it
//! by a gesture of its own. A targetless vowel has no tongue gesture. Unmarked speech is unchanged.
use chimera::control::score::ControlKey;
use chimera::control::score::Gesture;
use chimera::phonetics::ipa::{skipped_ipa, tokenize_ipa};
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::sing::{NoteOnOptions, SongOptions, SungNote, plan_note_on, plan_song};
use chimera::phonetics::{Plan, Segment, Syllable};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;

fn labelled<'a>(gs: &'a [Gesture], label: &str) -> Vec<&'a Gesture> {
    gs.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == label)).collect()
}

/// Gestures as JSON: unset targets are NaN, so `==` on gestures never holds; the JSON compares
/// what is set
fn json(gs: &[&Gesture]) -> String {
    serde_json::to_string(gs).unwrap()
}

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
    // NFC composes a + ring below into one letter (ḁ): read as the vowel and the mark
    let a: Vec<(String, Option<bool>)> = tokenize_ipa("ka\u{325}ta").into_iter().map(|s| (s.ipa, s.devoiced)).collect();
    assert_eq!(a, [("k".into(), None), ("a".into(), Some(true)), ("t".into(), None), ("a".into(), None)]);
    // read on a vowel, so no longer reported there; still unread on a consonant
    let r = skipped_ipa("ki\u{325}ta n\u{325}a");
    assert_eq!(r.ignored_marks, vec!["n\u{325}".to_string()]);
}

#[test]
fn a_devoiced_vowel_adds_one_gesture_and_changes_no_other() {
    let (plain, marked) = (plan(&tokenize_ipa("ˈk˭it˭a")), plan(&tokenize_ipa("ˈk˭i\u{325}t˭a")));
    let own = labelled(&marked.gestures, "devoiced");
    assert_eq!(own.len(), 1);
    assert_eq!(own[0].targets.get(ControlKey::ABDUCTION), Some(0.6));
    let rest = |p: &Plan| json(&p.gestures.iter().filter(|g| g.label.as_ref().is_none_or(|l| l.as_str() != "devoiced")).collect::<Vec<_>>());
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
fn a_targetless_vowel_has_no_tongue_gesture_and_keeps_its_time_and_pitch() {
    let mut segs = tokenize_ipa("aˈsɯsa");
    let plain = plan(&segs);
    segs[2].targetless = Some(true);
    let bare = plan(&segs);
    assert_eq!(labelled(&plain.gestures, "ɯ").len(), 1);
    assert!(labelled(&bare.gestures, "ɯ").is_empty());
    assert_eq!(plain.segments, bare.segments);
    // the pitch gestures are the same: the vowel keeps its pitch target
    assert_eq!(json(&labelled(&plain.gestures, "pitch")), json(&labelled(&bare.gestures, "pitch")));
    // the vowels either side span it: the first lets go after it starts, the last ramps in
    // before it ends
    let (u0, u1) = (bare.segments[2].t0, bare.segments[2].t1);
    let a = labelled(&bare.gestures, "a");
    assert_eq!(a.len(), 2);
    assert!(a[0].end() > u0, "the first /a/ lets go at {:.3}, before /ɯ/ at {u0:.3}", a[0].end());
    assert!(a[1].t0 - a[1].ramp_in < u1, "the last /a/ ramps in from {:.3}, after /ɯ/ ends at {u1:.3}", a[1].t0 - a[1].ramp_in);
}

#[test]
fn sung_vowels_keep_their_devoicing_and_targetlessness() {
    let syl = |nucleus: Segment| Syllable { text: "x".into(), onset: vec![Segment::new("s")], nucleus, coda: vec![], melisma: None, word_end: true, phrase_end: false, ..Default::default() };
    let song = |nucleus: Segment| plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: syl(nucleus) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0 }).gestures;
    let live = |nucleus: Segment| plan_note_on(&syl(nucleus), 57.0, &NoteOnOptions { body: "male".into(), vibrato: 0.0, tag: 1.0, legato_from: None });
    let (u, u_dev, u_bare) = (Segment::new("ɯ"), Segment { devoiced: Some(true), ..Segment::new("ɯ") }, Segment { targetless: Some(true), ..Segment::new("ɯ") });
    for gs in [song(u_dev.clone()), live(u_dev)] {
        assert_eq!(labelled(&gs, "devoiced").len(), 1);
    }
    for gs in [song(u.clone()), live(u.clone())] {
        assert!(labelled(&gs, "devoiced").is_empty());
        assert!(!labelled(&gs, "ɯ").is_empty());
    }
    for gs in [song(u_bare.clone()), live(u_bare)] {
        assert!(labelled(&gs, "ɯ").is_empty());
    }
}

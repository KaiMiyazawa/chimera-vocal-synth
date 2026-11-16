//! A devoiced vowel (IPA ring below) is the front end's call: the folds are held apart through it
//! by a gesture of its own. A targetless vowel has no tongue gesture. Unmarked speech is unchanged.
use chimera::control::score::ControlKey;
use chimera::control::score::Gesture;
use chimera::phonetics::ipa::{skipped_ipa, tokenize_ipa};
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::sing::{NoteOnOptions, SongOptions, SungNote, plan_note_on, plan_song};
use chimera::phonetics::{Plan, Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

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
    voicing_with(p, k, false, 8.0, 1.0)
}

/// The same with the opening-dependent threshold on or off, and the breath (cmH₂O) and gate.
fn voicing_with(p: &Plan, k: usize, opening_threshold: bool, breath: f64, gate: f64) -> f64 {
    let mut v = HumanVoice::with_defaults(FS);
    v.set_opening_threshold(opening_threshold);
    v.set_source(&SourcePatch { pressure: Some(breath), ..Default::default() });
    v.set_gate(gate);
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

#[test]
fn a_devoiced_diphthong_holds_the_folds_apart_through_its_glide() {
    let p = plan(&tokenize_ipa("ˈk˭aɪ\u{325}t˭a"));
    let own = labelled(&p.gestures, "devoiced");
    assert_eq!(own.len(), 1, "one opening for the whole diphthong");
    // the diphthong is planned as two halves; the opening runs to the end of the second
    let (first, glide) = (&p.segments[1], &p.segments[2]);
    assert_eq!((first.kind.as_str(), glide.kind.as_str()), ("vowel", "vowel"));
    assert_eq!(own[0].t0, first.t0);
    assert_eq!(own[0].t1, glide.t1);
}

#[test]
fn a_targetless_last_vowel_leaves_the_one_before_it_as_it_was() {
    let mut segs = tokenize_ipa("ˈsasɯ");
    let plain = plan(&segs);
    segs[3].targetless = Some(true);
    let bare = plan(&segs);
    assert!(labelled(&bare.gestures, "ɯ").is_empty());
    // the first /a/ was already the last posed vowel's neighbour; with /ɯ/ gone it is the last
    // posed vowel, so it lets go as an utterance-final vowel would (ramp 0.4 × 80 ms + 10 ms)
    let a = labelled(&bare.gestures, "a");
    assert_eq!(a.len(), 1);
    assert!((a[0].ramp_out - (0.4 * 0.08 + 0.01)).abs() < 1e-12, "ramp out {}", a[0].ramp_out);
    assert_ne!(json(&labelled(&plain.gestures, "a")), json(&a));
}

/// Bug (f06b7b1): a targetless r-coloured vowel counted as having placed the tongue tip, so a
/// following stop behind a front closure was delayed although no tip gesture had been made.
#[test]
fn a_targetless_rhotic_vowel_does_not_pre_place_a_closure() {
    // /t/ has two gestures: the approach, which leads, and the closure, which starts at the
    // segment unless it is delayed
    let t0 = |segs: &[Segment]| labelled(&plan(segs).gestures, "t").iter().map(|g| g.t0).fold(f64::MIN, f64::max);
    let with_rhotic = tokenize_ipa("ˈɝpta");
    let mut bare_rhotic = with_rhotic.clone();
    bare_rhotic[0].targetless = Some(true);
    let plain_vowel = tokenize_ipa("ˈʌpta");
    let start_of = |segs: &[Segment]| plan(segs).segments[2].t0;
    // precondition: after a rhotic vowel the /t/ closure is delayed past its segment start (25 ms)
    assert!(t0(&with_rhotic) > start_of(&with_rhotic) + 0.02, "the rhotic vowel should pre-place the tip");
    // without its target it places nothing: no delay, as after /ʌ/
    assert!((t0(&bare_rhotic) - start_of(&bare_rhotic)).abs() < 1e-9);
    assert!((t0(&plain_vowel) - start_of(&plain_vowel)).abs() < 1e-9);
}

/// Both voicing models (the default area cutoff, and the opening-dependent threshold, a research
/// option) keep breathy voice and stop the folds for /h/ and voiceless obstruents at the default
/// breath. A threshold rising in proportion to the opening (ea59a5a) silenced breathy vowels
/// (0.93 → 0.06); these bars are the area cutoff's own values (71381df).
#[test]
fn both_voicing_models_keep_breathy_voice_and_stop_voiceless_sounds() {
    for on in [false, true] {
        let mid = |ipa: &str| voicing_with(&plan(&tokenize_ipa(ipa)), 1, on, 8.0, 1.0);
        for (ipa, what, at_least) in [("ˈbada", "modal /a/", 0.95), ("ˈba\u{324}da", "breathy /a/", 0.9), ("ˈbɦada", "/ɦ/", 0.6)] {
            let g = mid(ipa);
            assert!(g > at_least, "opening threshold {on}: {what} in {ipa}: voicing {g:.2}");
        }
        for (ipa, what) in [("ˈaha", "/h/"), ("ˈasa", "/s/"), ("ˈaɸa", "/ɸ/"), ("ˈaɕa", "/ɕ/"), ("ˈapa", "/p/")] {
            let g = mid(ipa);
            assert!(g < 0.1, "opening threshold {on}: {what} in {ipa}: voicing {g:.2}");
        }
    }
}

/// The default model keeps voiceless sounds voiceless at the loudest breath the controls give
/// (Breath 22 × gate 1.5); the opening-dependent threshold does not (it is a research option).
#[test]
fn the_default_voicing_model_keeps_loud_voiceless_sounds_voiceless() {
    for (ipa, what) in [("ˈaha", "/h/"), ("ˈasa", "/s/"), ("ˈaɸa", "/ɸ/"), ("ˈaɕa", "/ɕ/")] {
        let p = plan(&tokenize_ipa(ipa));
        let loud = voicing_with(&p, 1, false, 22.0, 1.5);
        assert!(loud < 0.05, "default model: {what} at Breath 22 × 1.5: voicing {loud:.2}");
    }
    let h = voicing_with(&plan(&tokenize_ipa("ˈaha")), 1, true, 22.0, 1.5);
    assert!(h > 0.5, "the research option's known limit moved: /h/ at Breath 22 × 1.5 now {h:.2}");
}

//! The two-mass vocal folds (`sources::two_mass`): they vibrate on their own, hold the requested
//! pitch, and pressed, normal and breathy voice — and a raised larynx — change how they vibrate.
use chimera::anatomy::human::{ArticKey, ArticPatch, BODY_FEMALE, BODY_MALE, HumanBody};
use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch, oversample_for};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;

struct Take {
    /// f0 measured from the output (Hz)
    f0: f64,
    /// open quotient of the folds' last cycles
    oq: f64,
    /// level of the 1st harmonic over the 2nd (dB) in the glottal flow
    h1h2: f64,
    /// output RMS
    rms: f64,
    /// high-frequency share of the output (dB): mean squared first difference over mean square
    hf: f64,
}

fn sing(body: HumanBody, fs: f64, f0: f64, tension: f64, breathiness: f64, larynx: f64) -> Take {
    let mut v = HumanVoice::new(fs, body, oversample_for(fs));
    v.set_source(&SourcePatch { f0: Some(f0), tension: Some(tension), breathiness: Some(breathiness), glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    v.set_articulation(&ArticPatch::default().with(ArticKey::Larynx, larynx), None);
    v.set_gate(1.0);
    let n = (1.2 * fs) as usize;
    let mut out = vec![0.0f32; n];
    let mut flow = Vec::with_capacity(n);
    let mut oq = Vec::new();
    for (i, c) in out.chunks_mut(1).enumerate() {
        v.render(c);
        if i > n / 2 {
            if i % 64 == 0 {
                oq.push(v.folds.open_quotient);
            }
            flow.push(v.airway.net.u[v.airway.glottis_port]);
        }
    }
    let x: Vec<f64> = out[n / 2..].iter().map(|s| *s as f64).collect();
    let rms = (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).sqrt();
    let d: f64 = x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let e: f64 = x.iter().map(|s| s * s).sum();
    // f0 from the autocorrelation's peak (simple, adequate for a steady tone)
    let f0m = autocorr_f0(&x, fs, f0);
    let h1h2 = harmonic_db(&flow, fs, f0m, 1) - harmonic_db(&flow, fs, f0m, 2);
    let oq = oq.iter().sum::<f64>() / oq.len().max(1) as f64;
    Take { f0: f0m, oq, h1h2, rms, hf: 10.0 * (d / e.max(1e-30)).log10() }
}

fn autocorr_f0(x: &[f64], fs: f64, guess: f64) -> f64 {
    let lo = (fs / (guess * 1.6)) as usize;
    let hi = (fs / (guess / 1.6)) as usize;
    let n = x.len() - hi;
    let mut best = (lo, f64::MIN);
    for lag in lo..hi {
        let r: f64 = (0..n).map(|i| x[i] * x[i + lag]).sum();
        if r > best.1 {
            best = (lag, r);
        }
    }
    // parabolic refinement
    let r = |lag: usize| (0..n).map(|i| x[i] * x[i + lag]).sum::<f64>();
    let (l, c) = (best.0, best.1);
    let (a, b) = (r(l - 1), r(l + 1));
    let off = 0.5 * (a - b) / (a - 2.0 * c + b);
    fs / (l as f64 + off)
}

fn harmonic_db(x: &[f64], fs: f64, f0: f64, k: usize) -> f64 {
    let w = std::f64::consts::TAU * f0 * k as f64 / fs;
    let n = x.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, s) in x.iter().enumerate() {
        let h = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n).cos();
        re += s * h * (w * i as f64).cos();
        im += s * h * (w * i as f64).sin();
    }
    10.0 * (re * re + im * im).max(1e-30).log10()
}

fn two_mass(body: HumanBody, fs: f64) -> HumanVoice {
    let mut v = HumanVoice::new(fs, body, oversample_for(fs));
    v.set_source(&SourcePatch { f0: Some(body_f0(body)), glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    v.set_gate(1.0);
    v
}

fn body_f0(body: HumanBody) -> f64 {
    if body.scale < 1.0 { 220.0 } else { 110.0 }
}

#[test]
fn the_folds_vibrate_at_the_requested_pitch() {
    // nothing draws the pulse: the folds must start on their own and the pitch loop must hold them
    // within a few cents across a singer's range, both bodies
    for (body, f0s) in [(BODY_MALE, [110.0, 220.0, 330.0]), (BODY_FEMALE, [220.0, 440.0, 660.0])] {
        for f0 in f0s {
            let r = sing(body, 48000.0, f0, 0.5, 0.15, 0.0);
            let cents = 1200.0 * (r.f0 / f0).log2();
            assert!(r.rms > 0.01, "{} at {f0} Hz: silent ({:.4})", body.name, r.rms);
            assert!(cents.abs() < 10.0, "{} at {f0} Hz: sings {:.1} Hz ({cents:.1} cents)", body.name, r.f0);
            assert!(r.oq > 0.5 && r.oq < 0.9, "{} at {f0} Hz: open quotient {:.2}", body.name, r.oq);
        }
    }
}

#[test]
fn pressed_normal_and_breathy_come_from_the_folds_setting() {
    // Tension sets the folds' rest gap. Breathy folds (apart) never close; pressed folds (pushed
    // together) stay closed longer, and the flow pulse loses its fundamental relative to the second
    // harmonic (H1-H2), as measured in real pressed and breathy voices (Holmberg et al. 1988, Hanson 1997).
    for (body, f0) in [(BODY_MALE, 110.0), (BODY_MALE, 220.0), (BODY_FEMALE, 220.0), (BODY_FEMALE, 440.0)] {
        let breathy = sing(body, 48000.0, f0, 0.0, 0.3, 0.0);
        let normal = sing(body, 48000.0, f0, 0.5, 0.15, 0.0);
        let pressed = sing(body, 48000.0, f0, 1.0, 0.0, 0.0);
        let who = format!("{} at {f0} Hz", body.name);
        println!("{who}: OQ {:.2} / {:.2} / {:.2}, H1-H2 {:.1} / {:.1} / {:.1} dB", breathy.oq, normal.oq, pressed.oq, breathy.h1h2, normal.h1h2, pressed.h1h2);
        assert!(pressed.rms > 0.01, "{who}: pressed folds stopped vibrating");
        assert!(breathy.oq > normal.oq + 0.1 && normal.oq > pressed.oq + 0.04, "{who}: open quotient");
        assert!(breathy.h1h2 > normal.h1h2 + 4.0 && normal.h1h2 > pressed.h1h2 + 1.0, "{who}: H1-H2");
    }
}

#[test]
fn a_raised_larynx_presses_the_folds() {
    // The raised-larynx fault is half throat (a shorter tract, moved resonances) and half source: the
    // folds press together (Sundberg & Askenfelt 1983). With the two-mass folds, raising the larynx
    // 0.8 cm closes the glottis for longer and lowers H1-H2; lowering it does the opposite.
    for (body, f0) in [(BODY_MALE, 110.0), (BODY_MALE, 220.0), (BODY_FEMALE, 220.0), (BODY_FEMALE, 440.0)] {
        let low = sing(body, 48000.0, f0, 0.5, 0.15, -0.8);
        let mid = sing(body, 48000.0, f0, 0.5, 0.15, 0.0);
        let high = sing(body, 48000.0, f0, 0.5, 0.15, 0.8);
        let who = format!("{} at {f0} Hz", body.name);
        println!("{who}: larynx -0.8/0/+0.8: OQ {:.2} / {:.2} / {:.2}, H1-H2 {:.1} / {:.1} / {:.1} dB", low.oq, mid.oq, high.oq, low.h1h2, mid.h1h2, high.h1h2);
        assert!(high.rms > 0.01, "{who}: raised larynx stopped the folds");
        assert!(high.oq < mid.oq - 0.03 && low.oq > mid.oq, "{who}: open quotient");
        assert!(high.h1h2 < mid.h1h2 - 1.0, "{who}: H1-H2");
        // and the pressed folds close more abruptly: a brighter voice (Gauffin & Sundberg 1989)
        assert!(high.hf > mid.hf + 0.5, "{who}: a raised larynx didn't brighten the folds ({:.1} against {:.1} dB)", high.hf, mid.hf);
    }
}

#[test]
fn the_folds_hold_at_any_device_rate() {
    // a 16 kHz headset to a 96 kHz interface: finite, voiced, on pitch
    for fs in [16000.0, 22050.0, 44100.0, 48000.0, 96000.0] {
        for body in [BODY_MALE, BODY_FEMALE] {
            let mut v = two_mass(body, fs);
            let mut out = vec![0.0f32; fs as usize];
            for c in out.chunks_mut(128) {
                v.render(c);
            }
            let tail = &out[out.len() / 2..];
            assert!(tail.iter().all(|x| x.is_finite()), "{} at {fs} Hz: not finite", body.name);
            let rms = (tail.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / tail.len() as f64).sqrt();
            assert!(rms > 0.01, "{} at {fs} Hz: silent ({rms:.4})", body.name);
            let x: Vec<f64> = tail.iter().map(|s| *s as f64).collect();
            let f = autocorr_f0(&x, fs, body_f0(body));
            let cents = 1200.0 * (f / body_f0(body)).log2();
            assert!(cents.abs() < 15.0, "{} at {fs} Hz: {f:.1} Hz", body.name);
        }
    }
}

#[test]
fn the_folds_do_not_depend_on_the_block_size() {
    let mut a = two_mass(BODY_MALE, 48000.0);
    let mut b = two_mass(BODY_MALE, 48000.0);
    let mut x = vec![0.0f32; 24000];
    let mut y = vec![0.0f32; 24000];
    for c in x.chunks_mut(128) {
        a.render(c);
    }
    for c in y.chunks_mut(37) {
        b.render(c);
    }
    assert!(x == y, "output differs with the block size");
}

#[test]
fn the_folds_do_not_allocate() {
    let mut v = two_mass(BODY_FEMALE, 48000.0);
    let mut out = vec![0.0f32; 128];
    v.render(&mut out);
    let before = ALLOCS.with(|n| n.get());
    for i in 0..400 {
        if i == 200 {
            v.set_articulation(&ArticPatch::default().with(ArticKey::Larynx, 1.0), None);
        }
        v.render(&mut out);
    }
    let n = ALLOCS.with(|n| n.get()) - before;
    assert_eq!(n, 0, "rendering with the two-mass folds allocated {n} times");
}

#[test]
fn the_folds_sing_about_as_loud_as_the_pulse() {
    // Heard by the owner (2026-11-14): the folds were 5 dB (male) to 11 dB (female) softer than the
    // parametric pulse at the same settings, which masked every comparison between them. The folds
    // must not take the pulse's small-body cut (it answers to the pulse's brightness).
    for (body, f0) in [(BODY_MALE, 110.0), (BODY_MALE, 220.0), (BODY_FEMALE, 220.0), (BODY_FEMALE, 440.0)] {
        let level = |glottis: GlottisModel| {
            let mut v = HumanVoice::new(48000.0, body, oversample_for(48000.0));
            v.set_source(&SourcePatch { f0: Some(f0), tension: Some(0.55), glottis: Some(glottis), ..Default::default() });
            v.set_gate(1.0);
            let mut out = vec![0.0f32; 48000];
            for c in out.chunks_mut(128) {
                v.render(c);
            }
            let t = &out[24000..];
            10.0 * (t.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / t.len() as f64).log10()
        };
        let d = level(GlottisModel::TwoMass) - level(GlottisModel::Parametric);
        println!("{} at {f0} Hz: folds {d:+.1} dB against the pulse", body.name);
        assert!(d.abs() < 3.0, "{} at {f0} Hz: folds {d:+.1} dB against the pulse", body.name);
    }
}

/// 2-5 kHz against 0-1 kHz in the voice's long-term spectrum (dB), and its level (dB).
fn colour(body: HumanBody, f0: f64, glottis: GlottisModel, bright: f64) -> (f64, f64) {
    let fs = 48000.0;
    let mut v = HumanVoice::new(fs, body, oversample_for(fs));
    v.set_source(&SourcePatch { f0: Some(f0), tension: Some(0.55), brightness: Some(bright), glottis: Some(glottis), ..Default::default() });
    v.set_gate(1.0);
    let n = 48000;
    let mut out = vec![0.0f32; n];
    for c in out.chunks_mut(128) {
        v.render(c);
    }
    let x: Vec<f64> = out[n / 2..].iter().map(|s| *s as f64).collect();
    let nfft = 4096;
    let (mut lo, mut hi) = (0.0, 0.0);
    let mut k = 0;
    while k + nfft <= x.len() {
        for b in 1..nfft / 2 {
            let f = b as f64 * fs / nfft as f64;
            if f < 1000.0 || (2000.0..5000.0).contains(&f) {
                let w = std::f64::consts::TAU * b as f64 / nfft as f64;
                let (mut re, mut im) = (0.0, 0.0);
                for (i, s) in x[k..k + nfft].iter().enumerate() {
                    let h = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / nfft as f64).cos();
                    re += s * h * (w * i as f64).cos();
                    im += s * h * (w * i as f64).sin();
                }
                if f < 1000.0 { lo += re * re + im * im } else { hi += re * re + im * im }
            }
        }
        k += nfft * 2;
    }
    let level = 10.0 * (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).log10();
    (10.0 * (hi / 3.0 / lo).log10(), level)
}

#[test]
fn brightness_brightens_the_folds_as_it_does_the_pulse() {
    // Heard by the owner (2026-11-14): the folds sounded darker than the pulse. The guides sing at
    // brightness 1, which only sharpened the parametric pulse's closure. Now it shapes the folds'
    // closure too: at its default nothing changes, and fully bright the folds' colour (2-5 kHz
    // against 0-1 kHz) lands near the pulse's.
    for (body, f0) in [(BODY_MALE, 147.0), (BODY_FEMALE, 294.0)] {
        let pulse = colour(body, f0, GlottisModel::Parametric, 1.0);
        let [dark, mid, bright] = [0.0, 0.25, 1.0].map(|b| colour(body, f0, GlottisModel::TwoMass, b));
        println!("{} at {f0} Hz: folds 0/0.25/1 {:.1} / {:.1} / {:.1} dB, pulse at 1 {:.1} dB; level {:.1} / {:.1} / {:.1}", body.name, dark.0, mid.0, bright.0, pulse.0, dark.1, mid.1, bright.1);
        assert!(dark.0 < mid.0 - 1.0 && mid.0 < bright.0 - 4.0, "{}: brightness doesn't order the folds' colour", body.name);
        assert!((bright.0 - pulse.0).abs() < 3.0, "{}: fully bright folds {:.1} dB, pulse {:.1} dB", body.name, bright.0, pulse.0);
        assert!((bright.1 - mid.1).abs() < 2.5, "{}: brightness changed the level by {:.1} dB", body.name, bright.1 - mid.1);
    }
}

#[test]
fn the_folds_leave_no_denormal_states() {
    // As the parametric source (tests/denormals.rs): once the breath is let go or the folds part,
    // every decaying state of the folds ends at zero, not among the slow denormal numbers.
    for disturb in [SourcePatch::default(), SourcePatch { voiced: Some(false), ..Default::default() }] {
        let mut v = two_mass(BODY_MALE, 48000.0);
        let mut buf = [0.0f32; 128];
        for _ in 0..375 {
            v.render(&mut buf);
        }
        if disturb.voiced.is_some() {
            v.set_source(&disturb);
        } else {
            v.set_gate(0.0);
        }
        for second in 1..=30 {
            for _ in 0..375 {
                v.render(&mut buf);
            }
            assert_eq!(v.folds.denormal_states(), 0, "{:?}: denormal fold states {second} s later", disturb.voiced);
            assert_eq!(v.denormal_states(), 0, "{:?}: denormal voice states {second} s later", disturb.voiced);
        }
    }
}

/// Periodicity of the output's second half at lags 2.5-20 ms (normalised autocorrelation peak).
fn periodicity(out: &[f32]) -> f64 {
    let x: Vec<f64> = out[out.len() / 2..].iter().map(|s| *s as f64).collect();
    let e: f64 = x.iter().map(|s| s * s).sum();
    if e <= 0.0 {
        return 0.0;
    }
    let mut best: f64 = 0.0;
    for lag in 120..960 {
        let r: f64 = (0..x.len() - lag).map(|i| x[i] * x[i + lag]).sum();
        best = best.max(r / e);
    }
    best
}

#[test]
fn a_whisper_stays_unvoiced_at_any_breath() {
    // Review finding: the folds ignored `voiced: false`, so a whisper at a firm breath vibrated
    // (pressure 15 cmH2O: voicing 1.00, a 110 Hz flow).
    for pressure in [8.0, 15.0, 20.0] {
        let mut v = HumanVoice::new(48000.0, BODY_MALE, oversample_for(48000.0));
        v.set_source(&SourcePatch { f0: Some(110.0), pressure: Some(pressure), voiced: Some(false), glottis: Some(GlottisModel::TwoMass), ..Default::default() });
        v.set_gate(1.0);
        let mut out = vec![0.0f32; 48000];
        for c in out.chunks_mut(128) {
            v.render(c);
        }
        let p = periodicity(&out);
        println!("whisper at {pressure} cmH2O: voicing {:.2}, periodicity {p:.2}", v.voicing());
        assert!(v.voicing() < 0.1 && p < 0.5, "whisper at {pressure} cmH2O voiced: voicing {:.2}, periodicity {p:.2}", v.voicing());
    }
}

#[test]
fn a_voiceless_abduction_stops_the_folds() {
    // The planner abducts the folds 0.22-0.36 cm² for voiceless consonants; held, the folds stop
    // even at a firm breath, and start again when it ends.
    use chimera::control::score::{ControlKey, Gesture, Targets};
    let mut v = HumanVoice::new(48000.0, BODY_MALE, oversample_for(48000.0));
    v.set_source(&SourcePatch { f0: Some(110.0), pressure: Some(15.0), glottis: Some(GlottisModel::TwoMass), ..Default::default() });
    v.set_gate(1.0);
    v.schedule(&[Gesture::new(0.5, 1.5, 0.03, 0.03, 1.0, Targets::default().with(ControlKey::ABDUCTION, 0.3))]);
    let mut buf = [0.0f32; 128];
    let mut at = |v: &mut HumanVoice, t: f64| {
        while v.clock < t {
            v.render(&mut buf);
        }
        v.voicing()
    };
    let before = at(&mut v, 0.45);
    let during = at(&mut v, 1.45);
    let after = at(&mut v, 2.2);
    println!("voicing before / during / after the abduction: {before:.2} / {during:.2} / {after:.2}");
    assert!(before > 0.7 && during < 0.1 && after > 0.7, "{before:.2} / {during:.2} / {after:.2}");
}

#[test]
fn voicing_reads_full_at_high_pitch() {
    // Review finding: the voicing telemetry judged the swing in absolute cm, which shrinks with
    // pitch; a child at 600 Hz read 0.28 while vibrating fully (the scope locks above 0.2).
    use chimera::anatomy::human::BODY_CHILD;
    for (body, f0) in [(BODY_MALE, 110.0), (BODY_MALE, 440.0), (BODY_FEMALE, 880.0), (BODY_CHILD, 600.0)] {
        let mut v = HumanVoice::new(48000.0, body, oversample_for(48000.0));
        v.set_source(&SourcePatch { f0: Some(f0), glottis: Some(GlottisModel::TwoMass), ..Default::default() });
        v.set_gate(1.0);
        let mut out = vec![0.0f32; 48000];
        for c in out.chunks_mut(128) {
            v.render(c);
        }
        println!("{} at {f0} Hz: voicing {:.2}", body.name, v.voicing());
        assert!(v.voicing() > 0.7, "{} at {f0} Hz: voicing {:.2}", body.name, v.voicing());
    }
}

mod research_inputs {
    use chimera::phonetics::ipa::tokenize_ipa;
    use chimera::phonetics::planner::{PlanOptions, plan_utterance};
    use chimera::voice::human_voice::{GlottisModel, HumanVoice, SourcePatch};

    const FS: f64 = 48000.0;

    fn render(ipa: &str, patch: SourcePatch, setup: impl Fn(&mut HumanVoice)) -> Vec<f32> {
        let plan = plan_utterance(&tokenize_ipa(ipa), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
        let mut v = HumanVoice::with_defaults(FS);
        v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..patch });
        setup(&mut v);
        v.schedule(&plan.gestures);
        let mut out = vec![0.0f32; (FS * (plan.end + 0.1)) as usize];
        for c in out.chunks_mut(128) {
            v.render(c);
        }
        out
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&y| (y as f64) * (y as f64)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// Bug (upstream review of PR 103): the breathiness setting parted the folds along their length,
    /// so breathy 1.0 whispered (voicing 0.00, 8 dB below the default breathiness). Only abduction
    /// beyond the voice's own parts them now: breathy 1.0 voices about as much as the default.
    #[test]
    fn a_breathy_voice_stays_voiced_with_the_folds() {
        let voicing = |breathiness: f64| {
            let plan = plan_utterance(&tokenize_ipa("ɑ"), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
            let mut v = HumanVoice::with_defaults(FS);
            v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), breathiness: Some(breathiness), ..Default::default() });
            v.schedule(&plan.gestures);
            let mut out = vec![0.0f32; 128];
            let mut vs = vec![];
            for k in 0..((FS * 0.6) as usize / 128) {
                v.render(&mut out);
                if k > 112 {
                    vs.push(v.voicing());
                }
            }
            vs.iter().sum::<f64>() / vs.len() as f64
        };
        let (modal, breathy) = (voicing(0.15), voicing(1.0));
        assert!(breathy > 0.8 * modal, "breathy 1.0 with the folds: voicing {breathy:.2}, default {modal:.2}");
    }

    /// Bug (upstream review of PR 103): a glottal stop did not stop the folds (they got louder).
    #[test]
    fn a_glottal_stop_silences_the_folds() {
        let out = render("ɑʔɑ", SourcePatch::default(), |_| {});
        let plan = plan_utterance(&tokenize_ipa("ɑʔɑ"), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
        let s = &plan.segments[1];
        let cut = |a: f64, b: f64| &out[(a * FS) as usize..(b * FS) as usize];
        let mid = 0.5 * (s.t0 + s.t1);
        let stop = rms(cut(mid - 0.01, mid + 0.01));
        let vowel = rms(cut(plan.segments[0].t0 + 0.05, plan.segments[0].t1 - 0.03));
        assert!(stop < 0.2 * vowel, "ʔ with the folds: {stop:.2e} against the vowel's {vowel:.2e}");
    }

    /// Bug (upstream review of PR 103): the press read the vowel targets' larynx heights, so the
    /// folds' setting changed vowel by vowel.
    #[test]
    fn the_folds_are_pressed_alike_for_every_vowel() {
        let rest = |ipa: &str| {
            let plan = plan_utterance(&tokenize_ipa(ipa), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
            let mut v = HumanVoice::with_defaults(FS);
            v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
            v.schedule(&plan.gestures);
            let mut out = vec![0.0f32; 128];
            for _ in 0..((0.25 * FS) as usize / 128) {
                v.render(&mut out);
            }
            v.folds.folds.rest
        };
        assert_eq!(rest("i"), rest("ɑ"));
    }

    /// The research inputs at their defaults leave the folds as they were, sample for sample.
    #[test]
    fn the_research_inputs_at_their_defaults_change_nothing() {
        let plain = render("ɑpɑ", SourcePatch::default(), |_| {});
        let explicit = render("ɑpɑ", SourcePatch::default(), |v| {
            v.folds.stiffness = 0.0;
            v.folds.trim_loop = true;
            v.folds.stiffness_keeps_gap = false;
            v.folds.contact_stiffness = 3.0;
            v.folds.contact_damping = 1.0;
            v.folds.closed_push_lower = true;
            v.folds.closed_push_upper = true;
            v.folds.spring_cubic = 0.0;
            v.folds.contact_cubic = 0.0;
        });
        assert_eq!(plain, explicit);
    }

    /// A small abduction held while the folds vibrate (trim loop frozen) lowers their F0: the
    /// cycles stop closing, and with them goes the lift that the subglottal push on the shut lower
    /// mass gives each cycle. Without that push the same opening raises F0 instead.
    #[test]
    fn the_push_on_the_shut_lower_mass_is_what_an_opening_takes_from_f0() {
        use chimera::control::score::{ControlKey, Gesture, Targets};
        use chimera::phonetics::Intonation;
        let change = |push: bool| {
            let mut input = tokenize_ipa("ɑ");
            input[0].dur = Some(1.2);
            input[0].stress = Some(0);
            let plan = plan_utterance(&input, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), accent: Some(0.0), intonation: Some(Intonation::Flat), ..Default::default() });
            let mut v = HumanVoice::with_defaults(FS);
            v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), roughness: Some(0.0), vibrato_depth: Some(0.0), drift: Some(0.0), ..Default::default() });
            v.folds.closed_push_lower = push;
            v.schedule(&plan.gestures);
            v.schedule(&[Gesture::new(0.7, 1.1, 0.02, 0.02, 6.0, Targets::default().with(ControlKey::ABDUCTION, 0.03))]);
            let mut out = [0.0f32; 1];
            let mut x = vec![];
            for i in 0..(1.1 * FS) as usize {
                if i == (0.6 * FS) as usize {
                    v.folds.trim_loop = false;
                }
                v.render(&mut out);
                x.push(v.folds.displacement());
            }
            // mean F0 from upward crossings of the window's mean displacement
            let f0 = |a: f64, b: f64| {
                let w = &x[(a * FS) as usize..(b * FS) as usize];
                let m = w.iter().sum::<f64>() / w.len() as f64;
                let ups: Vec<usize> = (1..w.len()).filter(|&i| w[i - 1] < m && w[i] >= m).collect();
                (ups.len() - 1) as f64 * FS / (ups[ups.len() - 1] - ups[0]) as f64
            };
            f0(0.85, 1.05) - f0(0.4, 0.6)
        };
        let (with, without) = (change(true), change(false));
        assert!(with < -5.0, "with the push, an opening of 0.03 cm² moves F0 by {with:+.1} Hz");
        assert!(without > 0.0, "without it, by {without:+.1} Hz");
    }

    /// Stiffening the folds (the trim loop off, so nothing pulls F0 back) raises F0.
    #[test]
    fn stiffer_folds_vibrate_faster() {
        let cycles = |stiff: f64| {
            let plan = plan_utterance(&tokenize_ipa("ɑ"), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
            let mut v = HumanVoice::with_defaults(FS);
            v.set_source(&SourcePatch { glottis: Some(GlottisModel::TwoMass), ..Default::default() });
            v.folds.trim_loop = false;
            v.folds.stiffness = stiff;
            v.schedule(&plan.gestures);
            let mut out = [0.0f32; 1];
            let (mut ups, mut last) = (0, 0.0);
            for i in 0..(0.6 * FS) as usize {
                v.render(&mut out);
                let a = v.folds.area;
                if i as f64 > 0.3 * FS && last < 0.05 && a >= 0.05 {
                    ups += 1;
                }
                last = a;
            }
            ups
        };
        let (soft, stiff) = (cycles(0.0), cycles(0.3));
        assert!(stiff > soft, "cycles in 0.3 s: {soft} at stiffness 0, {stiff} at 0.3");
    }
}

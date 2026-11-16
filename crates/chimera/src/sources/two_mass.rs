//! Self-oscillating vocal folds: a two-mass model with a triangular glottis, after Birkholz,
//! Kröger & Neuschaefer-Rube (2011), "Synthesis of breathy, normal, and pressed phonation using a
//! two-mass model with a triangular glottis" (Interspeech), as in VocalTractLab's
//! `TriangularGlottis`. The mechanics are Ishizaka & Flanagan's (1972) two masses per fold with the
//! flow-separation pressures of Steinecke & Herzel (1995).
//!
//! The parametric source (`glottis.rs`) draws the glottal area as a fixed pulse shape. Here the
//! area comes out of the folds' motion, driven by the transglottal pressure the airway computes
//! each sample, so the pulse shape answers to how the folds are set:
//!
//!  - **Adduction** (`rest`): how far each fold rests from the midline (cm, negative = pressed
//!    into the other fold). Pressed folds collide early and stay in contact longer (a short open
//!    phase and an abrupt closure: a pressed voice); folds at rest apart never close (breathy).
//!  - **Triangular glottis**: the arytenoids (abduction) open the posterior end, so the gap grows
//!    linearly from the anterior commissure to the vocal processes. Vibration rides on that
//!    triangle: a partly abducted glottis closes in front and leaks behind, as breathy voice does.
//!  - **Pitch** sets the folds' stiffness and mass by Ishizaka & Flanagan's tension factor Q. The
//!    folds' own frequency also moves with pressure and adduction, so a loop trims Q (by 35 % of the error, in octaves, each clear cycle: a time constant of two to
//!    three cycles) until
//!    the measured cycle rate matches the requested f0 (a singer listening to their own pitch).
//!
//! Real-time contract: no allocation, transcendental math through `chimera::math`, one state
//! update per sample at the airway's (oversampled, > 60 kHz) rate.

use crate::math::{Drift, Rng, TAU, clamp, exp, js_max, js_min, log2, pow, sin, sqrt, tanh};
use crate::sources::glottis::GlottisParams;

/// Lower mass (Ishizaka & Flanagan 1972, adult male, at Q = 1).
const M1: f64 = 0.125; // g
const M2: f64 = 0.025;
const K1: f64 = 80_000.0; // dyn/cm
const K2: f64 = 8_000.0;
const KC: f64 = 25_000.0; // coupling spring between the masses
/// thickness of each mass (cm)
const D1: f64 = 0.25;
const D2: f64 = 0.05;
/// damping ratios, open and in contact
const Z1_OPEN: f64 = 0.1;
const Z1_CLOSED: f64 = 1.1;
const Z2_OPEN: f64 = 0.6;
const Z2_CLOSED: f64 = 1.9;
/// in contact, the tissue is this much stiffer than the mass's own spring
const CONTACT_STIFFNESS: f64 = 3.0;
/// length of the vibrating folds (cm) of the adult male template
const LENGTH: f64 = 1.4;
/// approximate frequency of the folds at Q = 1 (starting point for the pitch loop)
const F_NATURAL: f64 = 125.0;
/// the lower mass's rest gap is this much wider than the upper's: a slightly convergent glottis,
/// which is what lets the folds take energy from the flow and start vibrating
const CONVERGENCE: f64 = 0.004;
/// fractional lengthening of the folds per octave of pitch
const ELONGATION: f64 = 0.3;
/// rest half-gap added when no vibration is asked for (whisper), cm at Q ≤ 1: well apart
const WHISPER_REST: f64 = 0.1;
/// how much of the posterior opening (abduction / length, cm) also parts the folds along their length
const ABDUCT_REST: f64 = 0.5;
/// deepest the folds can be pressed into each other at rest (cm, half-overlap, at Q ≤ 1)
const MAX_PRESS: f64 = 0.016;
/// how far a full glottal stop (closure 1) presses the folds past that knee (cm, at Q ≤ 1)
const GLOTTAL_STOP_PRESS: f64 = 0.12;

/// Folds' setting, per voice, beyond what [`GlottisParams`] carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoldSetting {
    /// rest half-gap of the upper mass at the anterior commissure (cm); negative = pressed
    pub rest: f64,
    /// length scale of the folds (1 = adult male template)
    pub size: f64,
    /// closure shape: while the glottis closes, its area is passed out as `peak · (A / peak)^closure`.
    /// 1 leaves the masses' own closure; below 1 the area falls to zero more steeply at the end of
    /// the closing phase (an abrupt closure: a brighter voice), above 1 more gently (a darker one).
    pub closure: f64,
}

pub const DEFAULT_FOLDS: FoldSetting = FoldSetting { rest: 0.0, size: 1.0, closure: 1.0 };

#[derive(Clone, Debug)]
pub struct TriangularGlottis {
    pub fs: f64,
    pub folds: FoldSetting,
    dt: f64,
    // displacements (cm, lateral, + = away from the midline) and velocities of the two masses
    x1: f64,
    x2: f64,
    v1: f64,
    v2: f64,
    // per-sample smoothed controls (steps would click)
    rest_sm: f64,
    post_sm: f64,
    chink_sm: f64,
    closure_sm: f64,
    q_sm: f64,
    sm_coef: f64,
    /// log2 trim of Q by the pitch loop
    q_trim: f64,
    /// research switch (on by default): the slow loop that trims the folds' tension toward the
    /// asked F0 after each clear cycle. Off, the tension follows the asked F0 alone, so F0 moves
    /// with the folds' mechanics (pressure, adduction) and nothing pulls it back.
    pub trim_loop: bool,
    /// research input (0 by default): extra stiffening of the folds, as a fraction of their tension
    /// factor Q (cricothyroid activity, as during voiceless obstruents: Halle & Stevens's [stiff
    /// vocal folds]). F0 and whether the folds can vibrate then follow from the mechanics.
    pub stiffness: f64,
    /// the voice's own static abduction (cm², its breathiness), set by the voice each control
    /// step: only abduction beyond it (a consonant's, /h/'s) parts the folds along their length,
    /// so a breathy voice stays a voice and does not turn into a whisper
    pub abduction_base: f64,
    /// research switch (off by default): stiffening leaves the rest gap as it was. Off, a stiffer
    /// fold (higher Q) also rests proportionally closer to the midline, as a higher pitch does, so
    /// stiffening and adduction are confounded
    pub stiffness_keeps_gap: bool,
    /// research input (`CONTACT_STIFFNESS` by default): how much stiffer than its own spring a mass
    /// is while the folds are in contact. The contact force stiffens the folds over the closed part
    /// of each cycle, so F0 depends on how long and how deeply they touch.
    pub contact_stiffness: f64,
    /// research input (1 by default): the fraction of the extra damping in contact (`Z1_CLOSED`,
    /// `Z2_CLOSED` over the open values) that is applied
    pub contact_damping: f64,
    /// research switch (on by default): while the lower mass is shut, the subglottal pressure
    /// pushes on it (Steinecke & Herzel 1995). Off, that push is left out: not a physical setting,
    /// a way to see what the push does to F0 (see the `onset_probe` example).
    pub closed_push_lower: bool,
    /// research switch (on by default): while only the upper mass is shut, the subglottal pressure
    /// pushes on it as well as on the lower one. Off, only the lower one is pushed then (a
    /// diagnosis, as `closed_push_lower`).
    pub closed_push_upper: bool,
    /// research input (0 by default: linear springs): the cubic term of each mass's spring, η in
    /// k·x·(1 + η·x²) (cm⁻²). Ishizaka & Flanagan 1972 use 100 (as reproduced in Assaneo &
    /// Trevisan 2013, Eq. 2): the folds stiffen with the size of their swing.
    pub spring_cubic: f64,
    /// research input (0 by default): the cubic term of the contact force, η in
    /// c·k·d·(1 + η·d²) with d the penetration (cm⁻²; Ishizaka & Flanagan 1972 use 500)
    pub contact_cubic: f64,
    /// research input (`ABDUCT_REST` by default): how much of the posterior opening also parts
    /// the folds along their length. At 0 the rest area equals the abduction asked for, as the
    /// parametric source's does; at 0.5 it is about twice that.
    pub abduct_rest: f64,
    // pitch loop: cycle detection on the upper mass against its running mean
    x2_mean: f64,
    mean_coef: f64,
    armed: bool,
    since: f64,
    period: f64,
    x2_lo: f64,
    x2_hi: f64,
    swing: f64,
    // pitch modulation (shared meaning with the parametric source)
    vib_phase: f64,
    drift: Drift,
    drift_value: f64,
    drift_counter: i32,
    pitch_factor: f64,
    cycle_q: f64,
    rng: Rng,
    voice_coef: f64,
    /// size / Q of the last sample (the scale of the folds' gaps and swing)
    gap_scale: f64,
    // vibratory area: this cycle's highest so far, and the last cycle's peak (closure shaping)
    area_hi: f64,
    area_peak: f64,
    area_prev: f64,
    /// open fraction of the last cycle (telemetry, tests)
    pub open_quotient: f64,
    open_count: f64,
    /// last glottal area (cm²)
    pub area: f64,
    /// research telemetry: the narrower of the two masses' channels this sample (cm², before the
    /// closure shaping, the glottal stop and the chink), 0 while the folds are shut
    channel: f64,
    /// research telemetry: the force (dyn) with which the subglottal pressure pushes the lower
    /// mass this sample because a mass is shut (the closed-phase push; 0 while both are open)
    closed_push_force: f64,
    /// requested f0 after vibrato and drift (Hz)
    pub f0_now: f64,
    /// 0..1: how strongly the folds are vibrating
    pub voicing: f64,
}

impl TriangularGlottis {
    pub fn new(fs: f64) -> Self {
        TriangularGlottis {
            fs,
            folds: DEFAULT_FOLDS,
            dt: 1.0 / fs,
            x1: 0.0,
            x2: 0.0,
            v1: 0.0,
            v2: 0.0,
            rest_sm: 0.0,
            post_sm: 0.0,
            chink_sm: 0.0,
            closure_sm: 0.0,
            q_sm: 1.0,
            sm_coef: 1.0 - exp(-1.0 / (0.004 * fs)),
            q_trim: 0.0,
            trim_loop: true,
            stiffness: 0.0,
            abduction_base: 0.0,
            stiffness_keeps_gap: false,
            contact_stiffness: CONTACT_STIFFNESS,
            contact_damping: 1.0,
            closed_push_lower: true,
            closed_push_upper: true,
            spring_cubic: 0.0,
            contact_cubic: 0.0,
            abduct_rest: ABDUCT_REST,
            x2_mean: 0.0,
            mean_coef: 1.0 - exp(-TAU * 15.0 / fs),
            armed: false,
            since: 0.0,
            period: fs / F_NATURAL,
            x2_lo: 0.0,
            x2_hi: 0.0,
            swing: 0.0,
            vib_phase: 0.0,
            drift: Drift::new(fs / 32.0, 0.3, 4242),
            drift_value: 0.0,
            drift_counter: 0,
            pitch_factor: 1.0,
            cycle_q: 1.0,
            rng: Rng::new(777),
            voice_coef: 1.0 - exp(-TAU * 25.0 / fs),
            gap_scale: 1.0,
            area_hi: 0.0,
            area_peak: 0.1,
            area_prev: 0.0,
            open_quotient: 0.0,
            open_count: 0.0,
            area: 0.0,
            channel: 0.0,
            closed_push_force: 0.0,
            f0_now: 110.0,
            voicing: 0.0,
        }
    }

    /// The states that decay toward zero once the breath stops or the folds part.
    fn decaying(&mut self) -> [&mut f64; 15] {
        [
            &mut self.x1, &mut self.x2, &mut self.v1, &mut self.v2, &mut self.rest_sm, &mut self.post_sm, &mut self.chink_sm, &mut self.closure_sm,
            &mut self.x2_mean, &mut self.x2_lo, &mut self.x2_hi, &mut self.swing, &mut self.voicing, &mut self.area_prev, &mut self.area,
        ]
    }

    /// How many of the decaying states are denormal (for tests: none should be).
    pub fn denormal_states(&self) -> usize {
        [
            self.x1, self.x2, self.v1, self.v2, self.rest_sm, self.post_sm, self.chink_sm, self.closure_sm, self.x2_mean, self.x2_lo, self.x2_hi, self.swing, self.voicing, self.area_prev, self.area,
        ]
        .iter()
        .filter(|x| crate::airway::network::is_denormal(**x))
        .count()
    }

    /// Zero the states that have decayed to nothing (see `AirwayNetwork::flush_tiny`). Called at
    /// control rate.
    pub fn flush_tiny(&mut self) {
        for x in self.decaying() {
            crate::airway::network::tiny(x);
        }
    }

    /// Research telemetry: the narrowest channel between the folds this sample (cm²; see `channel`).
    pub fn channel_area(&self) -> f64 {
        self.channel
    }

    /// Research telemetry: the closed-phase push on the lower mass this sample (dyn).
    pub fn closed_push_force(&self) -> f64 {
        self.closed_push_force
    }

    /// Research: the pitch loop's trim (log2 of its tension factor). Read after the loop has
    /// settled and set into another run with `trim_loop` off, the folds keep that operating point.
    pub fn trim(&self) -> f64 {
        self.q_trim
    }

    /// Research: set the pitch loop's trim (see `trim`).
    pub fn set_trim(&mut self, t: f64) {
        self.q_trim = t;
    }

    /// Research telemetry: the smoothed tension factor Q the masses run at (masses / Q, springs · Q).
    pub fn tension(&self) -> f64 {
        self.q_sm
    }

    /// The upper mass's displacement from its rest (cm, telemetry): it swings with the folds and,
    /// unlike the glottal area, carries no abduction offset, so cycles can be read from it.
    pub fn displacement(&self) -> f64 {
        self.x2
    }

    /// Phase within the current cycle, 0..1 (restarts at each opening).
    pub fn cycle_phase(&self) -> f64 {
        js_min(1.0, self.since / js_max(1.0, self.period))
    }

    /// Advance one sample. `dp` is the transglottal pressure (p_sub − p_supra, dyn/cm²); `p`
    /// carries the shared source settings (f0, abduction, chink, closure, jitter, vibrato, drift).
    /// Returns the glottal area (cm²).
    #[inline]
    pub fn next(&mut self, dp: f64, p: &GlottisParams) -> f64 {
        // ---- pitch modulation, decimated (as the parametric source does it)
        self.drift_counter -= 1;
        if self.drift_counter <= 0 {
            self.drift_counter = 32;
            self.drift_value = self.drift.step();
            self.vib_phase += ((p.vibrato_rate * 32.0) / self.fs) * (1.0 + 0.04 * self.drift_value);
            if self.vib_phase > 1.0 {
                self.vib_phase -= 1.0;
            }
            let cents = p.vibrato_depth * sin(TAU * self.vib_phase) + p.drift * self.drift_value;
            self.pitch_factor = pow(2.0, cents / 1200.0);
        }
        let f0 = js_max(20.0, p.f0 * self.pitch_factor);
        self.f0_now = f0;

        // ---- controls → fold geometry
        let size = self.folds.size;
        // the folds lengthen as the cricothyroid raises the pitch (Titze 1994, ch. 8; VocalTractLab's
        // folds lengthen with f0 too): ELONGATION per octave above the body's low register, for two
        // octaves. Without it a high note's stiffer, smaller swing left the voice ~5 dB quieter per
        // octave than the parametric pulse, which keeps its size at any pitch.
        let octaves = clamp(log2(f0 * size / F_NATURAL), 0.0, 2.0);
        let len = LENGTH * size * (1.0 + ELONGATION * octaves);
        let sc = self.sm_coef;
        // a glottal stop presses the folds together and shuts the cartilaginous gap
        self.closure_sm += sc * (p.closure - self.closure_sm);
        // whisper (no vibratory amplitude asked for) parts the folds outright; abduction (voiceless
        // consonants, /h/) parts the vocal processes, which spreads along the folds as well as
        // opening the posterior triangle (as VocalTractLab's rest displacements do for voiceless
        // sounds). Without it a whisper at a firm breath vibrated fully (pressure 15 cmH2O).
        let parted = if p.amplitude > 0.0 { 0.0 } else { WHISPER_REST };
        self.rest_sm += sc * (self.folds.rest + parted + self.abduct_rest * js_max(0.0, p.abduction - self.abduction_base) / len - self.rest_sm);
        // abduction (cm²) opens the posterior end: a triangle of that area, apex at the commissure
        self.post_sm += sc * (p.abduction / len - self.post_sm);
        self.chink_sm += sc * (p.chink - self.chink_sm);
        let q_target = (f0 / F_NATURAL) * pow(2.0, self.q_trim) * self.cycle_q * (1.0 + self.stiffness);
        self.q_sm += sc * (clamp(q_target, 0.15, 12.0) - self.q_sm);
        let q = self.q_sm;
        let post = self.post_sm * (1.0 - 0.92 * self.closure_sm);
        // stiffer folds (higher pitch) swing less, by 1/Q: the rest gap is set in proportion, so
        // "pressed" or "lax" means the same at any pitch (a fixed gap in cm would stop pressed
        // folds from vibrating above an octave over F_NATURAL, and keep lax ones from closing)
        // smaller folds deflect less under the same pressure (force ∝ size², stiffness ∝ size), so
        // the gap scales with their size too
        let q_gap = if self.stiffness_keeps_gap { q / (1.0 + self.stiffness) } else { q };
        let gap_scale = size / js_max(1.0, q_gap);
        self.gap_scale = gap_scale;
        // pressing saturates (a tanh knee at MAX_PRESS): overlap beyond it would hold the folds shut
        // at speaking pressures, a silence no singer produces by pressing (they push harder)
        // a glottal stop presses beyond that knee: it holds the folds shut, as the pulse does
        let rest = (if self.rest_sm < 0.0 { -MAX_PRESS * tanh(-self.rest_sm / MAX_PRESS) } else { self.rest_sm }) - GLOTTAL_STOP_PRESS * self.closure_sm;
        let r2 = rest * gap_scale;
        let r1 = r2 + CONVERGENCE * gap_scale;

        // ---- geometry: open area and contact of each mass along the triangular glottis
        let (a1, c1, pen1) = channel(self.x1 + r1, post, len);
        let (a2, c2, pen2) = channel(self.x2 + r2, post, len);

        // ---- aerodynamic pressures on the masses (Steinecke & Herzel 1995)
        let eps = 1e-6;
        let (p1, p2) = if a1 <= eps {
            (if self.closed_push_lower { dp } else { 0.0 }, 0.0)
        } else if a2 <= eps {
            (dp, if self.closed_push_upper { dp } else { 0.0 })
        } else {
            let amin = js_min(a1, a2);
            let r = amin / a1;
            (dp * (1.0 - r * r), 0.0)
        };
        let f1 = p1 * len * D1 * size;
        self.closed_push_force = if a1 <= eps || a2 <= eps { f1 } else { 0.0 };
        let f2 = p2 * len * D2 * size;

        // ---- mechanics, at tension factor Q (masses / Q, stiffnesses · Q)
        let m1 = M1 * size / q;
        let m2 = M2 * size / q;
        let k1 = K1 * size * q;
        let k2 = K2 * size * q;
        let kc = KC * size * q;
        let z1 = Z1_OPEN + (Z1_CLOSED - Z1_OPEN) * c1 * self.contact_damping;
        let z2 = Z2_OPEN + (Z2_CLOSED - Z2_OPEN) * c2 * self.contact_damping;
        let b1 = 2.0 * z1 * sqrt(m1 * k1);
        let b2 = 2.0 * z2 * sqrt(m2 * k2);
        // tissue compression: the penetration (mean over the length) pushes back
        let fc1 = self.contact_stiffness * k1 * pen1 * (1.0 + self.contact_cubic * pen1 * pen1);
        let fc2 = self.contact_stiffness * k2 * pen2 * (1.0 + self.contact_cubic * pen2 * pen2);
        let s1 = k1 * self.x1 * (1.0 + self.spring_cubic * self.x1 * self.x1);
        let s2 = k2 * self.x2 * (1.0 + self.spring_cubic * self.x2 * self.x2);
        let acc1 = (f1 + fc1 - b1 * self.v1 - s1 - kc * (self.x1 - self.x2)) / m1;
        let acc2 = (f2 + fc2 - b2 * self.v2 - s2 - kc * (self.x2 - self.x1)) / m2;
        // semi-implicit Euler: velocities first, then positions with the new velocities
        self.v1 += acc1 * self.dt;
        self.v2 += acc2 * self.dt;
        self.x1 += self.v1 * self.dt;
        self.x2 += self.v2 * self.dt;
        if !(self.x1.is_finite() && self.x2.is_finite() && self.v1.is_finite() && self.v2.is_finite()) {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.v1 = 0.0;
            self.v2 = 0.0;
        }
        // keep the folds within anatomy (a fold cannot travel 4 mm from the midline)
        self.x1 = clamp(self.x1, -0.2, 0.4);
        self.x2 = clamp(self.x2, -0.2, 0.4);

        // ---- cycle detection: the upper mass crossing its running mean upward
        self.x2_mean += self.mean_coef * (self.x2 - self.x2_mean);
        self.x2_lo = js_min(self.x2_lo, self.x2);
        self.x2_hi = js_max(self.x2_hi, self.x2);
        self.since += 1.0;
        let amin = js_min(a1, a2);
        if amin > 1e-4 {
            self.open_count += 1.0;
        }
        let hyst = 0.002;
        if self.x2 < self.x2_mean - hyst {
            self.armed = true;
        }
        if self.armed && self.x2 > self.x2_mean + hyst {
            self.armed = false;
            self.cycle(f0, p.jitter);
        }
        if self.since > self.fs / 25.0 {
            // no cycle for 40 ms: not vibrating
            self.since = 0.0;
            self.swing = 0.0;
            self.open_count = 0.0;
            self.x2_lo = self.x2;
            self.x2_hi = self.x2;
        }
        // the swing shrinks with stiffer, smaller folds as their gaps do: judged against the same scale
        self.voicing += self.voice_coef * (clamp(self.swing / (0.04 * self.gap_scale), 0.0, 1.0) - self.voicing);

        // closure shape (brightness), scaled by the last cycle's peak so that it changes the pulse's
        // shape and not its size. The masses are a two-point sketch of a fold; how abruptly the real
        // medial surface meets along its length and depth is what brightness sets (the parametric
        // source's closure sharpness, for the folds). Only the area the airway sees is shaped: the
        // pressures on the masses use their own areas.
        let mut am = js_max(0.0, amin);
        self.channel = am;
        self.area_hi = js_max(self.area_hi, am);
        let closing = am < self.area_prev;
        self.area_prev = am;
        let g = self.folds.closure;
        // closing phase only: an abrupt closure is what brightens a voice; shaping the opening too
        // fattened the pulse and raised H1-H2, hiding the pressed folds' lower H1-H2
        if g != 1.0 && am > 0.0 && closing {
            let peak = js_max(self.area_peak, 1e-4);
            am = if am < peak { peak * pow(am / peak, g) } else { am };
        }
        let a = am * (1.0 - self.closure_sm) + self.chink_sm * (1.0 - 0.92 * self.closure_sm);
        self.area = a;
        a
    }

    /// A cycle ended: measure it and trim Q toward the requested pitch.
    fn cycle(&mut self, f0: f64, jitter: f64) {
        let period = self.since;
        if self.area_hi > 1e-4 {
            self.area_peak = self.area_hi;
        }
        self.area_hi = 0.0;
        self.swing = self.x2_hi - self.x2_lo;
        self.x2_lo = self.x2;
        self.x2_hi = self.x2;
        if period > 0.0 {
            self.open_quotient = self.open_count / period;
        }
        self.open_count = 0.0;
        self.since = 0.0;
        let f = self.fs / js_max(1.0, period);
        // only a real oscillation (a clear swing, a plausible rate) steers the pitch
        if self.swing > 0.004 && f > 25.0 && f < 2500.0 {
            self.period = period;
            if self.trim_loop {
                let err = log2(f0 / f);
                self.q_trim = clamp(self.q_trim + 0.35 * err, -2.0, 2.0);
            }
        }
        // jitter: each cycle's stiffness varies a little, so its period does
        self.cycle_q = clamp(1.0 + jitter * self.rng.gauss(), 0.8, 1.25);
    }
}

/// Open area (cm²), contact fraction (0..1) and mean tissue penetration (cm, ≥ 0) of one mass's
/// channel. Each fold's edge runs straight from the anterior commissure (half-gap `h`) to the
/// vocal process (half-gap `h + post`), so the gap between the folds is `2·(h + post·s)` at
/// `s` ∈ [0, 1] along the length.
#[inline]
fn channel(h: f64, post: f64, len: f64) -> (f64, f64, f64) {
    let a = 2.0 * h; // gap at the front
    let b = 2.0 * (h + post); // gap at the back
    if a >= 0.0 && b >= 0.0 {
        return (len * 0.5 * (a + b), 0.0, 0.0);
    }
    if a <= 0.0 && b <= 0.0 {
        // closed along the whole length; penetration is half the overlap (per fold)
        return (0.0, 1.0, -0.25 * (a + b));
    }
    // the edge crosses the midline at fraction z of the length
    let z = a / (a - b);
    if a > 0.0 {
        // open in front, closed behind (pressed with the vocal processes overlapping)
        (len * 0.5 * a * z, 1.0 - z, -0.25 * b * (1.0 - z))
    } else {
        // closed in front, open behind (the common case: a posterior gap)
        (len * 0.5 * b * (1.0 - z), z, -0.25 * a * z)
    }
}

//! Cycle reader shared by the research probes: successive peaks of a displacement-like signal.
//! Checked against the parametric source's known phase in `cycle_check` (after a consonant: within
//! 0.72 Hz, no missed cycle; before one: up to 3 Hz, not used for conclusions). On the two-mass
//! folds' upper-mass displacement (threshold 0.002 cm) it has not been checked separately.

/// Peaks of `y` sampled at `fs`: (index, refined time in s). A sample is a peak when it is the
/// single maximum of the ±`half` window and stands at least `min_prominence` above the lowest
/// point since the previous peak (or the 4·`half` samples before the first).
pub fn peaks(y: &[f64], fs: f64, half: usize, min_prominence: f64) -> Vec<(usize, f64)> {
    let n = y.len();
    let mut out = vec![];
    let mut last: Option<usize> = None;
    for i in half..n.saturating_sub(half) {
        let w = &y[i - half..=i + half];
        if w.iter().all(|&v| v <= y[i]) && w.iter().filter(|&&v| v == y[i]).count() == 1 {
            let lo = last.map_or(&y[i.saturating_sub(4 * half)..i], |j| &y[j..i]).iter().cloned().fold(f64::MAX, f64::min);
            if y[i] - lo >= min_prominence {
                let (a, b, c) = (y[i - 1], y[i], y[i + 1]);
                let den = a - 2.0 * b + c;
                let off = if den.abs() > 1e-18 { 0.5 * (a - c) / den } else { 0.0 };
                out.push((i, (i as f64 + off) / fs));
                last = Some(i);
            }
        }
    }
    out
}

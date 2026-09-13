use crate::constants::*;

pub mod kerr_schild;
pub mod boyer_lindquist;

#[derive(Debug, PartialEq, Eq)]
pub enum RegionType {
    Us, Para,
    AfterOuter, AfterInner,
    UsSing, ParaSing
}
pub type Region = (i32, RegionType);

#[inline]
/// Converts from cartesian coordinates to the oblate spheroidal radius.
/// Has an extra argument `rev` for the sign.
pub fn radius(x: f64, y: f64, z: f64, rev: bool) -> f64 {
    let d = x*x + y*y + z*z - SPIN*SPIN;
    let result = f64::sqrt(0.5 * (d + f64::sqrt(d*d + 4.0 * (SPIN*SPIN) * (z*z))));
    if rev {
        - result
    } else {
        result
    }
}

#[inline]
pub fn discr(r: f64) -> f64 {
    r*r - 2.0*MASS*r + SPIN*SPIN
}

#[inline]
/// The function R(r) appearing in the first order equations of motion.
/// Must be non-negative at any point on the geodesic.
pub fn motionR(r: f64, modulus: f64, energy: f64, angular: f64, carter: f64) -> f64 {
    ((r*r + SPIN*SPIN) * energy - SPIN * angular).powi(2)
    - discr(r) * (carter + (angular - SPIN * energy).powi(2) - r*r * modulus)
}

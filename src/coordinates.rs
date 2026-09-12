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
fn radius(x: f64, y: f64, z: f64, rev: bool) -> f64 {
    let d = x*x + y*y + z*z - SPIN*SPIN;
    let result = f64::sqrt(0.5 * (d + f64::sqrt(d*d + 4.0 * (SPIN*SPIN) * (z*z))));
    if rev {
        - result
    } else {
        result
    }
}

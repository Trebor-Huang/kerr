use crate::{coordinates::*, utils::*, constants::*};

#[derive(Debug, Clone, Copy)]
/// A point is given by the coordinates, the number of universes
/// and whether it is the "other" piece. By default we take
/// the radius to decrease as we go from bottom-right to top-left.
/// Outside the outer horizon and inside the inner horizon, shifting
/// takes us to the parallel universe, while between the horizons
/// shifting flips the timelike orientation.
pub struct Pt {
    t: f64, r: f64, θ: f64, φ: f64,
    base: i32,
    shift: bool,
}

impl Pt {
    /// Returns the cosmological region
    pub fn region(self: Self) -> Region {
        if self.r < R_INNER {
            (self.base, if self.shift {RegionType::UsSing} else {RegionType::ParaSing})
        } else if self.r < R_OUTER {
            (self.base, if self.shift {RegionType::AfterInner} else {RegionType::AfterOuter})
        } else {
            (self.base, if self.shift {RegionType::Para} else {RegionType::Us})
        }
    }
}

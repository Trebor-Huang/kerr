use crate::{coordinates::*, utils::*, constants::*};

#[derive(Debug, Clone, Copy)]
/// A point is given by the coordinates, the number of universes
/// and whether it is the "other" piece. By default we take
/// the radius to decrease as we go from bottom-right to top-left.
/// Outside the outer horizon and inside the inner horizon, shifting
/// takes us to the parallel universe, while between the horizons
/// shifting flips the timelike orientation.
pub struct Pt {
    coord: Quad,
    base: i32,
    radius_rev: bool,
    shift: bool,
}

impl Pt {
    pub fn radius(self) -> f64 {
        // We don't cache this since it's not intended for the hot path,
        // i.e. the integration of geodesics.
        let (_, x, y, z) = self.coord.explode();
        radius(x, y, z, self.radius_rev)
    }

    /// Returns the cosmological region
    pub fn region(self: Self) -> Region {
        let r = self.radius();
        if r < R_INNER {
            (self.base, if self.shift {RegionType::UsSing} else {RegionType::ParaSing})
        } else if r < R_OUTER {
            (self.base, if self.shift {RegionType::AfterInner} else {RegionType::AfterOuter})
        } else {
            (self.base, if self.shift {RegionType::Para} else {RegionType::Us})
        }
    }

    pub fn from_kerr_schild(pt: kerr_schild::Pt) -> Pt {
        let (t0, x0, y0, z0) = pt.coord.explode();
        let r0 = radius(x0, y0, z0, pt.radius().is_sign_negative());
        let δφ = f64::atan2(SPIN, r0)
            + 0.5 * SPIN / f64::sqrt(MASS*MASS - SPIN*SPIN) *
            f64::ln(f64::abs(if f64::abs(r0) < 1.0 {
                (r0 - R_OUTER)/(r0 - R_INNER)
            } else {
                (1.0 - R_OUTER/r0)/(1.0 - R_INNER/r0)
            }));
        let δt = MASS / f64::sqrt(MASS*MASS - SPIN*SPIN) * (
            R_OUTER * f64::ln(f64::abs((r0 - R_OUTER)/(2.0 * MASS))) -
            R_INNER * f64::ln(f64::abs((r0 - R_INNER)/(2.0 * MASS)))
        );
        let cos = f64::cos(δφ);
        let sin = f64::sin(δφ);
        // TODO properly take the reversion into account
        let x = cos * x0 - sin * y0;
        let y = sin * x0 + cos * y0;
        let t = t0 - δt;
        Pt {
            coord: Quad::new(t, x, y, z0),
            base: todo!(),
            radius_rev: todo!(),
            shift: todo!(),
        }
    }
}

// TODO convert from kerr–schild, and (co)tangent vectors
// metric is probably not gonna be useful except for double checking

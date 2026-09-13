use crate::{coordinates::*, utils::*, constants::*};

#[derive(Debug, Clone, Copy)]
/// A point is given by the coordinates, the number of universes
/// and whether it is the "other" piece. By default we take
/// the radius to decrease as we go from bottom-right to top-left.
/// Outside the outer horizon and inside the inner horizon, shifting
/// takes us to the parallel universe, while between the horizons
/// shifting flips the timelike orientation.
pub struct Pt {
    pub coord: Quad,
    pub base: i32,
    pub radius_rev: bool,
    pub shift: bool,
}

impl Pt {
    #[cfg(test)]
    pub fn error(self: Pt, other: Pt) -> f64 {
        assert_eq!(self.base, other.base);
        assert_eq!(self.radius_rev, other.radius_rev);
        assert_eq!(self.shift, other.shift);
        self.coord.error(other.coord)
    }

    #[cfg(test)]
    pub fn rand() -> Pt {
        use rand::*;
        Pt {
            coord: Quad::rand(),
            base: random_range(-5..5),
            radius_rev: random_bool(0.5),
            shift: random_bool(0.5),
        }
    }

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
        let sin = f64::sin(-δφ);
        /*  A.Inner
         P.S.      U.S.
            A.Outer
         Para       Us
        */
        // By default, the Kerr coordinate chart spans P.S., A.Outer and Us
        // `base` shifts the universe number
        // `parallel` flips left and right
        // `time_rev` flips up and down
        // However, `time_rev` does not always flip the time *coordinate*.
        // This is flipped on the sides, whereas in the middle it flips the roles
        // of AfterInner and AfterOuter.
        let flipped = pt.time_rev ^ pt.parallel;
        let t = if flipped {δt - t0} else {t0 - δt};
        let x = cos * x0 - sin * y0;
        let y = sin * x0 + cos * y0;
        let y = if flipped {-y} else {y}; // TODO should I flip y or y0
        let base = if r0 < R_OUTER && pt.time_rev {
            pt.base - 1
        } else {
            pt.base
        };
        let shift = if r0 < R_INNER || r0 > R_OUTER {
            pt.parallel
        } else {
            pt.time_rev
        };
        Pt {
            coord: Quad::new(t, x, y, z0),
            base,
            radius_rev: r0.is_sign_negative(),
            shift,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::coordinates::{boyer_lindquist::*, kerr_schild};

    const NUM: i32 = 100;
    const ERR: f64 = 1e-12;

    #[test]
    fn from_kerr_schild_region() {
        for i in 0..NUM {
            let pt = kerr_schild::Pt::rand();
            assert_eq!(pt.region(), Pt::from_kerr_schild(pt).region())
        }
    }

    #[test]
    fn from_kerr_schild_flip() {
        let mut err = 0.0;
        for i in 0..NUM {
            let pt = kerr_schild::Pt::rand();
            let bl1 = Pt::from_kerr_schild(pt);
            let bl2 = Pt::from_kerr_schild(pt.flip());
            err += bl1.error(bl2);
        }
        assert!(err < ERR * NUM as f64, "{err}");
    }
}

// TODO convert from kerr–schild, and (co)tangent vectors
// metric is probably not gonna be useful except for double checking

use crate::{constants::*, utils::*, coordinates::*};
use std::autodiff::*;

// TODO better numerical behavior
#[autodiff_forward(_fflip, Dual, Dual, Dual, Dual, Const, Dual, Dual, Dual)]
#[autodiff_reverse(_rflip, Active, Active, Active, Active, Const, Duplicated, Duplicated, Duplicated)]
fn _flip(
    t0: f64, x0: f64, y0: f64, z0: f64, rev: bool,
    t: &mut f64, x: &mut f64, y: &mut f64
) {
    let r0 = radius(x0, y0, z0, rev);
    let δφ = 2.0 * f64::atan2(SPIN, r0)
        + SPIN / f64::sqrt(MASS*MASS - SPIN*SPIN) *
        f64::ln(f64::abs(if f64::abs(r0) < 1.0 {
            (r0 - R_OUTER)/(r0 - R_INNER)
        } else {
            (1.0 - R_OUTER/r0)/(1.0 - R_INNER/r0)
        }));
    let δt = 2.0 * MASS / f64::sqrt(MASS*MASS - SPIN*SPIN) * (
        R_OUTER * f64::ln(f64::abs((r0 - R_OUTER)/(2.0 * MASS))) -
        R_INNER * f64::ln(f64::abs((r0 - R_INNER)/(2.0 * MASS)))
    );
    let cos = f64::cos(δφ);
    let sin = f64::sin(δφ);
    *x = cos * x0 + sin * y0;
    *y = sin * x0 - cos * y0;
    *t = δt - t0;
}

fn flip(t0: f64, x0: f64, y0: f64, z0: f64, rev: bool) -> (f64, f64, f64) {
    let mut t = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    _flip(t0,x0,y0,z0,rev,&mut t,&mut x,&mut y);
    (t, x, y)
}

fn flip_tangent(
    t0: f64, x0: f64, y0: f64, z0: f64, rev: bool,
    dt0: f64, dx0: f64, dy0: f64, dz0: f64,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mut t = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut dt = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    _fflip(
        t0, dt0, x0, dx0, y0, dy0, z0, dz0, rev,
        &mut t, &mut dt, &mut x, &mut dx, &mut y, &mut dy
    );
    ((t,x,y), (dt,dx,dy))
}

// This happens to work because flipping is an involution
fn flip_cotangent(
    t0: f64, x0: f64, y0: f64, z0: f64, rev: bool,
    d_dt: f64, d_dx: f64, d_dy: f64, d_dz: f64
) -> ((f64, f64, f64), Quad) {
    let (mut t, mut x, mut y) = flip(t0, x0, y0, z0, rev);
    let mut d_dt = d_dt;
    let mut d_dx = d_dx;
    let mut d_dy = d_dy;
    let (d_dt0, d_dx0, d_dy0, d_dz0) = _rflip(
        t, x, y, z0, rev,
        &mut t, &mut d_dt, &mut x, &mut d_dx, &mut y, &mut d_dy
    );
    ((t, x, y), Quad::new(d_dt0, d_dx0, d_dy0, d_dz0 + d_dz))
}

#[autodiff_reverse(_dinv_sq, Active, Active, Active, Const, Const, Const, Const, Const, Active)]
fn inv_sq(x: f64, y: f64, z: f64, rev: bool, qt: f64, qx: f64, qy: f64, qz: f64) -> f64 {
    let r = radius(x, y, z, rev);
    let ra = r*r + SPIN*SPIN;
    let qk = qt - qx * (r*x + SPIN*y)/ra - qy * (r*y - SPIN*x)/ra - qz * z/r;
    let h = ks_scalar(z, r);
    qx*qx + qy*qy + qz*qz - qt*qt - 2.0*h*(qk*qk)
}

fn diff_inv_sq(x: f64, y: f64, z: f64, rev: bool, qt: f64, qx: f64, qy: f64, qz: f64) -> (f64, f64, f64) {
    let r = _dinv_sq(x, y, z, rev, qt, qx, qy, qz, 1.0);
    (r.1, r.2, r.3)
}

#[inline]
fn ks_scalar(z: f64, r: f64) -> f64 {
    MASS * (r.powi(3))/(r.powi(4) + (SPIN*SPIN) * (z*z))
}


#[derive(Debug, Clone, Copy)]
pub struct Pt {
    pub coord: Quad,
    pub base: i32,
    pub parallel: bool,
    pub time_rev: bool,
    radius: f64,
}

impl Pt {
    pub fn new(coord: Quad, base: i32, parallel: bool, time_rev: bool, radius_rev: bool) -> Pt {
        let (_, x, y, z) = coord.explode();
        Pt {
            coord, base, parallel, time_rev,
            radius: radius(x, y, z, radius_rev),
        }
    }

    #[inline]
    pub fn radius(self: Self) -> f64 {
        self.radius
    }

    #[cfg(test)]
    pub fn error(self: &Pt, other: &Pt) -> f64 {
        assert_eq!(self.base, other.base);
        assert_eq!(self.parallel, other.parallel);
        assert_eq!(self.time_rev, other.time_rev);
        self.coord.error(other.coord)
    }

    /// Returns the cosmological region
    pub fn region(self: Self) -> Region {
        if self.radius <= R_INNER {
            (
                if self.time_rev {self.base - 1} else {self.base},
                if self.parallel {RegionType::UsSing} else {RegionType::ParaSing}
            )
        } else if self.radius <= R_OUTER {
            if self.time_rev {
                (self.base - 1, RegionType::AfterInner)
            } else {
                (self.base, RegionType::AfterOuter)
            }
        } else {
            (self.base, if self.parallel {RegionType::Para} else {RegionType::Us})
        }
    }

    pub fn flip(self: Self) -> Self {
        let (t0, x0, y0, z0) = self.coord.explode();
        let (t,x,y) = flip(
            t0, x0, y0, z0,
            self.radius.is_sign_negative()
        );

        // If we are in the singular region, then flipping will increment/decrement base universe
        let base = if self.radius <= R_INNER {
            if self.time_rev {
                self.base - 1
            } else {
                self.base + 1
            }
        } else {
            self.base
        };
        // If we are not in the middle region, then our parallel universe status flips
        let is_middle = R_INNER < self.radius && self.radius < R_OUTER;
        let parallel = is_middle != self.parallel;
        let time_rev = is_middle == self.time_rev;
        Pt {
            coord: Quad::new(t, x, y, z0),
            base,
            parallel,
            time_rev,
            radius: self.radius,
        }
    }

    #[inline]
    pub fn discr(self: Self) -> f64 {
        let r = self.radius;
        r*r - 2.0*MASS*r + SPIN*SPIN
    }

    /// Nudges the coordinate in a direction. If it went through the ring,
    /// flip the sign of the radius.
    pub fn nudge(self: Self, delta: Quad) -> Self {
        let (t,x,y,z) = self.coord.explode();
        let (dt, dx, dy, dz) = delta.explode();
        let new_coord = self.coord + delta;
        let z1 = new_coord.explode().3;
        if (z.signum() * z1.signum()).is_sign_negative() {
            let u = - z / dz;
            let x0 = x + u * dx;
            let y0 = y + u * dy;
            if x0*x0 + y0*y0 < SPIN*SPIN {
                return Pt::new(
                    new_coord,
                    self.base, self.parallel, self.time_rev,
                    !self.radius.is_sign_negative()
                )
            }
        }
        return Pt::new(
            new_coord,
            self.base, self.parallel, self.time_rev,
            self.radius.is_sign_negative()
        );
    }

    #[inline]
    pub fn incoming(self: Self) -> Quad {
        let (_,x,y,z) = self.coord.explode();
        let r = self.radius;
        Quad::new(
            -1.0,
            (r*x + SPIN*y)/(r*r + SPIN*SPIN),
            (r*y - SPIN*x)/(r*r + SPIN*SPIN),
            z/r
        )
    }

    #[inline]
    pub fn incoming_dual(self: Self) -> Quad {
        let (_,x,y,z) = self.coord.explode();
        let r = self.radius;
        Quad::new(
            1.0,
            (r*x + SPIN*y)/(r*r + SPIN*SPIN),
            (r*y - SPIN*x)/(r*r + SPIN*SPIN),
            z/r
        )
    }

    #[inline]
    pub fn outgoing(self: Self) -> Quad {
        let (_,x,y,z) = self.coord.explode();
        let r = self.radius;
        let delta = self.discr();
        Quad::new(
            1.0 + (4.0 * MASS * r)/delta,
            (r*x + SPIN*y)/(r*r + SPIN*SPIN) - (2.0 * SPIN * y)/delta,
            (r*y - SPIN*x)/(r*r + SPIN*SPIN) + (2.0 * SPIN * x)/delta,
            z/r
        )
    }

    #[inline]
    pub fn outgoing_dual(self: Self) -> Quad {
        let qk = self.incoming_outgoing_dot();
        let h = 2.0
            * ks_scalar(self.coord.explode().3, self.radius)
            * qk;
        self.outgoing() * Quad::ETA
            + self.incoming_dual().scale(h)
    }

    #[inline]
    pub fn incoming_outgoing_dot(self: Self) -> f64 {
        let r = self.radius;
        let (_, x, y, _) = self.coord.explode();
        let delta = self.discr();
        let ra = r*r + SPIN*SPIN;
        2.0 + 4.0 * MASS * r / delta - 2.0 * (SPIN*SPIN) * (x*x + y*y)/(delta * ra)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tangent {
    pub vec: Quad,
    pub pt: Pt,
}

impl Tangent {
    #[cfg(test)]
    pub fn error(this: &Tangent, other: &Tangent) -> f64 {
        (Pt::error(&this.pt, &other.pt) + this.vec.error(other.vec)) / 2.0
    }

    pub fn modulus(self: Self) -> f64 {
        let qk = self.vec.dot(self.pt.incoming_dual());
        let h = ks_scalar(self.pt.coord.explode().3, self.pt.radius);
        self.vec.dot(self.vec * Quad::ETA)
            + 2.0*h*(qk*qk)
    }

    pub fn dual(self: Self) -> Cotangent {
        let q = self.vec;
        let k = self.pt.incoming_dual();
        let qk = q.dot(k);
        let h = 2.0 * ks_scalar(self.pt.coord.explode().3, self.pt.radius) * qk;
        Cotangent {
            covec: q * Quad::ETA
                + k.scale(h),
            pt: self.pt,
        }
    }

    pub fn flip(self: Self) -> Self {
        let (t, x, y, z) = self.pt.coord.explode();
        let (dt, dx, dy, dz) = self.vec.explode();
        let (_, (dt1, dx1, dy1)) = flip_tangent(
            t, x, y, z, self.pt.radius.is_sign_negative(),
            dt, dx, dy, dz
        );
        Tangent {
            vec: Quad::new(dt1, dx1, dy1, dz),
            pt: self.pt.flip(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Cotangent {
    pub covec: Quad,
    pub pt: Pt,
}
#[derive(Clone, Copy)]
pub struct CotangentDelta {
    covec: Quad,
    pt: Quad
}

impl std::ops::Add for CotangentDelta {
    type Output = CotangentDelta;

    fn add(self: CotangentDelta, rhs: CotangentDelta) -> CotangentDelta {
        CotangentDelta {
            covec: self.covec + rhs.covec,
            pt: self.pt + rhs.pt,
        }
    }
}

impl CotangentDelta {
    pub fn scale(self: Self, rhs: f64) -> CotangentDelta {
        CotangentDelta {
            covec: self.covec.scale(rhs),
            pt: self.pt.scale(rhs),
        }
    }

    pub fn error(self: Self, rhs: Self) -> f64 {
        self.pt.error(rhs.pt)
            .max(self.covec.error(rhs.covec))
    }
}

impl Cotangent {
    #[cfg(test)]
    pub fn error(this: &Cotangent, other: &Cotangent) -> f64 {
        (Pt::error(&this.pt, &other.pt) + this.covec.error(other.covec)) / 2.0
    }

    pub fn modulus(self: Self) -> f64 {
        let qk = self.covec.dot(self.pt.incoming());
        let h = ks_scalar(self.pt.coord.explode().3, self.pt.radius);
        self.covec.dot(self.covec * Quad::ETA) - 2.0*h*(qk*qk)
    }

    pub fn dual(self: Self) -> Tangent {
        let k = self.pt.incoming();
        let qk = self.covec.dot(self.pt.incoming());
        let h = 2.0 * ks_scalar(self.pt.coord.explode().3, self.pt.radius) * qk;
        Tangent {
            vec: self.covec * Quad::ETA - k.scale(h),
            pt: self.pt,
        }
    }

    pub fn flip(self: Self) -> Self {
        let (t, x, y, z) = self.pt.coord.explode();
        let (dt, dx, dy, dz) = self.covec.explode();
        let (_, cov) = flip_cotangent(
            t, x, y, z, self.pt.radius.is_sign_negative(),
            dt, dx, dy, dz
        );
        Cotangent {
            covec: cov,
            pt: self.pt.flip(),
        }
    }

    pub fn energy(self: Self) -> f64 {
        - self.covec.explode().0
    }

    pub fn angular(self: Self) -> f64 {
        let (_, x, y, _) = self.pt.coord.explode();
        let (_, px, py, _) = self.covec.explode();
        x * py - y * px
    }

    pub fn carter(self: Self) -> f64 {
        // -2 Sigma ⟨k, p⟩⟨ℓ, p⟩/⟨k, l⟩ + r^2 g(p, p)
        let r = self.pt.radius;
        let z = self.pt.coord.explode().3;
        let k = self.pt.incoming();
        let l = self.pt.outgoing();
        let pk = self.covec.dot(k);
        let pl = self.covec.dot(l);
        let kl = self.pt.incoming_outgoing_dot();
        let sigma = r*r + (SPIN*SPIN) * (z*z) / (r*r);
        r*r * self.modulus() - 2.0 * sigma * pk * pl / kl
    }

    pub fn dynamics(self: Self) -> CotangentDelta {
        // dx = g(p, -)
        // dp = -.5 * ∂g/∂x (p, p)
        let (_, x, y, z) = self.pt.coord.explode();
        let (pt, px, py, pz) = self.covec.explode();
        let (dpx, dpy, dpz) = diff_inv_sq(
            x, y, z,
            self.pt.radius.is_sign_negative(),
            pt, px, py, pz
        );
        CotangentDelta {
            covec: Quad::new(0.0, -0.5*dpx, -0.5*dpy, -0.5*dpz),
            pt: self.dual().vec,
        }
    }

    /// Applying the cotangent delta data (without rescaling)
    pub fn nudge(self: Self, delta: CotangentDelta) -> Self {
        Cotangent {
            covec: self.covec + delta.covec,
            pt: self.pt.nudge(delta.pt),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::*;
    use crate::coordinates::kerr_schild::*;

    fn rand_pt() -> Pt {
        Pt::new(
            Quad::rand(),
            random_range(-5..5),
            random_bool(0.5),
            random_bool(0.5),
            random_bool(0.5)
        )
    }

    const NUM: i32 = 100;
    const ERR: f64 = 1e-13;

    #[test]
    fn flip_flip() {
        let mut pt_err = 0.0;
        let mut tg_err = 0.0;
        let mut ct_err = 0.0;
        for _ in 0..NUM {
            let pt = rand_pt();
            let vec = Tangent {
                vec: Quad::rand(),
                pt: rand_pt(),
            };
            let covec = Cotangent {
                covec: Quad::rand(),
                pt: rand_pt(),
            };
            pt_err += Pt::error(&pt, &pt.flip().flip());
            tg_err += Tangent::error(&vec, &vec.flip().flip());
            ct_err += Cotangent::error(&covec, &covec.flip().flip());
        }
        assert!(pt_err < ERR * NUM as f64);
        assert!(tg_err < ERR * NUM as f64);
        assert!(ct_err < ERR * NUM as f64);
    }

    #[test]
    fn flip_region() {
        for _ in 0..NUM {
            let pt = rand_pt();
            assert_eq!(pt.region(), pt.flip().region());
        }
    }

    #[test]
    fn dual_dual() {
        let mut tg_err = 0.0;
        let mut ct_err = 0.0;
        for _ in 0..NUM {
            let vec = Tangent {
                vec: Quad::rand(),
                pt: rand_pt(),
            };
            let covec = Cotangent {
                covec: Quad::rand(),
                pt: rand_pt(),
            };
            tg_err += Tangent::error(&vec, &vec.dual().dual());
            ct_err += Cotangent::error(&covec, &covec.dual().dual());
        }
        assert!(tg_err < ERR * NUM as f64);
        assert!(ct_err < ERR * NUM as f64);
    }

    #[test]
    fn dual_flip() {
        let mut tg_err = 0.0;
        let mut ct_err = 0.0;
        for _ in 0..NUM {
            let vec = Tangent {
                vec: Quad::rand(),
                pt: rand_pt(),
            };
            let covec = Cotangent {
                covec: Quad::rand(),
                pt: rand_pt(),
            };
            ct_err += Cotangent::error(&vec.flip().dual(), &vec.dual().flip());
            tg_err += Tangent::error(&covec.flip().dual(), &covec.dual().flip());
        }
        assert!(tg_err < ERR * NUM as f64);
        assert!(ct_err < ERR * NUM as f64);
    }

    #[test]
    fn dual_modulus() {
        let mut tg_err = 0.0;
        let mut ct_err = 0.0;
        for _ in 0..NUM {
            let vec = Tangent {
                vec: Quad::rand(),
                pt: rand_pt(),
            };
            let covec = Cotangent {
                covec: Quad::rand(),
                pt: rand_pt(),
            };
            tg_err += (vec.modulus() - vec.dual().modulus()).abs();
            ct_err += (covec.modulus() - covec.dual().modulus()).abs();
        }
        assert!(tg_err < ERR * NUM as f64);
        assert!(ct_err < ERR * NUM as f64);
    }

    #[test]
    fn flip_modulus() {
        let mut tg_err = 0.0;
        let mut ct_err = 0.0;
        for _ in 0..NUM {
            let vec = Tangent {
                vec: Quad::rand(),
                pt: rand_pt(),
            };
            let covec = Cotangent {
                covec: Quad::rand(),
                pt: rand_pt(),
            };
            tg_err += (vec.modulus() - vec.flip().modulus()).abs();
            ct_err += (covec.modulus() - covec.flip().modulus()).abs();
        }
        assert!(tg_err < ERR * NUM as f64, "{tg_err}");
        assert!(ct_err < ERR * NUM as f64, "{ct_err}");
    }

    #[test]
    fn principal_null() {
        let mut k_err = 0.0;
        let mut l_err = 0.0;
        for _ in 0..NUM {
            let pt = rand_pt();
            let k = Tangent { vec: pt.incoming(), pt };
            let cok = Cotangent { covec: pt.incoming_dual(), pt };
            k_err += Cotangent::error(&k.dual(), &cok);
            let l = Tangent { vec: pt.outgoing(), pt };
            let col = Cotangent { covec: pt.outgoing_dual(), pt };
            l_err += Cotangent::error(&l.dual(), &col);
        }
        assert!(k_err < ERR * NUM as f64);
        assert!(l_err < ERR * NUM as f64);
    }

    #[test]
    fn principal_null_flip() {
        let mut err = 0.0;
        for _ in 0..NUM {
            let pt = rand_pt();
            let pt1 = pt.flip();
            let k = Tangent { vec: pt.incoming(), pt };
            let l = Tangent { vec: pt1.outgoing(), pt: pt1 };
            err += Tangent::error(&k.flip(), &l);
        }
        assert!(err < ERR * NUM as f64);
    }

    #[test]
    fn principal_dot() {
        let mut err = 0.0;
        for _ in 0..NUM {
            let pt = rand_pt();
            err += (pt.incoming_outgoing_dot()
                - pt.incoming().dot(pt.outgoing_dual()))
                .abs();
        }
        assert!(err < ERR * NUM as f64, "{err}");
    }

    #[test]
    fn principal_null_modulus() {
        let mut k_err = 0.0;
        let mut l_err = 0.0;
        for _ in 0..NUM {
            let pt = rand_pt();
            let k = Tangent { vec: pt.incoming(), pt };
            k_err += k.modulus().abs();
            let l = Tangent { vec: pt.outgoing(), pt };
            l_err += l.modulus().abs();
        }
        assert!(k_err < ERR * NUM as f64);
        assert!(l_err < ERR * NUM as f64);
    }

    #[test]
    fn rk45_conserved() {
        let cov = Tangent {
            vec: Quad::new(1.0, 0.0, -0.45, 0.0),
            pt: Pt::new(
                Quad::new(0.0, 0.0003, -0.0002, 5.0),
                0, false, false, false
            ),
        }.dual();
        for (i, (t, st)) in rk45(cov).enumerate() {
            if t > 10000.0 || i > 1000_000 {
                assert!((st.modulus() - cov.modulus()).abs() < 1e-12);
                assert!((st.energy() - cov.energy()).abs() < 1e-12);
                assert!((st.angular() - cov.angular()).abs() < 1e-12);
                assert!((st.carter() - cov.carter()).abs() < 1e-10);
                break;
            }
        }
    }
}

/// Integrates the geodesics in Kerr–Schild coordinates specifically,
/// using the flipping mechanism to select coordinate charts.
pub fn rk45(state: Cotangent) -> impl Iterator<Item = (f64, Cotangent)> {
    // Assumes we are future-directed and timelike
    let mut state = state;
    let mut eps = 1e-3;
    let mut cur = 0.0;
    let mut flipped = false;
    std::iter::from_fn(move || loop {
        // If we are in the inner horizon and future directed
        // or if we are in the outer horizon and past directed
        // we switch immediately
        if eps < 1e-3 {  // Only consider switching when we are slowing down
        if state.pt.radius <= R_INNER {
            if !state.pt.time_rev {
                flipped = !flipped;
                state = state.flip();
            }
        } else if state.pt.radius >= R_OUTER {
            if state.pt.time_rev {
                flipped = !flipped;
                state = state.flip();
            }
        } else {
            /* Otherwise, switching happens in between the two horizons
            From geodesics equations we know the term that blows up is
                a/(Delta, negative) * (2m r E - aL)
            So we want to check if  a * (2 m r_horizon E - a L)  is positive
            Flip otherwise.

            (This is the same as dotting with horizon generating vector fields)
            TODO figure out "hovering" geodesics
            */
            let should_flip = SPIN.is_sign_positive() ^
                (2.0*MASS*(if state.pt.time_rev {R_OUTER} else {R_INNER})*state.energy()
                - SPIN*state.angular()).is_sign_positive() ^
                state.pt.time_rev;
            if should_flip {
                flipped = !flipped;
                state = state.flip();
            }
        }
        }


        let k1 = state.dynamics().scale(eps);
        let k2 = state.nudge(k1.scale(1./4.)).dynamics().scale(eps);
        let k3 = state.nudge(k1.scale(3./32.) + k2.scale(9./32.)).dynamics().scale(eps);
        let k4 = state.nudge(k1.scale(1932./2197.) + k2.scale(-7200./2197.) + k3.scale(7296./2197.)).dynamics().scale(eps);
        let k5 = state.nudge(k1.scale(439./216.) + k2.scale(-8.) + k3.scale(3680./513.) + k4.scale(-845./4104.)).dynamics().scale(eps);
        let k6 = state.nudge(k1.scale(-8./27.) + k2.scale(2.) + k3.scale(-3544./2565.) + k4.scale(1859./4104.) + k5.scale(-11./40.)).dynamics().scale(eps);

        let r5 = k1.scale(25./216.) + k3.scale(1408./2565.) + k4.scale(2197./4104.) + k5.scale(-1./5.);
        let r6 = k1.scale(16./135.) + k3.scale(6656./12825.) + k4.scale(28561./56430.) + k5.scale(-9./50.) + k6.scale(2./55.);

        let error = r5.error(r6);
        eps *= (0.8 * f64::powf(RK_TOLERANCE / error, 1./5.))
            .max(0.2).min(2.0);
        if error >= RK_TOLERANCE {
            if eps < 1e-12 {
                panic!("Step size is too small: {eps}")
            }
            continue;
        }
        // println!("{eps}");
        cur += eps;
        state = state.nudge(r6);
        // TODO output boyer lindquist coordinates
        return Some((cur, if flipped { state.flip() } else { state }));
    })
}

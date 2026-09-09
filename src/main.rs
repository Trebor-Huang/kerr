#![allow(mixed_script_confusables, unused)]
#![feature(autodiff)]

use std::{autodiff::*, io, io::*};

const MASS: f64 = 1.0;
const SPIN: f64 = 0.8;
const R_OUTER: f64 = 1.6; // MASS + f64::sqrt(MASS*MASS - SPIN*SPIN);
const R_INNER: f64 = 0.4; // MASS - f64::sqrt(MASS*MASS - SPIN*SPIN);

enum RegionType {
    Us, Para,
    AfterOuter, AfterInner,
    UsSing, ParaSing
}
type Region = (i32, RegionType);

type Quad = (f64, f64, f64, f64);

#[derive(Debug, Clone, Copy)]
struct Pt {
    coord: Quad,
    base: i32,
    parallel: bool,
    time_rev: bool,
    radius: f64,
}

fn radius(x: f64, y: f64, z: f64, rev: bool) -> f64 {
    let d = x*x + y*y + z*z - SPIN*SPIN;
    let result = f64::sqrt(0.5 * (d + f64::sqrt(d*d + 4.0 * (SPIN*SPIN) * (z*z))));
    if rev {
        - result
    } else {
        result
    }
}

#[autodiff_forward(_fflip, Dual, Dual, Dual, Dual, Const, Dual, Dual, Dual)]
#[autodiff_reverse(_rflip, Active, Active, Active, Active, Const, Duplicated, Duplicated, Duplicated)]
fn _flip(
    t0: f64, x0: f64, y0: f64, z0: f64, rev: bool,
    t: &mut f64, x: &mut f64, y: &mut f64
) {
    let r0 = radius(x0, y0, z0, rev);
    let δφ = 2.0 * f64::atan2(SPIN, r0)
        + SPIN / f64::sqrt(MASS*MASS - SPIN*SPIN) *
        f64::ln(f64::abs((r0 - R_OUTER)/(r0 - R_INNER)));
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
    ((t, x, y), (d_dt0, d_dx0, d_dy0, d_dz0 + d_dz))
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

fn ks_scalar(z: f64, r: f64) -> f64 {
    MASS * (r.powi(3))/(r.powi(4) + (SPIN*SPIN) * (z*z))
}

impl Pt {
    fn new(coord: Quad, base: i32, parallel: bool, time_rev: bool, radius_rev: bool) -> Pt {
        Pt {
            coord, base, parallel, time_rev,
            radius: radius(coord.1, coord.2, coord.3, radius_rev),
        }
    }

    fn region(self: &Self) -> Region {
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

    fn flip(self: &Self) -> Self {
        let (t,x,y) = flip(
            self.coord.0, self.coord.1, self.coord.2, self.coord.3,
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
        let parallel = is_middle == self.parallel;
        let time_rev = is_middle == self.time_rev;
        Pt {
            coord: (t, x, y, self.coord.3),
            base,
            parallel,
            time_rev,
            radius: self.radius,
        }
    }

    /// Nudges the coordinate in a direction. If it went through the ring,
    /// flip the sign of the radius.
    fn nudge(self: &Self, delta: Quad) -> Self {
        let (t,x,y,z) = self.coord;
        let (dt, dx, dy, dz) = delta;
        let t1 = t + dt;
        let x1 = x + dx;
        let y1 = y + dy;
        let z1 = z + dz;
        if z.signum() * z1.signum() == -1.0 {
            let u = - z / dz;
            let x0 = x + u * dx;
            let y0 = y + u * dy;
            if x0*x0 + y0*y0 < SPIN*SPIN {
                return Pt::new(
                    (t1, x1, y1, z1),
                    self.base, self.parallel, self.time_rev,
                    !self.radius.is_sign_negative()
                )
            }
        }
        return Pt::new(
            (t1, x1, y1, z1),
            self.base, self.parallel, self.time_rev,
            self.radius.is_sign_negative()
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Tangent {
    vec: Quad,
    pt: Pt,
}

impl Tangent {
    fn modulus(self: &Self) -> f64 {
        let r = self.pt.radius;
        let ra = r*r + SPIN*SPIN;
        let (_t, x, y, z) = self.pt.coord;
        let (qt, qx, qy, qz) = self.vec;
        let qk = qt + qx * (r*x + SPIN*y)/ra + qy * (r*y - SPIN*x)/ra + qz * z/r;
        let h = ks_scalar(z, r);
        qx*qx + qy*qy + qz*qz - qt*qt + 2.0*h*(qk*qk)
    }

    fn dual(self: &Self) -> Cotangent {
        let r = self.pt.radius;
        let ra = r*r + SPIN*SPIN;
        let (_t, x, y, z) = self.pt.coord;
        let (qt, qx, qy, qz) = self.vec;
        let qk = qt + qx * (r*x + SPIN*y)/ra + qy * (r*y - SPIN*x)/ra + qz * z/r;
        let h = ks_scalar(z, r);
        Cotangent {
            covec: (
                - qt + 2.0 * h * qk,
                qx + 2.0 * h * qk * (r*x + SPIN*y)/ra,
                qy + 2.0 * h * qk * (r*y - SPIN*x)/ra,
                qz + 2.0 * h * qk * z/r
            ),
            pt: self.pt,
        }
    }

    fn flip(self: &Self) -> Self {
        let (t, x, y, z) = self.pt.coord;
        let (dt, dx, dy, dz) = self.vec;
        let (_, (dt1, dx1, dy1)) = flip_tangent(
            t, x, y, z, self.pt.radius.is_sign_negative(),
            dt, dx, dy, dz
        );
        Tangent {
            vec: (dt1, dx1, dy1, dz),
            pt: self.pt.flip(),
        }
    }

    // The point given by moving along the tangent vector direction for delta
    fn nudge(self: &Self, delta: f64) -> Pt {
        self.pt.nudge((
            self.vec.0 * delta,
            self.vec.1 * delta,
            self.vec.2 * delta,
            self.vec.3 * delta
        ))
    }
}

#[derive(Clone, Copy)]
struct Cotangent {
    covec: Quad,
    pt: Pt,
}
struct CotangentDelta {
    covec: Quad,
    pt: Quad
}

impl std::ops::Add for CotangentDelta {
    type Output = CotangentDelta;

    fn add(self: CotangentDelta, rhs: CotangentDelta) -> CotangentDelta {
        CotangentDelta {
            covec: (
                self.covec.0 + rhs.covec.0,
                self.covec.1 + rhs.covec.1,
                self.covec.2 + rhs.covec.2,
                self.covec.3 + rhs.covec.3,
            ),
            pt: (
                self.pt.0 + rhs.pt.0,
                self.pt.1 + rhs.pt.1,
                self.pt.2 + rhs.pt.2,
                self.pt.3 + rhs.pt.3,
            ),
        }
    }
}

impl CotangentDelta {
    fn scale(self: &Self, rhs: f64) -> CotangentDelta {
        CotangentDelta {
            covec: (
                rhs * self.covec.0,
                rhs * self.covec.1,
                rhs * self.covec.2,
                rhs * self.covec.3,
            ),
            pt: (
                rhs * self.pt.0,
                rhs * self.pt.1,
                rhs * self.pt.2,
                rhs * self.pt.3,
            ),
        }
    }

    fn error(self: &Self, rhs: &Self) -> f64 {
        (self.pt.0 - rhs.pt.0).abs()
            .max((self.pt.1 - rhs.pt.1).abs())
            .max((self.pt.2 - rhs.pt.2).abs())
            .max((self.pt.3 - rhs.pt.3).abs())
            .max((self.covec.0 - rhs.covec.0).abs())
            .max((self.covec.1 - rhs.covec.1).abs())
            .max((self.covec.2 - rhs.covec.2).abs())
            .max((self.covec.3 - rhs.covec.3).abs())
    }
}

impl Cotangent {
    fn modulus(self: &Self) -> f64 {
        let r = self.pt.radius;
        let ra = r*r + SPIN*SPIN;
        let (_t, x, y, z) = self.pt.coord;
        let (qt, qx, qy, qz) = self.covec;
        let qk = qt - qx * (r*x + SPIN*y)/ra - qy * (r*y - SPIN*x)/ra - qz * z/r;
        let h = ks_scalar(z, r);
        qx*qx + qy*qy + qz*qz - qt*qt - 2.0*h*(qk*qk)
    }

    fn dual(self: &Self) -> Tangent {
        let r = self.pt.radius;
        let ra = r*r + SPIN*SPIN;
        let (_t, x, y, z) = self.pt.coord;
        let (qt, qx, qy, qz) = self.covec;
        let qk = qt - qx * (r*x + SPIN*y)/ra - qy * (r*y - SPIN*x)/ra - qz * z/r;
        let h = ks_scalar(z, r);
        Tangent {
            vec: (
                - qt - 2.0 * h * qk,
                qx + 2.0 * h * qk * (r*x + SPIN*y)/ra,
                qy + 2.0 * h * qk * (r*y - SPIN*x)/ra,
                qz + 2.0 * h * qk * z/r
            ),
            pt: self.pt,
        }
    }

    fn flip(self: &Self) -> Self {
        let (t, x, y, z) = self.pt.coord;
        let (dt, dx, dy, dz) = self.covec;
        let (_, (dt1, dx1, dy1, dz1)) = flip_cotangent(
            t, x, y, z, self.pt.radius.is_sign_negative(),
            dt, dx, dy, dz
        );
        Cotangent {
            covec: (dt1, dx1, dy1, dz1),
            pt: self.pt.flip(),
        }
    }

    fn energy(self: &Self) -> f64 {
        - self.covec.0
    }

    fn angular(self: &Self) -> f64 {
        self.pt.coord.1 * self.covec.2 - self.pt.coord.2 * self.covec.1
    }

    fn leapfrog(self: &Self, delta: f64) -> Self {
        let new_pt = self.dual().nudge(delta);
        let (dpx, dpy, dpz) = diff_inv_sq(
            self.pt.coord.1, self.pt.coord.2, self.pt.coord.3,
            self.pt.radius.is_sign_negative(),
            self.covec.0, self.covec.1, self.covec.2, self.covec.3
        );
        Cotangent {
            covec: (
                self.covec.0,
                self.covec.1 - 0.5 * dpx * delta,
                self.covec.2 - 0.5 * dpy * delta,
                self.covec.3 - 0.5 * dpz * delta,
            ),
            pt: new_pt,
        }
    }

    fn dynamics(self: &Self) -> CotangentDelta {
        // dx = g(p, -)
        // dp = -.5 * ∂g/∂x (p, p)
        let (dpx, dpy, dpz) = diff_inv_sq(
            self.pt.coord.1, self.pt.coord.2, self.pt.coord.3,
            self.pt.radius.is_sign_negative(),
            self.covec.0, self.covec.1, self.covec.2, self.covec.3
        );
        CotangentDelta {
            covec: (0.0, -0.5*dpx, -0.5*dpy, -0.5*dpz),
            pt: self.dual().vec,
        }
    }

    /// Applying the cotangent delta data (without rescaling)
    fn nudge(self: &Self, delta: CotangentDelta) -> Self {
        Cotangent {
            covec: (
                self.covec.0 + delta.covec.0,
                self.covec.1 + delta.covec.1,
                self.covec.2 + delta.covec.2,
                self.covec.3 + delta.covec.3
            ),
            pt: self.pt.nudge(delta.pt),
        }
    }
}

const TOLERANCE: f64 = 1e-14;
fn rk45(state: Cotangent) -> impl Iterator<Item = (f64, Cotangent)> {
    let mut state = state;
    let mut eps = 1e-3;
    let mut cur = 0.0;
    std::iter::from_fn(move || loop {
        let k1 = state.dynamics().scale(eps);
        let k2 = state.nudge(k1.scale(1./4.)).dynamics().scale(eps);
        let k3 = state.nudge(k1.scale(3./32.) + k2.scale(9./32.)).dynamics().scale(eps);
        let k4 = state.nudge(k1.scale(1932./2197.) + k2.scale(-7200./2197.) + k3.scale(7296./2197.)).dynamics().scale(eps);
        let k5 = state.nudge(k1.scale(439./216.) + k2.scale(-8.) + k3.scale(3680./513.) + k4.scale(-845./4104.)).dynamics().scale(eps);
        let k6 = state.nudge(k1.scale(-8./27.) + k2.scale(2.) + k3.scale(-3544./2565.) + k4.scale(1859./4104.) + k5.scale(-11./40.)).dynamics().scale(eps);

        let r5 = k1.scale(25./216.) + k3.scale(1408./2565.) + k4.scale(2197./4104.) + k5.scale(-1./5.);
        let r6 = k1.scale(16./135.) + k3.scale(6656./12825.) + k4.scale(28561./56430.) + k5.scale(-9./50.) + k6.scale(2./55.);

        let error = r5.error(&r6);
        eps *= (0.9 * f64::powf(TOLERANCE / error, 1./5.))
            .max(0.2).min(2.0);
        if error >= TOLERANCE {
            continue;
        }
        // println!("{eps}");
        cur += eps;
        state = state.nudge(r6);
        return Some((cur, state));
    })
}

fn main() {
    let mut stderr = io::stderr();
    let cov = Tangent {
        vec: (6.0, 0.0, 0.5, 0.0),
        pt: Pt::new(
            (0.0, 50.0, 0.0, 1.0),
            0, false, false,false
        ),
    }.dual();
    let time = std::time::SystemTime::now();
    println!("{:}", cov.angular());
    for (t, st) in rk45(cov) {
        writeln!(stderr, "{:.7} {:.7} {:.7}", st.modulus(), st.energy(), st.angular()).unwrap();
        if t > 1000.0 { break; }
    }
    println!("Elapsed: {:?}", time.elapsed());
}

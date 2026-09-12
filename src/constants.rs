pub const MASS: f64 = 1.0;
pub const SPIN: f64 = 0.8;
const SQRTMA: f64 = 0.6; // f64::sqrt(MASS*MASS - SPIN*SPIN)
pub const R_OUTER: f64 = MASS + SQRTMA;
pub const R_INNER: f64 = MASS - SQRTMA;
/// Surface gravity at the outer horizon
pub const KAPPA_OUTER: f64 = SQRTMA / (2.0 * MASS * R_OUTER);
/// Surface gravity at the inner horizon
pub const KAPPA_INNER: f64 = SQRTMA / (2.0 * MASS * R_INNER);
/// Angular velocity of outer horizon
pub const OMEGA_OUTER: f64 = SPIN / (2.0 * MASS * R_OUTER);
/// Angular velocity of inner horizon
pub const OMEGA_INNER: f64 = SPIN / (2.0 * MASS * R_INNER);

/// Tolerance of the integrator
pub const RK_TOLERANCE: f64 = 1e-14;

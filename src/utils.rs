#[derive(Debug, Clone, Copy)]
pub struct Quad(f64, f64, f64, f64);

impl std::ops::Add for Quad {
    type Output = Quad;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Quad(
            self.0 + rhs.0,
            self.1 + rhs.1,
            self.2 + rhs.2,
            self.3 + rhs.3,
        )
    }
}

impl std::ops::Sub for Quad {
    type Output = Quad;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Quad(
            self.0 - rhs.0,
            self.1 - rhs.1,
            self.2 - rhs.2,
            self.3 - rhs.3,
        )
    }
}

impl std::ops::Mul for Quad {
    type Output = Quad;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Quad(
            self.0 * rhs.0,
            self.1 * rhs.1,
            self.2 * rhs.2,
            self.3 * rhs.3,
        )
    }
}

impl Quad {
    pub const ETA: Quad = Quad(-1.0, 1.0, 1.0, 1.0);

    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Quad {
        Quad(t,x,y,z)
    }

    #[inline]
    pub fn explode(self) -> (f64, f64, f64, f64) {
        (self.0, self.1, self.2, self.3)
    }

    #[inline]
    pub fn scale(self: Quad, scale: f64) -> Quad {
        Quad(
            self.0 * scale,
            self.1 * scale,
            self.2 * scale,
            self.3 * scale
        )
    }

    #[inline]
    pub fn dot(self: Quad, other: Quad) -> f64 {
        self.0 * other.0 +
        self.1 * other.1 +
        self.2 * other.2 +
        self.3 * other.3
    }

    #[inline]
    pub fn error(self: Quad, other: Quad) -> f64 {
        ((self.0 - other.0).abs()
        + (self.1 - other.1).abs()
        + (self.2 - other.2).abs()
        + (self.3 - other.3).abs()) / 4.0
    }

    #[cfg(test)]
    pub fn rand() -> Quad {
        use rand::*;
        Quad(
            random_range(-10.0..10.0),
            random_range(-10.0..10.0),
            random_range(-10.0..10.0),
            random_range(-10.0..10.0)
        )
    }
}

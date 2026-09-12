pub type Quad = (f64, f64, f64, f64); // TODO make this a proper type, use wide

#[inline]
pub fn quad_dot(this: Quad, other: Quad) -> f64 {
    this.0 * other.0 +
    this.1 * other.1 +
    this.2 * other.2 +
    this.3 * other.3
}

#[cfg(test)]
pub fn quad_error(this: Quad, other: Quad) -> f64 {
     ((this.0 - other.0).abs()
    + (this.1 - other.1).abs()
    + (this.2 - other.2).abs()
    + (this.3 - other.3).abs()) / 4.0
}

#[cfg(test)]
pub fn rand_quad() -> Quad {
    use rand::*;
    (
        random_range(-10.0..10.0),
        random_range(-10.0..10.0),
        random_range(-10.0..10.0),
        random_range(-10.0..10.0)
    )
}

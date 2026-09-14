#![allow(mixed_script_confusables, unused)]
#![feature(autodiff)]

mod constants;
mod utils;
mod coordinates;

use std::io::*;
use crate::{constants::*, utils::*};

fn main() {
    use coordinates::kerr_schild::*;
    use coordinates::boyer_lindquist;

    let lock = stdout().lock();
    let mut stdout = BufWriter::new(lock);
    let cov = Tangent {
        vec: Quad::new(1.0, -0.6, 0.05, -0.2),
        pt: Pt::new(Quad::new(0.0, 8.0, 0.0, 5.0),
            0, false, false, true),
    }.dual();
    eprintln!("Modulus: {:}", cov.modulus());
    if cov.modulus() > 0.0 {
        panic!("Velocity is spacelike");
    }
    let time = std::time::SystemTime::now();
    for (i, (t, st)) in rk45(cov).enumerate() {
        let (_, x, y, z) = boyer_lindquist::Pt::from_kerr_schild(st.pt).coord.explode();
        writeln!(stdout, "{i},{:},{:},{:},{:},{:}", t, x, y, z, st.pt.radius()).unwrap();
        if i % 100 == 0 {
            eprintln!("#{i:>5}  t={:>5.2}  r={:>5.2}  m={:>10.7}  E={:>10.7}  L={:>10.7}  Q={:>10.7}  future: {}",
                t, st.pt.radius(),
                st.modulus(), st.energy() * st.pt.sign(), st.angular() * st.pt.sign(), st.carter(),
                st.future_directed()
            );
        }
        if t > 100.0 { break; }
    }
    stdout.flush().unwrap();
    eprintln!("Elapsed: {:?}", time.elapsed());
}

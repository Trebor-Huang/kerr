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

    let mut stderr = stderr();
    let lock = stdout().lock();
    let mut stdout = BufWriter::new(lock);
    let cov = Cotangent {
        covec: Quad::new(0.0, -1.0, 0.01, 0.0),
        pt: Pt::new(
            Quad::new(0.0, 1.4, 0.0, 0.0),
            0, false, true, false
        ),
    };
    let std_cov = Tangent {
        vec: Quad::new(1.0, 0.0, -0.45, 0.0),
        pt: Pt::new(
            Quad::new(0.0, 0.0003, -0.0002, 5.0),
            0, false, false, false
        ),
    }.dual();
    writeln!(stderr, "Modulus: {:}", cov.modulus()).unwrap();
    if cov.modulus() > 0.0 {
        return;
    }
    let rotor_outer = 2.0 * MASS * R_OUTER * cov.energy()
        - SPIN * cov.angular();
    let rotor_inner = 2.0 * MASS * R_INNER * cov.energy()
        - SPIN * cov.angular();
    writeln!(stderr, "Rotor: {rotor_outer:.3}, {rotor_inner:.3}").unwrap();
    let time = std::time::SystemTime::now();
    for (i, (t, st)) in rk45(cov).enumerate() {
        let (_, x, y, z) = boyer_lindquist::Pt::from_kerr_schild(st.pt).coord.explode();
        writeln!(stdout, "{i},{:},{:},{:},{:},{:}", t, x, y, z, st.pt.radius()).unwrap();
        if i % 100 == 0 {
            writeln!(stderr, "#{i:>5}  t={:>5.2}  r={:>5.2}  m={:>10.7}  E={:>10.7}  L={:>10.7}  Q={:>10.7}  future: {}",
                t, st.pt.radius(),
                st.modulus(), st.energy(), st.angular(), st.carter(),
                st.future_directed()
            );
        }
        if t > 20.0 { break; }
    }
    stdout.flush().unwrap();
    writeln!(stderr, "Elapsed: {:?}", time.elapsed()).unwrap();
}

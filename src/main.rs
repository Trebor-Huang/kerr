#![allow(mixed_script_confusables, unused)]
#![feature(autodiff)]

mod constants;
mod utils;
mod coordinates;

use std::io::*;
use crate::utils::*;

fn main() {
    use coordinates::kerr_schild::*;

    let mut stderr = stderr();
    let lock = stdout().lock();
    let mut stdout = BufWriter::new(lock);
    // let cov = Tangent {
    //     vec: Quad::new(1.3, -1.0, -0.2, 0.2),
    //     pt: Pt::new(
    //         Quad::new(0.0, 0.0, 1.2, 0.0),
    //         0, false, false, false
    //     ),
    // }.dual();
    let cov = Tangent {
        vec: Quad::new(1.0, 0.0, -0.45, 0.0),
        pt: Pt::new(
            Quad::new(0.0, 0.0003, -0.0002, 5.0),
            0, false, false, false
        ),
    }.dual();
    if cov.modulus() > 0.0 {
        writeln!(stderr, "Spacelike, {:}", cov.modulus()).unwrap();
        return;
    }
    let time = std::time::SystemTime::now();
    for (i, (t, st)) in rk45(cov).enumerate() {
        let (_, x, y, z) = st.pt.coord.explode();
        writeln!(stdout, "{i},{:},{:},{:},{:},{:}", t, x, y, z, st.pt.radius()).unwrap();
        if i % 100 == 0 {
            writeln!(stderr, "#{i:>5}  t={:>5.2}  m={:>10.7}  E={:>10.7}  L={:>10.7}  Q={:>10.7}",
                t,
                st.modulus(), st.energy(), st.angular(), st.carter()
            ).unwrap();
        }
        if t > 2000.0 || i > 1000000 { break; }
    }
    stdout.flush().unwrap();
    writeln!(stderr, "Elapsed: {:?}", time.elapsed()).unwrap();
}

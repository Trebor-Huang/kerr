#![allow(mixed_script_confusables, unused)]
#![feature(autodiff)]

mod constants;
mod utils;
mod coordinates;

use std::io::*;
use utils::*;
use constants::*;

fn main() {
    use coordinates::kerr_schild::*;

    let mut stderr = stderr();
    let lock = stdout().lock();
    let mut stdout = BufWriter::new(lock);
    let cov = Tangent {
        vec: (1.3, -1.0, -0.2, 0.2),
        pt: Pt::new(
            (0.0, 0.0, 1.2, 0.0),
            0, false, false, false
        ),
    }.dual();
    if cov.modulus() > 0.0 {
        writeln!(stderr, "Spacelike, {:}", cov.modulus()).unwrap();
        return;
    }
    let time = std::time::SystemTime::now();
    for (i, (t, st)) in rk45(cov).enumerate() {
        writeln!(stdout, "{i},{:},{:},{:},{:},{:}", t, st.pt.coord.1, st.pt.coord.2, st.pt.coord.3, st.pt.radius()).unwrap();
        if i % 100 == 0 {
            writeln!(stderr, "#{i:>5}  t={:>5.2}  m={:>10.7}  E={:>10.7}  L={:>10.7}  Q={:>10.7}",
                t,
                st.modulus(), st.energy(), st.angular(), st.carter()
            ).unwrap();
        }
        if t > 13.0 || i > 100000 { break; }
    }
    stdout.flush().unwrap();
    writeln!(stderr, "Elapsed: {:?}", time.elapsed()).unwrap();
}

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
    let cov = Cotangent { covec: Quad::new(9.051295820489578, 2.396341987716024, -1.5728614700585766, -2.5647394619273056), pt: Pt::new( Quad::new(-1.7971761623090856, -1.8123546170430807, 6.375473728148215, 7.406668036390435), 4, false, true, false ) };
    eprintln!("Modulus: {:}", cov.modulus());
    if cov.modulus() > 0.0 {
        panic!("Velocity is spacelike");
    }
    let time = std::time::SystemTime::now();
    for (i, (t, st)) in rk45(cov).enumerate() {
        let (_, x, y, z) = boyer_lindquist::Pt::from_kerr_schild(st.pt).coord.explode();
        writeln!(stdout, "{i},{:},{:},{:},{:},{:}", t, x, y, z, st.pt.radius()).unwrap();
        if i % 100000 == 0 {
            eprintln!("#{i:>5}  t={:>5.2}  r={:>5.2}  m={:>10.7}  E={:>10.7}  L={:>10.7}  Q={:>10.7}  future: {}",
                t, st.pt.radius(),
                st.modulus(), st.energy() * st.pt.sign(), st.angular() * st.pt.sign(), st.carter(),
                st.future_directed()
            );
        }
        if t > 1000.0 { break; }
    }
    stdout.flush().unwrap();
    eprintln!("Elapsed: {:?}", time.elapsed());
}

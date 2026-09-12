pub mod kerr_schild;
pub mod boyer_lindquist;

#[derive(Debug, PartialEq, Eq)]
pub enum RegionType {
    Us, Para,
    AfterOuter, AfterInner,
    UsSing, ParaSing
}
pub type Region = (i32, RegionType);

//! Compare the pure-Rust HDF5 readers hdf5-pure and rust-hdf5.
//!
//! usage:
//!   h5bench kallisto <reader> <abundance.h5> <out.tsv> <repeats>
//!   h5bench sc <reader> <file> <spec from ref_sc.py> <full|partial>
//! reader: hdf5-pure (File::open, buffers the whole file) | hdf5-pure-threads (File::open,
//! whole numeric datasets read as row ranges in H5BENCH_THREADS threads, default 8) | hdf5-pure-stream
//! | rust-hdf5-threads (the same application-level threading)
//! (File::open_streaming) | rust-hdf5 (built with or without its `parallel` feature;
//! the `par` feature of this crate turns it on). The `zlibrs` feature makes hdf5-pure inflate
//! with zlib-rs instead of miniz_oxide.
//!
//! kallisto: reads aux/ids, est_counts, aux/eff_lengths and every bootstrap/bs<k>,
//! writes them as a TSV and prints the median and minimum of `repeats` passes.
//! sc: reads the datasets listed in the spec (10x .h5 or AnnData .h5ad), all of them
//! or (partial) indptr plus the data/indices of 1000 cells from the middle, prints the
//! elapsed seconds, then checks every dataset's checksums against the spec (skipped
//! with H5BENCH_NO_CHECK set, so that peak RSS covers the read alone).
use std::time::Instant;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Debug>(what: &str) -> impl Fn(E) -> String + '_ {
    move |err| format!("{}: {:?}", what, err)
}

/// Typed values as read, before any conversion
enum Num {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Num {
    fn len(&self) -> usize {
        match self {
            Num::I32(v) => v.len(),
            Num::I64(v) => v.len(),
            Num::F32(v) => v.len(),
            Num::F64(v) => v.len(),
        }
    }
    /// Values as u64 bit patterns (ints sign-extended, floats as IEEE bits), as ref_sc.py
    fn bits(&self) -> Vec<u64> {
        match self {
            Num::I32(v) => v.iter().map(|&x| x as i64 as u64).collect(),
            Num::I64(v) => v.iter().map(|&x| x as u64).collect(),
            Num::F32(v) => v.iter().map(|&x| x.to_bits() as u64).collect(),
            Num::F64(v) => v.iter().map(|&x| x.to_bits()).collect(),
        }
    }
    fn into_i32(self) -> Vec<i32> {
        if let Num::I32(v) = self { v } else { unreachable!() }
    }
    fn into_i64(self) -> Vec<i64> {
        if let Num::I64(v) = self { v } else { unreachable!() }
    }
    fn into_f32(self) -> Vec<f32> {
        if let Num::F32(v) = self { v } else { unreachable!() }
    }
    fn into_f64(self) -> Vec<f64> {
        if let Num::F64(v) = self { v } else { unreachable!() }
    }
    fn to_i64(&self) -> Vec<i64> {
        match self {
            Num::I32(v) => v.iter().map(|&x| x as i64).collect(),
            Num::I64(v) => v.clone(),
            _ => panic!("indptr must be an integer dataset"),
        }
    }
}

enum Reader {
    Pure(hdf5_pure::File),
    /// hdf5-pure with application-level threads: a whole numeric dataset is read as
    /// row ranges in `n` threads sharing the (buffered) file
    PureThreads(hdf5_pure::File, usize),
    Rh(rust_hdf5::H5File),
    /// rust-hdf5 with the same application-level threading (without its `parallel`).
    /// H5File is not Sync (Rc inside, unless the `threadsafe` feature is on), so every
    /// thread opens the file itself
    RhThreads(rust_hdf5::H5File, String, usize),
}

fn env_threads() -> usize {
    std::env::var("H5BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// Read [0, len) of a 1-D dataset as `threads` row ranges in parallel and concatenate
fn threaded<T: Send>(len: u64, threads: usize, read: impl Fn(u64, u64) -> R<Vec<T>> + Sync) -> R<Vec<T>> {
    let step = len.div_ceil(threads as u64).max(1);
    let parts: Vec<R<Vec<T>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..len)
            .step_by(step as usize)
            .map(|lo| {
                let read = &read;
                s.spawn(move || read(lo, step.min(len - lo)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });
    let mut out = Vec::with_capacity(len as usize);
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

impl Reader {
    fn open(name: &str, path: &str) -> R<Self> {
        Ok(match name {
            "hdf5-pure" => Reader::Pure(hdf5_pure::File::open(path).map_err(e("open"))?),
            "hdf5-pure-stream" => Reader::Pure(hdf5_pure::File::open_streaming(path).map_err(e("open"))?),
            "hdf5-pure-threads" => Reader::PureThreads(hdf5_pure::File::open(path).map_err(e("open"))?, env_threads()),
            "rust-hdf5" => Reader::Rh(rust_hdf5::H5File::open(path).map_err(e("open"))?),
            "rust-hdf5-threads" => {
                Reader::RhThreads(rust_hdf5::H5File::open(path).map_err(e("open"))?, path.to_string(), env_threads())
            }
            _ => return Err(format!("unknown reader {}", name)),
        })
    }

    fn pure(&self) -> Option<&hdf5_pure::File> {
        match self {
            Reader::Pure(f) | Reader::PureThreads(f, _) => Some(f),
            _ => None,
        }
    }

    fn strings(&self, p: &str) -> R<Vec<String>> {
        match self {
            Reader::Rh(f) | Reader::RhThreads(f, _, _) => f.dataset(p).map_err(e(p))?.read_strings().map_err(e(p)),
            _ => self.pure().unwrap().dataset(p).map_err(e(p))?.read_string().map_err(e(p)),
        }
    }

    fn num(&self, p: &str, kind: &str) -> R<Num> {
        match self {
            Reader::PureThreads(f, n) => {
                // the threads share `f` (hdf5_pure::File is Sync)
                let d = f.dataset(p).map_err(e(p))?;
                let len = d.shape().map_err(e(p))?[0];
                let d = &d;
                Ok(match kind {
                    "i32" => Num::I32(threaded(len, *n, |lo, c| d.read_i32_rows(lo, c).map_err(e(p)))?),
                    "i64" => Num::I64(threaded(len, *n, |lo, c| d.read_i64_rows(lo, c).map_err(e(p)))?),
                    "f32" => Num::F32(threaded(len, *n, |lo, c| d.read_f32_rows(lo, c).map_err(e(p)))?),
                    _ => Num::F64(threaded(len, *n, |lo, c| d.read_f64_rows(lo, c).map_err(e(p)))?),
                })
            }
            Reader::RhThreads(f, path, n) => {
                let len = f.dataset(p).map_err(e(p))?.shape()[0] as u64;
                // a range read on a file opened by this thread
                let part = |lo: u64, c: u64| -> R<Num> {
                    Reader::Rh(rust_hdf5::H5File::open(path).map_err(e("open"))?).num_range(p, kind, lo, c)
                };
                Ok(match kind {
                    "i32" => Num::I32(threaded(len, *n, |lo, c| part(lo, c).map(Num::into_i32))?),
                    "i64" => Num::I64(threaded(len, *n, |lo, c| part(lo, c).map(Num::into_i64))?),
                    "f32" => Num::F32(threaded(len, *n, |lo, c| part(lo, c).map(Num::into_f32))?),
                    _ => Num::F64(threaded(len, *n, |lo, c| part(lo, c).map(Num::into_f64))?),
                })
            }
            Reader::Pure(f) => {
                let d = f.dataset(p).map_err(e(p))?;
                Ok(match kind {
                    "i32" => Num::I32(d.read_i32().map_err(e(p))?),
                    "i64" => Num::I64(d.read_i64().map_err(e(p))?),
                    "f32" => Num::F32(d.read_f32().map_err(e(p))?),
                    _ => Num::F64(d.read_f64().map_err(e(p))?),
                })
            }
            Reader::Rh(f) => {
                let d = f.dataset(p).map_err(e(p))?;
                Ok(match kind {
                    "i32" => Num::I32(d.read_raw::<i32>().map_err(e(p))?),
                    "i64" => Num::I64(d.read_raw::<i64>().map_err(e(p))?),
                    "f32" => Num::F32(d.read_raw::<f32>().map_err(e(p))?),
                    _ => Num::F64(d.read_raw::<f64>().map_err(e(p))?),
                })
            }
        }
    }

    /// Elements [lo, lo + n) of a 1-D dataset
    fn num_range(&self, p: &str, kind: &str, lo: u64, n: u64) -> R<Num> {
        match self {
            Reader::Pure(f) | Reader::PureThreads(f, _) => {
                let d = f.dataset(p).map_err(e(p))?;
                Ok(match kind {
                    "i32" => Num::I32(d.read_i32_rows(lo, n).map_err(e(p))?),
                    "i64" => Num::I64(d.read_i64_rows(lo, n).map_err(e(p))?),
                    "f32" => Num::F32(d.read_f32_rows(lo, n).map_err(e(p))?),
                    _ => Num::F64(d.read_f64_rows(lo, n).map_err(e(p))?),
                })
            }
            Reader::Rh(f) | Reader::RhThreads(f, _, _) => {
                let d = f.dataset(p).map_err(e(p))?;
                let (s, c) = ([lo as usize], [n as usize]);
                Ok(match kind {
                    "i32" => Num::I32(d.read_slice::<i32>(&s, &c).map_err(e(p))?),
                    "i64" => Num::I64(d.read_slice::<i64>(&s, &c).map_err(e(p))?),
                    "f32" => Num::F32(d.read_slice::<f32>(&s, &c).map_err(e(p))?),
                    _ => Num::F64(d.read_slice::<f64>(&s, &c).map_err(e(p))?),
                })
            }
        }
    }
}

/// (sum x, sum (i+1) x) mod 2^64, as ref_sc.py
fn sums(x: &[u64]) -> (u64, u64) {
    let mut s1 = 0u64;
    let mut s2 = 0u64;
    for (i, &v) in x.iter().enumerate() {
        s1 = s1.wrapping_add(v);
        s2 = s2.wrapping_add(v.wrapping_mul(i as u64 + 1));
    }
    (s1, s2)
}

fn string_bits(v: &[String]) -> Vec<u64> {
    v.join("\n").bytes().map(|b| b as u64).collect()
}

const BLOCK: usize = 1000;

fn run_sc(reader: &str, path: &str, spec: &str, mode: &str) -> R<String> {
    let lines: Vec<Vec<String>> = std::fs::read_to_string(spec)
        .map_err(e("spec"))?
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    enum Got {
        S(Vec<String>),
        N(Num),
    }
    let t = Instant::now();
    let f = Reader::open(reader, path)?;
    let mut got = Vec::new();
    let mut indptr: Vec<i64> = Vec::new();
    for l in &lines {
        let (p, kind, role) = (l[0].as_str(), l[1].as_str(), l[2].as_str());
        let g = match role {
            "str" => Got::S(f.strings(p)?),
            "indptr" => {
                let n = f.num(p, kind)?;
                indptr = n.to_i64();
                Got::N(n)
            }
            _ if mode == "full" => Got::N(f.num(p, kind)?),
            _ => {
                let s = (indptr.len() - 1) / 2;
                let (lo, hi) = (indptr[s], indptr[(s + BLOCK).min(indptr.len() - 1)]);
                Got::N(f.num_range(p, kind, lo as u64, (hi - lo) as u64)?)
            }
        };
        got.push(g);
    }
    let secs = t.elapsed().as_secs_f64();
    // H5BENCH_NO_CHECK: skip the checksums so that peak RSS reflects the read alone
    if std::env::var_os("H5BENCH_NO_CHECK").is_some() {
        std::hint::black_box(&got);
        return Ok(format!("seconds={:.3}\tunchecked", secs));
    }

    let mut bad = Vec::new();
    for (l, g) in lines.iter().zip(&got) {
        let (n, bits) = match g {
            Got::S(v) => (v.len(), string_bits(v)),
            Got::N(x) => (x.len(), x.bits()),
        };
        let (s1, s2) = sums(&bits);
        if n.to_string() != l[3] || s1.to_string() != l[4] || s2.to_string() != l[5] {
            bad.push(format!("{} (n {} vs {})", l[0], n, l[3]));
        }
    }
    let check = if bad.is_empty() { format!("identical ({} datasets)", lines.len()) } else { format!("DIFFERENT: {}", bad.join(", ")) };
    Ok(format!("seconds={:.3}\t{}", secs, check))
}

struct Kallisto {
    ids: Vec<String>,
    counts: Vec<f64>,
    eff_lengths: Vec<f64>,
    bootstraps: Vec<Vec<f64>>,
}

fn read_kallisto(reader: &str, path: &str) -> R<Kallisto> {
    let f = Reader::open(reader, path)?;
    let f64s = |p: &str| -> R<Vec<f64>> {
        match f.num(p, "f64")? {
            Num::F64(v) => Ok(v),
            _ => unreachable!(),
        }
    };
    let ids = f.strings("aux/ids")?;
    let counts = f64s("est_counts")?;
    let eff_lengths = f64s("aux/eff_lengths")?;
    let mut bootstraps = Vec::new();
    while let Ok(v) = f64s(&format!("bootstrap/bs{}", bootstraps.len())) {
        bootstraps.push(v);
    }
    Ok(Kallisto { ids, counts, eff_lengths, bootstraps })
}

fn run_kallisto(reader: &str, path: &str, out: &str, repeats: usize) -> R<String> {
    let k = read_kallisto(reader, path)?;
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        std::hint::black_box(read_kallisto(reader, path)?);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut s = String::from("target_id\test_counts\teff_length");
    for b in 0..k.bootstraps.len() {
        s.push_str(&format!("\tbs{}", b));
    }
    s.push('\n');
    for i in 0..k.ids.len() {
        s.push_str(&format!("{}\t{:?}\t{:?}", k.ids[i], k.counts[i], k.eff_lengths[i]));
        for b in &k.bootstraps {
            s.push_str(&format!("\t{:?}", b[i]));
        }
        s.push('\n');
    }
    std::fs::write(out, s).map_err(e("write"))?;
    Ok(format!(
        "transcripts={}\tbootstraps={}\tmedian_ms={:.2}\tmin_ms={:.2}",
        k.ids.len(),
        k.bootstraps.len(),
        times[times.len() / 2],
        times[0]
    ))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 || (a[1] != "kallisto" && a[1] != "sc") {
        eprintln!("usage: h5bench kallisto <reader> <abundance.h5> <out.tsv> <repeats>");
        eprintln!("       h5bench sc <reader> <file> <spec> <full|partial>");
        std::process::exit(2);
    }
    let mut label = a[2].clone();
    if a[2] == "rust-hdf5" && cfg!(feature = "par") {
        label.push_str("+parallel");
    }
    if a[2].starts_with("hdf5-pure") && cfg!(feature = "zlibrs") {
        label.push_str("+zlib-rs");
    }
    if a[2].ends_with("-threads") {
        label.push_str(&format!("({})", std::env::var("H5BENCH_THREADS").unwrap_or_else(|_| "8".into())));
    }
    let r = if a[1] == "kallisto" {
        run_kallisto(&a[2], &a[3], &a[4], a[5].parse().expect("repeats"))
    } else {
        run_sc(&a[2], &a[3], &a[4], &a[5])
    };
    match r {
        Ok(line) => println!("{}\tok\t{}", label, line),
        Err(msg) => {
            println!("{}\tFAILED\t{}", label, msg.replace('\n', " "));
            std::process::exit(1);
        }
    }
}

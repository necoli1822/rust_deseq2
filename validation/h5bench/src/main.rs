//! Compare pure-Rust HDF5 readers on kallisto `abundance.h5` files.
//!
//! usage: h5bench <crate> <abundance.h5> <out.tsv> <repeats>
//!   crate: hdf5-reader | hdf5-pure | rust-hdf5 | hdf5-rust
//!
//! Reads what tximport's read_kallisto_h5() reads (aux/ids, est_counts,
//! aux/eff_lengths) plus every bootstrap/bs<k>, writes them as a TSV (one row per
//! transcript, values as %.17g-equivalent shortest round-trip decimals), and
//! prints the median and minimum wall time of `repeats` full open-and-read passes.
use std::time::Instant;

struct Kallisto {
    ids: Vec<String>,
    counts: Vec<f64>,
    eff_lengths: Vec<f64>,
    bootstraps: Vec<Vec<f64>>,
}

type R<T> = Result<T, String>;

fn e<E: std::fmt::Debug>(what: &str) -> impl Fn(E) -> String + '_ {
    move |err| format!("{}: {:?}", what, err)
}

/// Fixed-length HDF5 strings: elements of `size` bytes, NUL- or space-padded
fn fixed_strings(bytes: &[u8], size: usize) -> Vec<String> {
    bytes
        .chunks(size)
        .map(|c| {
            let end = c.iter().position(|&b| b == 0).unwrap_or(c.len());
            String::from_utf8_lossy(&c[..end]).trim_end().to_string()
        })
        .collect()
}

fn read_hdf5_reader(path: &str) -> R<Kallisto> {
    use hdf5_reader::Hdf5File;
    let f = Hdf5File::open(path).map_err(e("open"))?;
    let num = |name: &str| -> R<Vec<f64>> {
        let a = f.dataset(name).map_err(e(name))?.read_array::<f64>().map_err(e(name))?;
        Ok(a.iter().copied().collect())
    };
    let ids = f.dataset("/aux/ids").map_err(e("ids"))?.read_strings().map_err(e("ids"))?;
    let counts = num("/est_counts")?;
    let eff_lengths = num("/aux/eff_lengths")?;
    let mut bootstraps = Vec::new();
    while let Ok(ds) = f.dataset(&format!("/bootstrap/bs{}", bootstraps.len())) {
        bootstraps.push(ds.read_array::<f64>().map_err(e("bs"))?.iter().copied().collect());
    }
    Ok(Kallisto { ids, counts, eff_lengths, bootstraps })
}

fn read_hdf5_pure(path: &str) -> R<Kallisto> {
    use hdf5_pure::File;
    let f = File::open(path).map_err(e("open"))?;
    let num = |name: &str| -> R<Vec<f64>> { f.dataset(name).map_err(e(name))?.read_f64().map_err(e(name)) };
    let ids = f.dataset("aux/ids").map_err(e("ids"))?.read_string().map_err(e("ids"))?;
    let counts = num("est_counts")?;
    let eff_lengths = num("aux/eff_lengths")?;
    let mut bootstraps = Vec::new();
    while let Ok(ds) = f.dataset(&format!("bootstrap/bs{}", bootstraps.len())) {
        bootstraps.push(ds.read_f64().map_err(e("bs"))?);
    }
    Ok(Kallisto { ids, counts, eff_lengths, bootstraps })
}

fn read_rust_hdf5(path: &str) -> R<Kallisto> {
    use rust_hdf5::H5File;
    let f = H5File::open(path).map_err(e("open"))?;
    let num = |name: &str| -> R<Vec<f64>> { f.dataset(name).map_err(e(name))?.read_raw::<f64>().map_err(e(name)) };
    let ids = f.dataset("aux/ids").map_err(e("ids"))?.read_strings().map_err(e("ids"))?;
    let counts = num("est_counts")?;
    let eff_lengths = num("aux/eff_lengths")?;
    let mut bootstraps = Vec::new();
    while let Ok(ds) = f.dataset(&format!("bootstrap/bs{}", bootstraps.len())) {
        bootstraps.push(ds.read_raw::<f64>().map_err(e("bs"))?);
    }
    Ok(Kallisto { ids, counts, eff_lengths, bootstraps })
}

fn read_hdf5_rust(path: &str) -> R<Kallisto> {
    use hdf5_rust::Hdf5File;
    let f = Hdf5File::open(path).map_err(e("open"))?;
    // read_f64 returns IEEE binary64 bit patterns
    let num = |name: &str| -> R<Vec<f64>> {
        let (_, bits) = f.read_f64(name).map_err(e(name))?;
        Ok(bits.into_iter().map(f64::from_bits).collect())
    };
    // no string reader: decode the raw fixed-length elements
    let (_, size, bytes) = f.read_raw("aux/ids").map_err(e("ids"))?;
    let ids = fixed_strings(&bytes, size);
    let counts = num("est_counts")?;
    let eff_lengths = num("aux/eff_lengths")?;
    let mut bootstraps = Vec::new();
    while let Ok((_, bits)) = f.read_f64(&format!("bootstrap/bs{}", bootstraps.len())) {
        bootstraps.push(bits.into_iter().map(f64::from_bits).collect());
    }
    Ok(Kallisto { ids, counts, eff_lengths, bootstraps })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: h5bench <hdf5-reader|hdf5-pure|rust-hdf5|hdf5-rust> <abundance.h5> <out.tsv> <repeats>");
        std::process::exit(2);
    }
    let (krate, path, out) = (a[1].as_str(), a[2].as_str(), a[3].as_str());
    let repeats: usize = a[4].parse().expect("repeats");
    let read: fn(&str) -> R<Kallisto> = match krate {
        "hdf5-reader" => read_hdf5_reader,
        "hdf5-pure" => read_hdf5_pure,
        "rust-hdf5" => read_rust_hdf5,
        "hdf5-rust" => read_hdf5_rust,
        _ => panic!("unknown crate {}", krate),
    };

    let k = match read(path) {
        Ok(k) => k,
        Err(msg) => {
            println!("{}\tFAILED\t{}", krate, msg.replace('\n', " "));
            std::process::exit(1);
        }
    };
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        let k2 = read(path).expect("second read");
        std::hint::black_box(&k2);
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
    std::fs::write(out, s).expect("write");
    println!(
        "{}\tok\ttranscripts={}\tbootstraps={}\tmedian_ms={:.2}\tmin_ms={:.2}",
        krate,
        k.ids.len(),
        k.bootstraps.len(),
        times[times.len() / 2],
        times[0]
    );
}

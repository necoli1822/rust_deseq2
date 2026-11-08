//! Ports of the R functions that design formulas evaluate in `model.frame()`:
//! `stats::poly()`, `contr.poly()`, `splines::ns()`, `splines::bs()`,
//! `splines::splineDesign()`, `scale()`, `quantile(type = 7)`, `mean()`, and the
//! LINPACK QR (`dqrdc2` / `dqrsl`) behind R's `qr()`, `qr.qy()` and `qr.qty()`
//!
//! Matrices are `ndarray::Array2` (samples x columns). R's long double
//! accumulations (`sum()`, `mean()`, `colMeans()`) are emulated with
//! double-double arithmetic, which agrees with R's 80-bit results except in rare
//! double-rounding cases.

use ndarray::Array2;

/// Error of an R function, with R's message
pub type RResult<T> = std::result::Result<T, String>;

/// Double-double accumulator (error-free transformations)
#[derive(Clone, Copy, Default)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    fn add(&mut self, v: f64) {
        let s = self.hi + v;
        let bb = s - self.hi;
        let err = (self.hi - (s - bb)) + (v - bb);
        self.hi = s;
        self.lo += err;
    }

    fn value(self) -> f64 {
        if !self.hi.is_finite() {
            return self.hi;
        }
        self.hi + self.lo
    }

    /// The accumulated value divided by `n`, rounded once
    fn div(self, n: f64) -> f64 {
        if !self.hi.is_finite() {
            return self.hi / n;
        }
        let s = self.hi + self.lo;
        let slo = self.lo - (s - self.hi);
        let q = s / n;
        let r = (-q).mul_add(n, s) + slo;
        q + r / n
    }
}

/// R's `sum()` of a double vector (long double accumulation)
pub fn r_sum(x: &[f64]) -> f64 {
    let mut acc = Dd::default();
    for &v in x {
        acc.add(v);
    }
    acc.value()
}

/// R's `mean()` of a double vector: the long double sum divided by n, then the
/// long double correction `s += sum(x - s) / n` (summary.c `rmean`)
pub fn r_mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut acc = Dd::default();
    for &v in x {
        acc.add(v);
    }
    let s = acc.div(n);
    if !s.is_finite() {
        return s;
    }
    let mut t = Dd::default();
    for &v in x {
        t.add(v - s);
        // the rounding error of v - s, as the long double difference keeps it
        let d = v - s;
        let bb = d - v;
        t.add((v - (d - bb)) - (s + bb));
    }
    s + t.div(n)
}

/// R's `colMeans()` of one column: the long double sum divided by n
pub fn r_col_mean(x: &[f64]) -> f64 {
    let mut acc = Dd::default();
    for &v in x {
        acc.add(v);
    }
    acc.div(x.len() as f64)
}

/// BLAS `dnrm2`, with the sum of squares accumulated in extended precision as
/// OpenBLAS's x86-64 kernel does
fn dnrm2(x: &[f64]) -> f64 {
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for &xi in x {
        let p = xi * xi;
        let pe = xi.mul_add(xi, -p);
        let s = hi + p;
        let bb = s - hi;
        let se = (hi - (s - bb)) + (p - bb);
        hi = s;
        lo += se + pe;
    }
    let ss = hi + lo;
    let lo2 = lo - (ss - hi);
    if ss <= 0.0 {
        return 0.0;
    }
    let r = ss.sqrt();
    let resid = (-r).mul_add(r, ss) + lo2;
    r + resid / (2.0 * r)
}

/// R's `qr(x)` (LINPACK `dqrdc2` with `tol = 1e-07`): Householder QR with R's
/// limited column pivoting, which moves columns with negligible norm to the end
pub struct Qr {
    /// `qr$qr`, column-major n x p
    qr: Vec<f64>,
    n: usize,
    p: usize,
    /// `qr$qraux`
    qraux: Vec<f64>,
    /// `qr$rank`
    pub rank: usize,
    /// `qr$pivot` (0-based)
    pub pivot: Vec<usize>,
}

impl Qr {
    /// R equivalent: qr(x) (LINPACK dqrdc2, tol = 1e-07)
    pub fn new(x: &Array2<f64>) -> Qr {
        const TOL: f64 = 1e-7;
        let (n, p) = x.dim();
        let mut a = vec![0.0; n * p];
        for j in 0..p {
            for i in 0..n {
                a[i + j * n] = x[[i, j]];
            }
        }
        let mut qraux = vec![0.0; p];
        let mut work1 = vec![0.0; p];
        let mut work2 = vec![0.0; p];
        let mut jpvt: Vec<usize> = (0..p).collect();
        for j in 0..p {
            qraux[j] = dnrm2(&a[j * n..(j + 1) * n]);
            work1[j] = qraux[j];
            work2[j] = if qraux[j] == 0.0 { 1.0 } else { qraux[j] };
        }
        let lup = n.min(p);
        // k is the 1-based Fortran k: p + 1, decremented for each column moved to the end
        let mut k = p + 1;
        for l in 0..lup {
            // cycle negligible columns to the end (Fortran: l .ge. k avoids infinite cycling)
            while l + 1 < k && qraux[l] < work2[l] * TOL {
                for i in 0..n {
                    let t = a[i + l * n];
                    for j in (l + 1)..p {
                        a[i + (j - 1) * n] = a[i + j * n];
                    }
                    a[i + (p - 1) * n] = t;
                }
                let (pi, t, tt, ttt) = (jpvt[l], qraux[l], work1[l], work2[l]);
                for j in (l + 1)..p {
                    jpvt[j - 1] = jpvt[j];
                    qraux[j - 1] = qraux[j];
                    work1[j - 1] = work1[j];
                    work2[j - 1] = work2[j];
                }
                jpvt[p - 1] = pi;
                qraux[p - 1] = t;
                work1[p - 1] = tt;
                work2[p - 1] = ttt;
                k -= 1;
            }
            if l + 1 == n {
                break;
            }
            let mut nrmxl = dnrm2(&a[l + l * n..(l + 1) * n]);
            if nrmxl == 0.0 {
                continue;
            }
            if a[l + l * n] != 0.0 {
                nrmxl = nrmxl.abs().copysign(a[l + l * n]);
            }
            let sc = 1.0 / nrmxl;
            for i in l..n {
                a[i + l * n] *= sc;
            }
            a[l + l * n] += 1.0;
            for j in (l + 1)..p {
                let mut dot = 0.0;
                for i in l..n {
                    dot += a[i + l * n] * a[i + j * n];
                }
                let t = -dot / a[l + l * n];
                for i in l..n {
                    a[i + j * n] += t * a[i + l * n];
                }
                if qraux[j] != 0.0 {
                    let tt = (1.0 - (a[l + j * n].abs() / qraux[j]).powi(2)).max(0.0);
                    if tt.abs() < 1e-6 {
                        qraux[j] = dnrm2(&a[l + 1 + j * n..(j + 1) * n]);
                        work1[j] = qraux[j];
                    } else {
                        qraux[j] *= tt.sqrt();
                    }
                }
            }
            qraux[l] = a[l + l * n];
            a[l + l * n] = -nrmxl;
        }
        let rank = (k - 1).min(n);
        Qr { qr: a, n, p, qraux, rank, pivot: jpvt }
    }

    /// LINPACK `dqrsl` applying Q (`transpose = false`, R: qr.qy) or Q^T
    /// (`transpose = true`, R: qr.qty) to `y` (length n), with k = rank
    fn apply(&self, y: &mut [f64], transpose: bool) {
        let n = self.n;
        let ju = self.rank.min(n.saturating_sub(1));
        let step = |j: usize, y: &mut [f64]| {
            if self.qraux[j] == 0.0 {
                return;
            }
            let diag = self.qraux[j];
            let col = |i: usize| if i == j { diag } else { self.qr[i + j * n] };
            let mut dot = 0.0;
            for i in j..n {
                dot += col(i) * y[i];
            }
            let t = -dot / diag;
            for i in j..n {
                y[i] += t * col(i);
            }
        };
        if transpose {
            for j in 0..ju {
                step(j, y);
            }
        } else {
            for j in (0..ju).rev() {
                step(j, y);
            }
        }
    }

    /// R's `qr.qy(qr, y)` for each column of `y` (n rows)
    pub fn qy(&self, y: &Array2<f64>) -> Array2<f64> {
        self.apply_columns(y, false)
    }

    /// R's `qr.qty(qr, y)` for each column of `y` (n rows)
    pub fn qty(&self, y: &Array2<f64>) -> Array2<f64> {
        self.apply_columns(y, true)
    }

    fn apply_columns(&self, y: &Array2<f64>, transpose: bool) -> Array2<f64> {
        let mut out = y.clone();
        for j in 0..y.ncols() {
            let mut col: Vec<f64> = y.column(j).to_vec();
            self.apply(&mut col, transpose);
            for (i, v) in col.into_iter().enumerate() {
                out[[i, j]] = v;
            }
        }
        out
    }

    /// The diagonal of `qr$qr` (R: diag(qr$qr), the diagonal of R)
    fn diag(&self) -> Vec<f64> {
        (0..self.n.min(self.p)).map(|i| self.qr[i + i * self.n]).collect()
    }
}

/// R's `x ^ y` for doubles (arithmetic.c `R_POW` / `R_pow`)
pub fn r_pow(x: f64, y: f64) -> f64 {
    if y == 2.0 {
        return x * x;
    }
    if x == 1.0 || y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y < 0.0 {
            f64::INFINITY
        } else {
            y
        };
    }
    if x.is_finite() && y.is_finite() {
        return x.powf(y);
    }
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if !x.is_finite() {
        if x > 0.0 {
            return if y < 0.0 { 0.0 } else { f64::INFINITY };
        }
        // (-Inf) ^ y
        if y.is_finite() && y == y.floor() {
            return if y < 0.0 {
                0.0
            } else if (y / 2.0).floor() * 2.0 != y {
                x
            } else {
                -x
            };
        }
    }
    if !y.is_finite() && x >= 0.0 {
        if y > 0.0 {
            return if x >= 1.0 { f64::INFINITY } else { 0.0 };
        }
        return if x < 1.0 { f64::INFINITY } else { 0.0 };
    }
    f64::NAN
}

/// The orthogonal polynomial basis of `poly(x, degree)` (coefs = NULL)
/// R equivalent: stats::poly(x, degree, raw = FALSE)
pub fn poly_orthogonal(x: &[f64], degree: usize) -> RResult<Array2<f64>> {
    if degree < 1 {
        return Err("'degree' must be at least 1".to_string());
    }
    if x.iter().any(|v| v.is_nan()) {
        return Err("missing values are not allowed in 'poly'".to_string());
    }
    let mut uniq: Vec<u64> = x.iter().map(|v| (v + 0.0).to_bits()).collect();
    uniq.sort_unstable();
    uniq.dedup();
    if degree >= uniq.len() {
        return Err("'degree' must be less than number of unique points".to_string());
    }
    let n = x.len();
    let xbar = r_mean(x);
    let xc: Vec<f64> = x.iter().map(|v| v - xbar).collect();
    let xm = Array2::from_shape_fn((n, degree + 1), |(i, j)| r_pow(xc[i], j as f64));
    let qr = Qr::new(&xm);
    if qr.rank < degree {
        return Err("'degree' must be less than number of unique points".to_string());
    }
    // z <- QR$qr * (row(z) == col(z)); Z <- qr.qy(QR, z)
    let d = qr.diag();
    let z = Array2::from_shape_fn((n, degree + 1), |(i, j)| if i == j { d[j] } else { 0.0 });
    let zq = qr.qy(&z);
    // norm2 <- colSums(Z^2); Z / rep(sqrt(norm2[-1]), each = n), first column dropped
    let mut out = Array2::zeros((n, degree));
    for j in 1..=degree {
        let sq: Vec<f64> = (0..n).map(|i| zq[[i, j]] * zq[[i, j]]).collect();
        let norm2 = r_sum(&sq);
        let s = norm2.sqrt();
        for i in 0..n {
            out[[i, j - 1]] = zq[[i, j]] / s;
        }
    }
    Ok(out)
}

/// `poly(x, degree, raw = TRUE)`: the powers `x^1, ..., x^degree`
/// R equivalent: outer(x, 1L:degree, "^")
pub fn poly_raw(x: &[f64], degree: usize) -> RResult<Array2<f64>> {
    if degree < 1 {
        return Err("'degree' must be at least 1".to_string());
    }
    Ok(Array2::from_shape_fn((x.len(), degree), |(i, j)| r_pow(x[i], (j + 1) as f64)))
}

/// The polynomial contrasts of an ordered factor with `n` levels (scores 1:n)
/// R equivalent: contr.poly(n), with column names .L, .Q, .C, ^4, ...
pub fn contr_poly(n: usize) -> RResult<(Array2<f64>, Vec<String>)> {
    if n < 2 {
        return Err(format!("contrasts not defined for {} degrees of freedom", n as i64 - 1));
    }
    if n > 95 {
        return Err(format!(
            "orthogonal polynomials cannot be represented accurately enough for {} degrees of freedom",
            n - 1
        ));
    }
    let scores: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mean = r_mean(&scores);
    let y: Vec<f64> = scores.iter().map(|s| s - mean).collect();
    let xm = Array2::from_shape_fn((n, n), |(i, j)| r_pow(y[i], j as f64));
    let qr = Qr::new(&xm);
    let d = qr.diag();
    let z = Array2::from_shape_fn((n, n), |(i, j)| if i == j { d[j] } else { 0.0 });
    let raw = qr.qy(&z);
    let mut out = Array2::zeros((n, n - 1));
    for j in 1..n {
        let sq: Vec<f64> = (0..n).map(|i| raw[[i, j]] * raw[[i, j]]).collect();
        let s = r_sum(&sq).sqrt();
        for i in 0..n {
            out[[i, j - 1]] = raw[[i, j]] / s;
        }
    }
    let names = (1..n)
        .map(|k| match k {
            1 => ".L".to_string(),
            2 => ".Q".to_string(),
            3 => ".C".to_string(),
            _ => format!("^{}", k),
        })
        .collect();
    Ok((out, names))
}

/// R's `seq.int(from, to, length.out = n)` (seq.c): `from + i * by`, with the last
/// value set to `to`
fn seq_length_out(from: f64, to: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    if n > 0 {
        out[0] = from;
    }
    if n > 1 {
        out[n - 1] = to;
    }
    if n > 2 {
        let by = (to - from) / (n - 1) as f64;
        for (i, o) in out.iter_mut().enumerate().take(n - 1).skip(1) {
            *o = from + i as f64 * by;
        }
    }
    out
}

/// R's `quantile(x, probs, names = FALSE)` with the default type 7
pub fn quantile7(x: &[f64], probs: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    probs
        .iter()
        .map(|&p| {
            let index = 1.0 + (n.max(1) - 1) as f64 * p;
            let lo = index.floor();
            let hi = index.ceil();
            let qs = v[lo as usize - 1];
            let xh = v[hi as usize - 1];
            if index > lo && xh != qs {
                let h = index - lo;
                (1.0 - h) * qs + h * xh
            } else {
                qs
            }
        })
        .collect()
}

/// The B-spline evaluation state of R's splines.c (`spl_struct`)
struct Spline<'a> {
    order: usize,
    knots: &'a [f64],
    curs: usize,
    boundary: bool,
    ldel: Vec<f64>,
    rdel: Vec<f64>,
    a: Vec<f64>,
}

impl Spline<'_> {
    /// splines.c `set_cursor`
    fn set_cursor(&mut self, x: f64) -> i64 {
        let nk = self.knots.len();
        let mut curs: i64 = -1;
        self.boundary = false;
        for (i, &k) in self.knots.iter().enumerate() {
            if k >= x {
                curs = i as i64;
            }
            if k > x {
                break;
            }
        }
        let last_legit = nk as i64 - self.order as i64;
        if curs > last_legit && x == self.knots[last_legit as usize] {
            self.boundary = true;
            curs = last_legit;
        }
        if curs >= 0 {
            self.curs = curs as usize;
        }
        curs
    }

    /// splines.c `diff_table`
    fn diff_table(&mut self, x: f64, ndiff: usize) {
        for i in 0..ndiff {
            self.rdel[i] = self.knots[self.curs + i] - x;
            self.ldel[i] = x - self.knots[self.curs - (i + 1)];
        }
    }

    /// splines.c `basis_funcs`: the non-zero B-splines at x
    fn basis_funcs(&mut self, x: f64, b: &mut [f64]) {
        let ordm1 = self.order - 1;
        self.diff_table(x, ordm1);
        b[0] = 1.0;
        for j in 1..=ordm1 {
            let mut saved = 0.0;
            for r in 0..j {
                let den = self.rdel[r] + self.ldel[j - 1 - r];
                if den != 0.0 {
                    let term = b[r] / den;
                    b[r] = saved + self.rdel[r] * term;
                    saved = self.ldel[j - 1 - r] * term;
                } else {
                    if r != 0 || self.rdel[r] != 0.0 {
                        b[r] = saved;
                    }
                    saved = 0.0;
                }
            }
            b[j] = saved;
        }
    }

    /// splines.c `evaluate`: derivative `nder` of the spline with coefficients `a`
    fn evaluate(&mut self, x: f64, nder: usize) -> f64 {
        let ordm1 = self.order - 1;
        let mut outer = ordm1;
        if self.boundary && nder == ordm1 {
            return 0.0;
        }
        let ti = self.curs;
        for _ in 0..nder {
            // lpt = ti - outer
            for k in 0..outer {
                let lpt = ti + k - outer;
                self.a[k] = outer as f64 * (self.a[k + 1] - self.a[k]) / (self.knots[lpt + outer] - self.knots[lpt]);
            }
            outer -= 1;
        }
        self.diff_table(x, outer);
        while outer > 0 {
            outer -= 1;
            // apt = a, lpt = ldel + outer (decreasing), rpt = rdel (increasing), outer + 1 steps
            for k in 0..=outer {
                let l = self.ldel[outer - k];
                let r = self.rdel[k];
                self.a[k] = (self.a[k + 1] * l + self.a[k] * r) / (r + l);
            }
        }
        self.a[0]
    }
}

/// The B-spline design matrix (rows: x, columns: the `length(knots) - ord` basis
/// functions), or its derivatives `derivs` (recycled over x)
/// R equivalent: splines::splineDesign(knots, x, ord, derivs, outer.ok = FALSE)
pub fn spline_design(knots: &[f64], x: &[f64], ord: usize, derivs: &[usize]) -> RResult<Array2<f64>> {
    let mut knots = knots.to_vec();
    let nk = knots.len();
    if nk == 0 {
        return Err("must have at least 'ord' knots".to_string());
    }
    knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if derivs.len() > x.len() {
        return Err("length of 'derivs' is larger than length of 'x'".to_string());
    }
    if derivs.is_empty() {
        return Err("empty 'derivs'".to_string());
    }
    if ord > nk || ord < 1 {
        return Err("'ord' must be positive integer, at most the number of knots".to_string());
    }
    if nk < 2 * ord - 1 {
        return Err(format!("need at least 2*ord -1 (={}) knots", 2 * ord - 1));
    }
    let degree = ord - 1;
    if x.iter().any(|&v| v < knots[ord - 1] || knots[nk - degree - 1] < v) {
        return Err(format!(
            "the 'x' data must be in the range {} to {} unless you set 'outer.ok = TRUE'",
            format_g(knots[ord - 1]),
            format_g(knots[nk - degree - 1])
        ));
    }
    let ncoef = nk - ord;
    let mut sp = Spline {
        order: ord,
        knots: &knots,
        curs: 0,
        boundary: false,
        ldel: vec![0.0; ord.max(2) - 1],
        rdel: vec![0.0; ord.max(2) - 1],
        a: vec![0.0; ord],
    };
    let mut design = Array2::zeros((x.len(), ncoef));
    let mut vals = vec![0.0; ord];
    for (i, &xi) in x.iter().enumerate() {
        let curs = sp.set_cursor(xi);
        let io = curs - ord as i64;
        if io < 0 || io > nk as i64 {
            return Err("subscript out of bounds".to_string());
        }
        let der = derivs[i % derivs.len()];
        if der > 0 {
            if der >= ord {
                return Err(format!("derivs = {} >= ord = {}, but should be in {{0,..,ord-1}}", der, ord));
            }
            for ii in 0..ord {
                sp.a.iter_mut().for_each(|v| *v = 0.0);
                sp.a[ii] = 1.0;
                vals[ii] = sp.evaluate(xi, der);
            }
        } else {
            sp.basis_funcs(xi, &mut vals);
        }
        for (j, &v) in vals.iter().enumerate() {
            let col = io as usize + j;
            if col >= ncoef {
                return Err("subscript out of bounds".to_string());
            }
            design[[i, col]] = v;
        }
    }
    Ok(design)
}

/// C's `%g` (6 significant digits), as R's gettextf("%g") prints numbers
fn format_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let e = x.abs().log10().floor() as i32;
    if !(-5..6).contains(&e) {
        let s = format!("{:.5e}", x);
        let (m, ex) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
        let exv: i32 = ex.parse().unwrap();
        format!("{}e{}{:02}", m, if exv < 0 { '-' } else { '+' }, exv.abs())
    } else {
        let s = format!("{:.*}", (5 - e).max(0) as usize, x);
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    }
}

/// Interior knots at quantiles of the points inside the boundary knots
/// R equivalent: the knot placement of splines::ns() / splines::bs():
/// `quantile(x[!outside], seq.int(0, 1, length.out = nIknots + 2L)[-c(1L, nIknots + 2L)])`,
/// where a fractional `nIknots` (from a fractional df) rounds `length.out` up and the
/// dropped index down, as R does
fn place_knots(x_inside: &[f64], n_interior: f64) -> Vec<f64> {
    if n_interior <= 0.0 {
        return Vec::new();
    }
    let probs = seq_length_out(0.0, 1.0, (n_interior + 2.0).ceil() as usize);
    let drop = (n_interior + 2.0).trunc() as usize - 1;
    let kept: Vec<f64> = probs.iter().enumerate().filter(|&(i, _)| i != 0 && i != drop).map(|(_, &p)| p).collect();
    quantile7(x_inside, &kept)
}

/// The arguments of ns() and bs()
#[derive(Debug, Clone, Default)]
pub struct SplineArgs {
    pub df: Option<f64>,
    pub knots: Option<Vec<f64>>,
    pub intercept: bool,
    pub boundary_knots: Option<[f64; 2]>,
}

/// Moves interior knots equal to a boundary knot inside, as ns() does (errors when
/// all match) or bs() does (`strict = false`: a warning when all match)
fn shove_knots(knots: &mut [f64], bk: [f64; 2], strict: bool) -> RResult<()> {
    let min = knots.iter().copied().fold(f64::INFINITY, f64::min);
    let max = knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lr_eq = [min == bk[0] || min == bk[1], max == bk[0] || max == bk[1]];
    if !(lr_eq[0] || lr_eq[1]) {
        return Ok(());
    }
    let mut all_eq = [false, false];
    if lr_eq[0] {
        let piv = bk[0];
        if knots.iter().all(|&k| k == piv) {
            if strict {
                return Err("all interior knots match left boundary knot".to_string());
            }
            all_eq[0] = true;
            log::warn!("all interior knots match left boundary knot");
        } else {
            let m = knots.iter().copied().filter(|&k| k > piv).fold(f64::INFINITY, f64::min);
            for k in knots.iter_mut().filter(|k| **k == piv) {
                *k += (m - piv) / 8.0;
            }
        }
    }
    if lr_eq[1] {
        let piv = bk[1];
        if knots.iter().all(|&k| k == piv) {
            if strict {
                return Err("all interior knots match right boundary knot".to_string());
            }
            all_eq[1] = true;
            log::warn!("all interior knots match right boundary knot");
        } else {
            let m = knots.iter().copied().filter(|&k| k < piv).fold(f64::NEG_INFINITY, f64::max);
            for k in knots.iter_mut().filter(|k| **k == piv) {
                *k -= (piv - m) / 8.0;
            }
        }
    }
    if strict || !(lr_eq[0] && all_eq[0] || lr_eq[1] && all_eq[1]) {
        log::warn!("shoving 'interior' knots matching boundary knots to inside");
    }
    Ok(())
}

/// The boundary knots (sorted, or `range(x)`) and the points outside them
fn boundary(x: &[f64], given: Option<[f64; 2]>) -> ([f64; 2], Vec<bool>, Vec<bool>) {
    match given {
        Some(b) => {
            let b = if b[0] <= b[1] { b } else { [b[1], b[0]] };
            let ol = x.iter().map(|&v| v < b[0]).collect();
            let or = x.iter().map(|&v| v > b[1]).collect();
            (b, ol, or)
        }
        None => {
            let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let b = if x.len() == 1 { [x[0] * 7.0 / 8.0, x[0] * 9.0 / 8.0] } else { [lo, hi] };
            (b, vec![false; x.len()], vec![false; x.len()])
        }
    }
}

/// `a %*% b`
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut s = 0.0;
            for k in 0..a.ncols() {
                s += a[[i, k]] * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

/// The natural cubic spline basis
/// R equivalent: splines::ns(x, df, knots, intercept, Boundary.knots)
pub fn ns(x: &[f64], args: &SplineArgs) -> RResult<Array2<f64>> {
    let (bk, ol, or) = boundary(x, args.boundary_knots);
    let outside: Vec<bool> = ol.iter().zip(&or).map(|(a, b)| *a || *b).collect();
    let mk_knots = args.df.is_some() && args.knots.is_none();
    let mut knots = if mk_knots {
        let df = args.df.unwrap();
        let mut n_int = df - 1.0 - if args.intercept { 1.0 } else { 0.0 };
        if n_int < 0.0 {
            n_int = 0.0;
            log::warn!("'df' was too small; have used {}", 1 + args.intercept as usize);
        }
        let inside: Vec<f64> = x.iter().zip(&outside).filter(|(_, o)| !**o).map(|(v, _)| *v).collect();
        place_knots(&inside, n_int)
    } else {
        let k = args.knots.clone().unwrap_or_default();
        if k.iter().any(|v| !v.is_finite()) {
            return Err("non-finite knots".to_string());
        }
        k
    };
    if mk_knots && !knots.is_empty() {
        shove_knots(&mut knots, bk, true)?;
    }
    let n_int = knots.len();
    let mut aknots: Vec<f64> = vec![bk[0]; 4].into_iter().chain(vec![bk[1]; 4]).chain(knots.iter().copied()).collect();
    aknots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut basis = if outside.iter().any(|&o| o) {
        let mut basis = Array2::zeros((x.len(), n_int + 4));
        for (side, mask, piv) in [(0, &ol, bk[0]), (1, &or, bk[1])] {
            let _ = side;
            let idx: Vec<usize> = (0..x.len()).filter(|&i| mask[i]).collect();
            if idx.is_empty() {
                continue;
            }
            let xl = Array2::from_shape_fn((idx.len(), 2), |(r, c)| if c == 0 { 1.0 } else { x[idx[r]] - piv });
            let tt = spline_design(&aknots, &[piv, piv], 4, &[0, 1])?;
            let rows = matmul(&xl, &tt);
            for (r, &i) in idx.iter().enumerate() {
                for c in 0..rows.ncols() {
                    basis[[i, c]] = rows[[r, c]];
                }
            }
        }
        let inside: Vec<usize> = (0..x.len()).filter(|&i| !outside[i]).collect();
        if !inside.is_empty() {
            let xs: Vec<f64> = inside.iter().map(|&i| x[i]).collect();
            let rows = spline_design(&aknots, &xs, 4, &[0])?;
            for (r, &i) in inside.iter().enumerate() {
                for c in 0..rows.ncols() {
                    basis[[i, c]] = rows[[r, c]];
                }
            }
        }
        basis
    } else {
        spline_design(&aknots, x, 4, &[0])?
    };
    let mut cons = spline_design(&aknots, &bk, 4, &[2, 2])?;
    if !args.intercept {
        basis = basis.slice(ndarray::s![.., 1..]).to_owned();
        cons = cons.slice(ndarray::s![.., 1..]).to_owned();
    }
    // qr.const <- qr(t(const)); basis <- t(qr.qty(qr.const, t(basis)))[, -(1:2)]
    let qr = Qr::new(&cons.t().to_owned());
    let qty = qr.qty(&basis.t().to_owned());
    Ok(qty.t().slice(ndarray::s![.., 2..]).to_owned())
}

/// The B-spline basis for a polynomial spline
/// R equivalent: splines::bs(x, df, knots, degree, intercept, Boundary.knots)
pub fn bs(x: &[f64], args: &SplineArgs, degree: f64) -> RResult<Array2<f64>> {
    // as.integer(degree) truncates
    let degree = degree.trunc();
    if !degree.is_finite() || degree < 1.0 {
        return Err("'degree' must be integer >= 1".to_string());
    }
    let degree = degree as usize;
    let ord = degree + 1;
    let (bk, ol, or) = boundary(x, args.boundary_knots);
    let outside: Vec<bool> = ol.iter().zip(&or).map(|(a, b)| *a || *b).collect();
    let mk_knots = args.df.is_some() && args.knots.is_none();
    let icpt = if args.intercept { 1.0 } else { 0.0 };
    let mut knots = if mk_knots {
        let df = args.df.unwrap();
        let mut n_int = df - ord as f64 + (1.0 - icpt);
        if n_int < 0.0 {
            n_int = 0.0;
            log::warn!("'df' was too small; have used {}", ord as f64 - (1.0 - icpt));
        }
        let inside: Vec<f64> = x.iter().zip(&outside).filter(|(_, o)| !**o).map(|(v, _)| *v).collect();
        place_knots(&inside, n_int)
    } else {
        let k = args.knots.clone().unwrap_or_default();
        if k.iter().any(|v| !v.is_finite()) {
            return Err("non-finite knots".to_string());
        }
        k
    };
    if mk_knots && !knots.is_empty() {
        shove_knots(&mut knots, bk, false)?;
    }
    let mut aknots: Vec<f64> =
        vec![bk[0]; ord].into_iter().chain(vec![bk[1]; ord]).chain(knots.iter().copied()).collect();
    aknots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let basis = if outside.iter().any(|&o| o) {
        log::warn!("some 'x' values beyond boundary knots may cause ill-conditioned bases");
        let derivs: Vec<usize> = (0..=degree).collect();
        // gamma(1:ord) = (k - 1)!
        let scalef: Vec<f64> = (1..=ord).map(|k| (1..k).product::<usize>() as f64).collect();
        let mut basis = Array2::zeros((x.len(), aknots.len() - degree - 1));
        let e = 0.25;
        let pivots = [
            (1.0 - e) * bk[0] + e * aknots[ord],
            (1.0 - e) * bk[1] + e * aknots[aknots.len() - ord - 1],
        ];
        for (mask, piv) in [(&ol, pivots[0]), (&or, pivots[1])] {
            let idx: Vec<usize> = (0..x.len()).filter(|&i| mask[i]).collect();
            if idx.is_empty() {
                continue;
            }
            let xl = Array2::from_shape_fn((idx.len(), ord), |(r, c)| {
                if c == 0 { 1.0 } else { r_pow(x[idx[r]] - piv, c as f64) }
            });
            let mut tt = spline_design(&aknots, &vec![piv; ord], ord, &derivs)?;
            for r in 0..tt.nrows() {
                for c in 0..tt.ncols() {
                    tt[[r, c]] /= scalef[r];
                }
            }
            let rows = matmul(&xl, &tt);
            for (r, &i) in idx.iter().enumerate() {
                for c in 0..rows.ncols() {
                    basis[[i, c]] = rows[[r, c]];
                }
            }
        }
        let inside: Vec<usize> = (0..x.len()).filter(|&i| !outside[i]).collect();
        if !inside.is_empty() {
            let xs: Vec<f64> = inside.iter().map(|&i| x[i]).collect();
            let rows = spline_design(&aknots, &xs, ord, &[0])?;
            for (r, &i) in inside.iter().enumerate() {
                for c in 0..rows.ncols() {
                    basis[[i, c]] = rows[[r, c]];
                }
            }
        }
        basis
    } else {
        spline_design(&aknots, x, ord, &[0])?
    };
    Ok(if args.intercept { basis } else { basis.slice(ndarray::s![.., 1..]).to_owned() })
}

/// The standardized values of `scale(x, center, scale)` for one column; `center` and
/// `scale` are `Some(value)` for numeric arguments and `None` for TRUE (computed),
/// and the flags say whether to center/scale at all
/// R equivalent: base::scale.default for a one-column matrix
pub fn scale_column(x: &[f64], center: Option<Option<f64>>, scale: Option<Option<f64>>) -> Vec<f64> {
    let mut v = x.to_vec();
    match center {
        Some(None) => {
            let m = r_col_mean(&v);
            v.iter_mut().for_each(|a| *a -= m);
        }
        Some(Some(c)) => v.iter_mut().for_each(|a| *a -= c),
        None => {}
    }
    match scale {
        Some(None) => {
            let sq: Vec<f64> = v.iter().map(|a| a * a).collect();
            let s = (r_sum(&sq) / ((v.len() as f64 - 1.0).max(1.0))).sqrt();
            v.iter_mut().for_each(|a| *a /= s);
        }
        Some(Some(s)) => v.iter_mut().for_each(|a| *a /= s),
        None => {}
    }
    v
}

/// R's formatting of a double with 15 significant digits, as `as.character()` and
/// `deparse()` print numbers: the fewest digits that give the same 15-digit value,
/// fixed notation unless scientific notation is narrower
/// R equivalent: formatReal(x, digits = 15) (format.c / scientific())
pub fn format_real15(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf".to_string() } else { "-Inf".to_string() };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.14e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let e10: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let nsig = digits.len().max(1) as i32;
    let negw = neg as i32;
    // fixed width
    let fixed_w = if e10 >= 0 {
        let rgt = (nsig - e10 - 1).max(0);
        negw + e10 + 1 + if rgt > 0 { rgt + 1 } else { 0 }
    } else {
        let rgt = nsig - e10 - 1;
        negw + 2 + rgt
    };
    let sci_w = negw + if nsig > 1 { nsig + 1 } else { 1 } + if e10.abs() >= 100 { 5 } else { 4 };
    let sign = if neg { "-" } else { "" };
    if fixed_w <= sci_w {
        let rgt = (nsig - e10 - 1).max(0) as usize;
        format!("{:.*}", rgt, x)
    } else {
        let m = if nsig > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.to_string() };
        format!("{}{}e{}{:02}", sign, m, if e10 < 0 { '-' } else { '+' }, e10.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 12] = [-1.104, -0.105, -1.362, 1.636, 0.075, -1.343, 0.27, 0.579, 0.379, -0.708, 1.533, 0.15];

    /// as.character() / deparse() of doubles in R 4.5.3
    #[test]
    fn test_format_real15() {
        let cases: [(f64, &str); 18] = [
            (0.1, "0.1"), (1.0 / 3.0, "0.333333333333333"), (1e5, "1e+05"), (123456.0, "123456"), (1e-4, "1e-04"),
            (1e15, "1e+15"), (0.1 + 0.2, "0.3"), (3e-5, "3e-05"), (2.5, "2.5"), (-0.5, "-0.5"), (-1.5e-7, "-1.5e-07"),
            (123456.7, "123456.7"), (1e100, "1e+100"), (-2e-100, "-2e-100"), (0.000123456, "0.000123456"),
            (1e5 + 0.1, "100000.1"), (100.0, "100"), (1234567.0, "1234567"),
        ];
        for (x, r) in cases {
            assert_eq!(format_real15(x), r, "{:e}", x);
        }
    }

    /// attr(ns(x, df), "knots"), attr(bs(x, df = 6), "knots"), quantile(), seq.int(),
    /// mean(), colMeans() and sum() in R
    #[test]
    fn test_knots_and_quantiles() {
        let close = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(p, q)| (p - q).abs() <= 1e-15);
        assert!(close(&place_knots(&X, 2.0), &[-0.30600000000000033, 0.30633333333333324]));
        assert!(close(
            &place_knots(&X, 4.0),
            &[-1.0247999999999999, -0.032999999999999932, 0.22200000000000017, 0.53900000000000015]
        ));
        // ns(x, df = 3.5): nIknots 2.5, length.out 5, element 4 dropped, so the last knot is max(x), which ns() then shoves inside
        assert!(close(&place_knots(&X, 2.5), &[-0.80699999999999994, 0.11249999999999999, 1.636]));
        assert!(close(&place_knots(&X, 3.0), &[-0.80699999999999994, 0.11249999999999999, 0.42899999999999999]));
        assert!(close(
            &quantile7(&X, &[0.1, 0.25, 0.5, 0.9]),
            &[-1.3190999999999999, -0.80699999999999994, 0.11249999999999999, 1.4376000000000002]
        ));
        assert_eq!(seq_length_out(0.0, 1.0, 4), vec![0.0, 0.33333333333333331, 0.66666666666666663, 1.0]);
        assert_eq!(seq_length_out(0.0, 1.0, 6)[3], 0.60000000000000009);
        assert_eq!(r_mean(&X), -2.7755575615628914e-17);
        assert_eq!(r_col_mean(&X), -2.7755575615628914e-17);
        assert_eq!(r_sum(&X), -3.3306690738754696e-16);
    }

    /// qr(cbind(1, x, 2 * x, x^2)): rank 3, the dependent column moved to the end
    #[test]
    fn test_qr_rank_and_pivot() {
        let m = Array2::from_shape_fn((12, 4), |(i, j)| match j {
            0 => 1.0,
            1 => X[i],
            2 => 2.0 * X[i],
            _ => X[i] * X[i],
        });
        let qr = Qr::new(&m);
        assert_eq!(qr.rank, 3);
        assert_eq!(qr.pivot, vec![0, 1, 3, 2]);
        // Q (Q^T y) = y
        let y = Array2::from_shape_fn((12, 1), |(i, _)| X[i] + 1.0);
        let back = qr.qy(&qr.qty(&y));
        assert!((0..12).all(|i| (back[[i, 0]] - y[[i, 0]]).abs() < 4e-15));
    }

    /// contr.poly(3) and the errors of contr.poly(), splineDesign() and poly() as R gives them
    #[test]
    #[allow(clippy::approx_constant)] // R's printed contr.poly(3), not math constants
    fn test_contr_poly_and_errors() {
        let (c, names) = contr_poly(3).unwrap();
        assert_eq!(names, vec![".L", ".Q"]);
        let r = [
            -0.70710678118654757, -9.0737996407892753e-17, 0.70710678118654746, 0.40824829046386307,
            -0.81649658092772603, 0.40824829046386307,
        ];
        assert!((0..6).all(|k| (c[[k % 3, k / 3]] - r[k]).abs() < 1e-16));
        assert_eq!(contr_poly(1).unwrap_err(), "contrasts not defined for 0 degrees of freedom");
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert!(spline_design(&knots, &[1.5], 4, &[0]).unwrap_err().contains("must be in the range 0 to 1"));
        assert_eq!(poly_orthogonal(&[1.0, 1.0, 2.0], 2).unwrap_err(), "'degree' must be less than number of unique points");
    }
}

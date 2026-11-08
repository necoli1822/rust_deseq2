//! Port of R's `loess()` + `predict.loess()` (stats package) for the case DESeq2 uses:
//! one predictor, `degree = 2`, `family = "gaussian"`, `surface = "interpolate"`,
//! `cell = 0.2`, unit weights.
//!
//! R fits the local regressions only at the vertices of a k-d tree built over `x`
//! (value and slope at each vertex) and evaluates the curve anywhere else by cubic
//! Hermite interpolation between the two vertices of the enclosing cell. This differs
//! from a direct local fit at every evaluation point, so it is ported here routine by
//! routine from R's `src/library/stats/src/loessf.f` / `loessc.c` (Cleveland, Grosse
//! and Shyu's dloess), together with the LINPACK `dqrdc`, `dqrsl` and `dsvdc` the local
//! fit relies on.
//!
//! All arrays below are 1-based (index 0 unused) so that the code reads like the
//! Fortran it ports.

/// R: `loess(cell = 0.2)` default.
const CELL: f64 = 0.2;

/// Fit `loess(y ~ x, span = span)` with R's defaults and return
/// `predict(fit, new_x)`.
///
/// R: `simpleLoess()` → C `loess_raw()` (`lowesd`, `lowesb`) for the fit, and
/// `predLoess()` → C `loess_ifit()` (`lowese`) for the prediction. As in `predLoess`,
/// points outside `range(x)` (and NaN points) get `NaN` (R's `NA`) because
/// `surface = "interpolate"` does not extrapolate.
pub(crate) fn loess_interpolate(x: &[f64], y: &[f64], span: f64, new_x: &[f64]) -> Vec<f64> {
    let n = x.len();
    assert_eq!(n, y.len(), "loess: x and y must have the same length");
    // R: loess_workspace(): nf = min(N, floor(N * span + 1e-5)); "span is too small" if nf <= 0
    let nf = n.min((n as f64 * span + 1e-5).floor() as usize);
    if n == 0 || nf == 0 {
        return vec![f64::NAN; new_x.len()];
    }

    let kd = KdFit::new(x, y, span, nf);

    // R: predLoess(): ranges <- apply(x, 2L, range); inside <- x.evaluate <= max & >= min
    let xmin = x.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    new_x
        .iter()
        .map(|&z| if z >= xmin && z <= xmax { kd.eval(z) } else { f64::NAN })
        .collect()
}

/// The k-d tree with vertex values and slopes that R keeps in `fit$kd`.
/// For d = 1 a cell has vc = 2 vertices.
struct KdFit {
    /// Vertex coordinates `v(1..nv)`.
    v: Vec<f64>,
    /// Vertex values and slopes `vval(0:1, 1..nv)`.
    vval: Vec<[f64; 2]>,
    /// Split dimension of each cell (`a(p)`, 0 for a leaf).
    a: Vec<usize>,
    /// Split value of each cell (`xi(p)`).
    xi: Vec<f64>,
    /// Lower/upper son of each split cell (`lo(p)`, `hi(p)`).
    lo: Vec<usize>,
    hi: Vec<usize>,
    /// The two vertices of each cell (`c(1:2, p)`).
    c: Vec<[usize; 2]>,
}

impl KdFit {
    /// R: `lowesd` (workspace setup) + `lowesb` → `ehg131` (k-d tree, then vertex fits).
    fn new(x: &[f64], y: &[f64], span: f64, nf: usize) -> Self {
        let n = x.len();
        // 1-based copies of the data.
        let mut xx = Vec::with_capacity(n + 1);
        xx.push(0.0);
        xx.extend_from_slice(x);
        let mut yy = Vec::with_capacity(n + 1);
        yy.push(0.0);
        yy.extend_from_slice(y);

        // R: loess_workspace(): nvmax = max(200, N); lowesd(): ncmax = nvmax
        let nvmax = n.max(200);
        let ncmax = nvmax;
        // R: simpleLoess passes span * cell, stored by loess_raw in v(2);
        // lowesb passes ifloor(n * v(2)) as fc and v(3) = 0 as fd.
        let fc = ifloor(n as f64 * (span * CELL));
        let fd0 = 0.0;

        let mut kd = KdFit {
            v: vec![0.0],
            vval: Vec::new(),
            a: vec![0; ncmax + 2],
            xi: vec![0.0; ncmax + 2],
            lo: vec![0; ncmax + 2],
            hi: vec![0; ncmax + 2],
            c: vec![[0, 0]; ncmax + 2],
        };

        // ehg131: build the k-d tree
        kd.ehg126(&xx);
        let vc = 2usize;
        let mut nv = vc;
        let mut nc = 1usize;
        kd.c[1] = [1, 2];
        // fd = fd * dnrm2(d, delta, 1), delta = v(vc) - v(1)
        let fd = fd0 * (kd.v[vc] - kd.v[1]).abs();
        let mut pi: Vec<usize> = (0..=n).collect();
        kd.ehg124(1, n, &mut nv, &mut nc, ncmax, nvmax, &xx, &mut pi, fc, fd);

        // ehg139: local fits at the vertices. psi is initialised once and its
        // partial ordering carries over from one vertex to the next.
        let mut psi: Vec<usize> = (0..=n).collect();
        let mut ws = LocalFitWork::new(n, nf);
        let mut vval = vec![[0.0; 2]; nv + 1];
        for l in 1..=nv {
            vval[l] = ehg127(kd.v[l], &xx, &yy, nf, span, &mut psi, &mut ws);
        }
        kd.vval = vval;
        kd
    }

    /// R: `ehg126` — bounding box of x, expanded by 0.5% (or a tiny absolute margin).
    fn ehg126(&mut self, x: &[f64]) {
        let machin = f64::MAX;
        let mut alpha = machin;
        let mut beta = -machin;
        for &t in &x[1..] {
            alpha = alpha.min(t);
            beta = beta.max(t);
        }
        let mu = 0.005 * (beta - alpha).max(1.0e-10 * alpha.abs().max(beta.abs()) + 1.0e-30);
        alpha -= mu;
        beta += mu;
        // v(1) = lower left, v(vc) = upper right; d = 1 has no other vertices.
        self.v.push(alpha);
        self.v.push(beta);
    }

    /// R: `ehg124` — recursive (here iterative, breadth-first as in the Fortran)
    /// k-d tree build. A cell is a leaf when it holds at most `fc` points, its diameter
    /// is at most `fd`, the tree would outgrow `ncmax`/`nvmax`, or the median split
    /// value coincides with a cell boundary.
    #[allow(clippy::too_many_arguments)]
    fn ehg124(
        &mut self,
        ll: usize,
        uu: usize,
        nv: &mut usize,
        nc: &mut usize,
        ncmax: usize,
        nvmax: usize,
        x: &[f64],
        pi: &mut [usize],
        fc: i64,
        fd: f64,
    ) {
        let vc = 2usize;
        let mut p = 1usize;
        let mut l = ll;
        let mut u = uu;
        self.lo[p] = l;
        self.hi[p] = u;
        while p <= *nc {
            let diag = self.v[self.c[p][1]] - self.v[self.c[p][0]];
            let diam = (diag * diag).sqrt();
            let mut leaf = if ((u - l) + 1) as i64 <= fc {
                true
            } else {
                diam <= fd
            };
            if !leaf {
                leaf = if ncmax < *nc + 2 {
                    true
                } else {
                    (nvmax as f64) < *nv as f64 + vc as f64 / 2.0
                };
            }
            let mut m = 0usize;
            if !leaf {
                // ehg129 + idamax: with d = 1 the split dimension k is always 1.
                m = ((l + u) as f64 / 2.0) as usize;
                ehg106(l, u, m, x, pi);

                // all ties go with hi son (bug fix from btyner@gmail.com 2006-07-20)
                let mut offset: i64 = 0;
                loop {
                    let mo = m as i64 + offset;
                    if mo >= u as i64 || mo < l as i64 {
                        break;
                    }
                    let (lower, check, upper) = if offset < 0 {
                        (l, mo as usize, mo as usize)
                    } else {
                        let lw = (mo + 1) as usize;
                        (lw, lw, u)
                    };
                    ehg106(lower, upper, check, x, pi);
                    if x[pi[mo as usize]] == x[pi[mo as usize + 1]] {
                        offset = -offset;
                        if offset >= 0 {
                            offset += 1;
                        }
                    } else {
                        m = mo as usize;
                        break;
                    }
                }

                leaf = self.v[self.c[p][0]] == x[pi[m]] || self.v[self.c[p][1]] == x[pi[m]];
            }
            if leaf {
                self.a[p] = 0;
            } else {
                self.a[p] = 1;
                self.xi[p] = x[pi[m]];
                // left son
                *nc += 1;
                self.lo[p] = *nc;
                self.lo[*nc] = l;
                self.hi[*nc] = m;
                // right son
                *nc += 1;
                self.hi[p] = *nc;
                self.lo[*nc] = m + 1;
                self.hi[*nc] = u;
                self.ehg125(p, nv);
            }
            p += 1;
            if p <= *nc {
                l = self.lo[p];
                u = self.hi[p];
            }
        }
    }

    /// R: `ehg125` — add the vertex created by splitting cell `p` at `xi(p)`
    /// (reusing an existing vertex with the same coordinate) and set the vertices
    /// of the two sons.
    fn ehg125(&mut self, p: usize, nv: &mut usize) {
        let t = self.xi[p];
        let m = match (1..=*nv).find(|&m| self.v[m] == t) {
            Some(m) => m,
            None => {
                self.v.push(t);
                *nv += 1;
                *nv
            }
        };
        let [f0, f1] = self.c[p];
        self.c[self.lo[p]] = [f0, m];
        self.c[self.hi[p]] = [m, f1];
    }

    /// R: `ehg128` (via `lowese` → `ehg133`) for d = 1 — descend to the leaf cell
    /// containing `z` and blend the vertex values and slopes with cubic Hermite
    /// polynomials.
    fn eval(&self, z: f64) -> f64 {
        let mut j = 1usize;
        while self.a[j] != 0 {
            j = if z <= self.xi[j] { self.lo[j] } else { self.hi[j] };
        }
        let [ll, ur] = self.c[j];
        let g1 = self.vval[ll];
        let g2 = self.vval[ur];
        let width = self.v[ur] - self.v[ll];
        let h = (z - self.v[ll]) / width;
        // Hermite basis
        let phi0 = ((1.0 - h) * (1.0 - h)) * (1.0 + 2.0 * h);
        let phi1 = (h * h) * (3.0 - 2.0 * h);
        let psi0 = h * ((1.0 - h) * (1.0 - h));
        let psi1 = (h * h) * (h - 1.0);
        phi0 * g1[0] + phi1 * g2[0] + (psi0 * g1[1] + psi1 * g2[1]) * width
    }
}

/// R: Fortran `ifloor`.
fn ifloor(x: f64) -> i64 {
    let i = x as i64;
    if i as f64 > x {
        i - 1
    } else {
        i
    }
}

/// R: `ehg106` — Floyd & Rivest (CACM Mar '75, Algorithm 489) selection: permute
/// `pi[il..=ir]` so that `p[pi[k]]` is the k-th smallest of `p[pi[il..=ir]]`.
fn ehg106(il: usize, ir: usize, k: usize, p: &[f64], pi: &mut [usize]) {
    let k = k as isize;
    let mut l = il as isize;
    let mut r = ir as isize;
    let at = |pi: &[usize], i: isize| p[pi[i as usize]];
    while l < r {
        let t = at(pi, k);
        let mut i = l;
        let mut j = r;
        pi.swap(l as usize, k as usize);
        if t < at(pi, r) {
            pi.swap(l as usize, r as usize);
        }
        while i < j {
            pi.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
            while at(pi, i) < t {
                i += 1;
            }
            while t < at(pi, j) {
                j -= 1;
            }
        }
        if at(pi, l) == t {
            pi.swap(l as usize, j as usize);
        } else {
            j += 1;
            pi.swap(r as usize, j as usize);
        }
        if j <= k {
            l = j + 1;
        }
        if k <= j {
            r = j - 1;
        }
    }
}

/// Number of local-model columns for d = 1, degree 2 (`iv(29)`).
const K: usize = 3;

/// Scratch arrays of `ehg127`.
struct LocalFitWork {
    dist: Vec<f64>,
    w: Vec<f64>,
    eta: Vec<f64>,
    /// Design matrix `b(nf, k)`, column-major, 1-based: `b[i + j * (nf + 1)]`.
    b: Vec<f64>,
}

impl LocalFitWork {
    fn new(n: usize, nf: usize) -> Self {
        LocalFitWork {
            dist: vec![0.0; n + 1],
            w: vec![0.0; nf + 1],
            eta: vec![0.0; nf + 1],
            b: vec![0.0; (nf + 1) * (K + 1)],
        }
    }
}

/// R: `ehg127` for d = 1, degree 2, kernel 1 (tricube), unit robustness weights —
/// weighted local quadratic fit at `q` over its `nf` nearest neighbours.
/// Returns `[value, slope]` at `q` (`s(0:od)` with od = d = 1).
///
/// The least-squares problem is solved as in the Fortran: equilibrate columns,
/// QR (`dqrdc`, no pivoting), `Q^T eta` (`dqrsl`), SVD of R (`dsvdc`), then the
/// pseudo-inverse with singular values below `sigma(1) * 100 * DBL_EPSILON` dropped.
fn ehg127(
    q: f64,
    x: &[f64],
    y: &[f64],
    nf: usize,
    f: f64,
    psi: &mut [usize],
    ws: &mut LocalFitWork,
) -> [f64; 2] {
    let n = x.len() - 1;
    let machep = f64::EPSILON; // d1mach(4)

    // sort by distance
    for i3 in 1..=n {
        ws.dist[i3] = 0.0;
    }
    for i3 in 1..=n {
        let dx = x[i3] - q;
        ws.dist[i3] += dx * dx;
    }
    ehg106(1, n, nf, &ws.dist, psi);
    let rho = ws.dist[psi[nf]] * f.max(1.0);
    // (rho <= 0 only triggers loesswarn(120) in R.)

    // compute neighbourhood weights (tricube); x**3 is evaluated as (x*x)*x
    for i3 in 1..=nf {
        ws.w[i3] = (ws.dist[psi[i3]] / rho).sqrt();
    }
    for i3 in 1..=nf {
        let wi = ws.w[i3];
        let t = 1.0 - (wi * wi) * wi;
        // rw(psi(i3)) = 1
        ws.w[i3] = (1.0 * ((t * t) * t)).sqrt();
    }

    // fill design matrix
    let ld = nf + 1;
    for i3 in 1..=nf {
        let wi = ws.w[i3];
        let dx = x[psi[i3]] - q;
        ws.b[i3 + ld] = wi;
        ws.b[i3 + 2 * ld] = wi * dx;
        ws.b[i3 + 3 * ld] = wi * (dx * dx);
        ws.eta[i3] = wi * y[psi[i3]];
    }

    // equilibrate columns
    let mut colnor = [0.0f64; K + 1];
    for j in 1..=K {
        let mut scal = 0.0;
        for i in 1..=nf {
            let bij = ws.b[i + j * ld];
            scal += bij * bij;
        }
        scal = scal.sqrt();
        if 0.0 < scal {
            for i in 1..=nf {
                ws.b[i + j * ld] /= scal;
            }
            colnor[j] = scal;
        } else {
            colnor[j] = 1.0;
        }
    }

    // QR, then Q^T eta
    let mut qraux = [0.0f64; K + 1];
    dqrdc(&mut ws.b, ld, nf, K, &mut qraux);
    dqrsl_qty(&mut ws.b, ld, nf, K, &qraux, &mut ws.eta);

    // singular value decomposition of the K x K triangular factor
    let ldu = K + 1;
    let mut u = vec![0.0f64; (K + 1) * (K + 1)];
    for i in 1..=K {
        for j in i..=K {
            u[i + j * ldu] = ws.b[i + j * ld];
        }
    }
    let mut sigma = [0.0f64; K + 2];
    let mut uu = vec![0.0f64; (K + 1) * (K + 1)];
    let mut e = vec![0.0f64; (K + 1) * (K + 1)];
    // (info != 0 only triggers loesswarn(182) in R.)
    let _info = dsvdc(&mut u, K, K, &mut sigma, &mut uu, &mut e);
    let tol = sigma[1] * (100.0 * machep);

    // compensate for equilibration
    for j in 1..=K {
        let i10 = colnor[j];
        for i3 in 1..=K {
            e[j + i3 * ldu] /= i10;
        }
    }

    // solve least squares problem
    let mut dgamma = [0.0f64; K + 1];
    for j in 1..=K {
        dgamma[j] = if tol < sigma[j] {
            let mut dot = 0.0;
            for i in 1..=K {
                dot += uu[i + j * ldu] * ws.eta[i];
            }
            dot / sigma[j]
        } else {
            0.0
        };
    }
    let mut s = [0.0f64; 2];
    for (j, sj) in s.iter_mut().enumerate() {
        let mut dot = 0.0;
        for i in 1..=K {
            dot += e[(j + 1) + i * ldu] * dgamma[i];
        }
        *sj = dot;
    }
    s
}

/// BLAS `dnrm2` of `x[off], x[off+1], …` (n values), computed with an extended
/// (double-double) accumulator and rounded once, like OpenBLAS' x86-64 kernel,
/// which accumulates in x87 extended precision.
fn dnrm2(n: usize, x: &[f64], off: usize) -> f64 {
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for &xi in &x[off..off + n] {
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
    // one Newton correction on the double-double radicand
    let resid = (-r).mul_add(r, ss) + lo2;
    r + resid / (2.0 * r)
}

/// Fortran `dsign(a, b)`.
fn dsign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// LINPACK `dqrdc` with `job = 0` (no column pivoting): Householder QR of the
/// n x p matrix `x` (column-major, 1-based, leading dimension `ld`).
fn dqrdc(x: &mut [f64], ld: usize, n: usize, p: usize, qraux: &mut [f64]) {
    let ix = |i: usize, j: usize| i + j * ld;
    let lup = n.min(p);
    for l in 1..=lup {
        qraux[l] = 0.0;
        if l == n {
            continue;
        }
        let mut nrmxl = dnrm2(n - l + 1, x, ix(l, l));
        if nrmxl == 0.0 {
            continue;
        }
        if x[ix(l, l)] != 0.0 {
            nrmxl = dsign(nrmxl, x[ix(l, l)]);
        }
        let sc = 1.0 / nrmxl;
        for i in l..=n {
            x[ix(i, l)] *= sc;
        }
        x[ix(l, l)] += 1.0;
        for j in (l + 1)..=p {
            let mut dot = 0.0;
            for i in l..=n {
                dot += x[ix(i, l)] * x[ix(i, j)];
            }
            let t = -dot / x[ix(l, l)];
            for i in l..=n {
                x[ix(i, j)] += t * x[ix(i, l)];
            }
        }
        qraux[l] = x[ix(l, l)];
        x[ix(l, l)] = -nrmxl;
    }
}

/// LINPACK `dqrsl` with `job = 1000`: overwrite `y` with `Q^T y`.
fn dqrsl_qty(x: &mut [f64], ld: usize, n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    let ix = |i: usize, j: usize| i + j * ld;
    let ju = k.min(n - 1);
    for j in 1..=ju {
        if qraux[j] == 0.0 {
            continue;
        }
        let temp = x[ix(j, j)];
        x[ix(j, j)] = qraux[j];
        let mut dot = 0.0;
        for i in j..=n {
            dot += x[ix(i, j)] * y[i];
        }
        let t = -dot / x[ix(j, j)];
        for i in j..=n {
            y[i] += t * x[ix(i, j)];
        }
        x[ix(j, j)] = temp;
    }
}

/// BLAS `drotg`: construct a Givens rotation. `r` is computed as
/// `sign(roe) * sqrt(a^2 + b^2)` with a single rounding (OpenBLAS evaluates it in
/// extended precision on x86-64).
fn drotg(da: &mut f64, db: &mut f64) -> (f64, f64) {
    let (a, b) = (*da, *db);
    let roe = if a.abs() > b.abs() { a } else { b };
    let scale = a.abs() + b.abs();
    let (c, s, r, z);
    if scale == 0.0 {
        c = 1.0;
        s = 0.0;
        r = 0.0;
        z = 0.0;
    } else {
        r = dsign(1.0, roe) * a.hypot(b);
        c = a / r;
        s = b / r;
        let mut zz = 1.0;
        if a.abs() > b.abs() {
            zz = s;
        }
        if b.abs() >= a.abs() && c != 0.0 {
            zz = 1.0 / c;
        }
        z = zz;
    }
    *da = r;
    *db = z;
    (c, s)
}

/// BLAS `drot` on columns `cx` and `cy` (rows 1..=n) of the 1-based matrix `m`.
fn drot(m: &mut [f64], ld: usize, n: usize, cx: usize, cy: usize, c: f64, s: f64) {
    for i in 1..=n {
        let xi = m[i + cx * ld];
        let yi = m[i + cy * ld];
        m[i + cx * ld] = c * xi + s * yi;
        m[i + cy * ld] = c * yi - s * xi;
    }
}

/// LINPACK `dsvdc` (R's copy in src/appl, with its relative convergence tests) with
/// `job = 21`: singular values `s`, left vectors `u` (n x min(n,p)) and right vectors
/// `v` (p x p) of the n x p matrix `x`. All matrices are column-major and 1-based
/// with leading dimension `n + 1` (`x`, `u`) or `p + 1` (`v`). Returns `info`.
fn dsvdc(x: &mut [f64], n: usize, p: usize, s: &mut [f64], u: &mut [f64], v: &mut [f64]) -> usize {
    let ldx = n + 1;
    let ldu = n + 1;
    let ldv = p + 1;
    let xi = |i: usize, j: usize| i + j * ldx;
    let ui = |i: usize, j: usize| i + j * ldu;
    let vi = |i: usize, j: usize| i + j * ldv;
    let mut e = vec![0.0f64; p + 2];
    let mut work = vec![0.0f64; n + 2];
    let maxit = 30;
    let wantu = true;
    let wantv = true;
    let ncu = n.min(p);
    let mut info = 0usize;

    // reduce x to bidiagonal form
    let nct = (n - 1).min(p);
    let nrt = (p as isize - 2).min(n as isize).max(0) as usize;
    let lu = nct.max(nrt);
    for l in 1..=lu {
        let lp1 = l + 1;
        if l <= nct {
            s[l] = dnrm2(n - l + 1, x, xi(l, l));
            if s[l] != 0.0 {
                if x[xi(l, l)] != 0.0 {
                    s[l] = dsign(s[l], x[xi(l, l)]);
                }
                let sc = 1.0 / s[l];
                for i in l..=n {
                    x[xi(i, l)] *= sc;
                }
                x[xi(l, l)] += 1.0;
            }
            s[l] = -s[l];
        }
        for j in lp1..=p {
            if l <= nct && s[l] != 0.0 {
                let mut dot = 0.0;
                for i in l..=n {
                    dot += x[xi(i, l)] * x[xi(i, j)];
                }
                let t = -dot / x[xi(l, l)];
                for i in l..=n {
                    x[xi(i, j)] += t * x[xi(i, l)];
                }
            }
            e[j] = x[xi(l, j)];
        }
        if wantu && l <= nct {
            for i in l..=n {
                u[ui(i, l)] = x[xi(i, l)];
            }
        }
        if l <= nrt {
            e[l] = dnrm2(p - l, &e, lp1);
            if e[l] != 0.0 {
                if e[lp1] != 0.0 {
                    e[l] = dsign(e[l], e[lp1]);
                }
                let sc = 1.0 / e[l];
                for i in lp1..=p {
                    e[i] *= sc;
                }
                e[lp1] += 1.0;
            }
            e[l] = -e[l];
            if lp1 <= n && e[l] != 0.0 {
                for w in work.iter_mut().take(n + 1).skip(lp1) {
                    *w = 0.0;
                }
                for j in lp1..=p {
                    let ej = e[j];
                    for i in lp1..=n {
                        work[i] += ej * x[xi(i, j)];
                    }
                }
                for j in lp1..=p {
                    let t = -e[j] / e[lp1];
                    for i in lp1..=n {
                        x[xi(i, j)] += t * work[i];
                    }
                }
            }
            if wantv {
                for i in lp1..=p {
                    v[vi(i, l)] = e[i];
                }
            }
        }
    }

    // set up the final bidiagonal matrix of order m
    let mut m = p.min(n + 1);
    let nctp1 = nct + 1;
    let nrtp1 = nrt + 1;
    if nct < p {
        s[nctp1] = x[xi(nctp1, nctp1)];
    }
    if n < m {
        s[m] = 0.0;
    }
    if nrtp1 < m {
        e[nrtp1] = x[xi(nrtp1, m)];
    }
    e[m] = 0.0;

    // generate u
    if wantu {
        for j in nctp1..=ncu {
            for i in 1..=n {
                u[ui(i, j)] = 0.0;
            }
            u[ui(j, j)] = 1.0;
        }
        for ll in 1..=nct {
            let l = nct - ll + 1;
            if s[l] != 0.0 {
                let lp1 = l + 1;
                for j in lp1..=ncu {
                    let mut dot = 0.0;
                    for i in l..=n {
                        dot += u[ui(i, l)] * u[ui(i, j)];
                    }
                    let t = -dot / u[ui(l, l)];
                    for i in l..=n {
                        u[ui(i, j)] += t * u[ui(i, l)];
                    }
                }
                for i in l..=n {
                    u[ui(i, l)] *= -1.0;
                }
                u[ui(l, l)] += 1.0;
                for i in 1..l {
                    u[ui(i, l)] = 0.0;
                }
            } else {
                for i in 1..=n {
                    u[ui(i, l)] = 0.0;
                }
                u[ui(l, l)] = 1.0;
            }
        }
    }

    // generate v
    if wantv {
        for ll in 1..=p {
            let l = p - ll + 1;
            let lp1 = l + 1;
            if l <= nrt && e[l] != 0.0 {
                for j in lp1..=p {
                    let mut dot = 0.0;
                    for i in lp1..=p {
                        dot += v[vi(i, l)] * v[vi(i, j)];
                    }
                    let t = -dot / v[vi(lp1, l)];
                    for i in lp1..=p {
                        v[vi(i, j)] += t * v[vi(i, l)];
                    }
                }
            }
            for i in 1..=p {
                v[vi(i, l)] = 0.0;
            }
            v[vi(l, l)] = 1.0;
        }
    }

    // main iteration loop for the singular values
    let mm = m;
    let mut iter = 0;
    while m != 0 {
        if iter >= maxit {
            info = m;
            break;
        }
        // inspect the bidiagonal for negligible elements
        let mut l = 0usize;
        for ll in 1..=m {
            l = m - ll;
            if l == 0 {
                break;
            }
            let test = s[l].abs() + s[l + 1].abs();
            let ztest = test + e[l].abs();
            let acc = (test - ztest).abs() / (1.0e-100 + test);
            if acc <= 1.0e-15 {
                e[l] = 0.0;
                break;
            }
        }
        let kase;
        if l == m - 1 {
            kase = 4;
        } else {
            let lp1 = l + 1;
            let mp1 = m + 1;
            let mut ls = l;
            for lls in lp1..=mp1 {
                ls = m + lp1 - lls;
                if ls == l {
                    break;
                }
                let mut test = 0.0;
                if ls != m {
                    test += e[ls].abs();
                }
                if ls != l + 1 {
                    test += e[ls - 1].abs();
                }
                let ztest = test + s[ls].abs();
                let acc = (test - ztest).abs() / (1.0e-100 + test);
                if acc <= 1.0e-15 {
                    s[ls] = 0.0;
                    break;
                }
            }
            if ls == l {
                kase = 3;
            } else if ls == m {
                kase = 1;
            } else {
                kase = 2;
                l = ls;
            }
        }
        l += 1;

        match kase {
            1 => {
                // deflate negligible s(m)
                let mm1 = m - 1;
                let mut f = e[m - 1];
                e[m - 1] = 0.0;
                for kk in l..=mm1 {
                    let k = mm1 - kk + l;
                    let mut t1 = s[k];
                    let (cs, sn) = drotg(&mut t1, &mut f);
                    s[k] = t1;
                    if k != l {
                        f = -sn * e[k - 1];
                        e[k - 1] *= cs;
                    }
                    if wantv {
                        drot(v, ldv, p, k, m, cs, sn);
                    }
                }
            }
            2 => {
                // split at negligible s(l)
                let mut f = e[l - 1];
                e[l - 1] = 0.0;
                for k in l..=m {
                    let mut t1 = s[k];
                    let (cs, sn) = drotg(&mut t1, &mut f);
                    s[k] = t1;
                    f = -sn * e[k];
                    e[k] *= cs;
                    if wantu {
                        drot(u, ldu, n, k, l - 1, cs, sn);
                    }
                }
            }
            3 => {
                // perform one qr step: calculate the shift
                let scale = s[m]
                    .abs()
                    .max(s[m - 1].abs())
                    .max(e[m - 1].abs())
                    .max(s[l].abs())
                    .max(e[l].abs());
                let sm = s[m] / scale;
                let smm1 = s[m - 1] / scale;
                let emm1 = e[m - 1] / scale;
                let sl = s[l] / scale;
                let el = e[l] / scale;
                let b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0;
                let c = (sm * emm1) * (sm * emm1);
                let mut shift = 0.0;
                if !(b == 0.0 && c == 0.0) {
                    shift = (b * b + c).sqrt();
                    if b < 0.0 {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }
                let mut f = (sl + sm) * (sl - sm) + shift;
                let mut g = sl * el;
                // chase zeros
                let mm1 = m - 1;
                for k in l..=mm1 {
                    let (cs, sn) = drotg(&mut f, &mut g);
                    if k != l {
                        e[k - 1] = f;
                    }
                    f = cs * s[k] + sn * e[k];
                    e[k] = cs * e[k] - sn * s[k];
                    g = sn * s[k + 1];
                    s[k + 1] *= cs;
                    if wantv {
                        drot(v, ldv, p, k, k + 1, cs, sn);
                    }
                    let (cs, sn) = drotg(&mut f, &mut g);
                    s[k] = f;
                    f = cs * e[k] + sn * s[k + 1];
                    s[k + 1] = -sn * e[k] + cs * s[k + 1];
                    g = sn * e[k + 1];
                    e[k + 1] *= cs;
                    if wantu && k < n {
                        drot(u, ldu, n, k, k + 1, cs, sn);
                    }
                }
                e[m - 1] = f;
                iter += 1;
            }
            _ => {
                // convergence: make the singular value positive
                if s[l] < 0.0 {
                    s[l] = -s[l];
                    if wantv {
                        for i in 1..=p {
                            v[vi(i, l)] *= -1.0;
                        }
                    }
                }
                // order the singular values
                while l != mm && s[l] < s[l + 1] {
                    s.swap(l, l + 1);
                    if wantv && l < p {
                        for i in 1..=p {
                            v.swap(vi(i, l), vi(i, l + 1));
                        }
                    }
                    if wantu && l < n {
                        for i in 1..=n {
                            u.swap(ui(i, l), ui(i, l + 1));
                        }
                    }
                    l += 1;
                }
                iter = 0;
                m -= 1;
            }
        }
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R's `seq(from, to, length.out = n)` (n > 2).
    fn r_seq(from: f64, to: f64, n: usize) -> Vec<f64> {
        let by = (to - from) / (n - 1) as f64;
        let mut v: Vec<f64> = (0..n).map(|i| from + i as f64 * by).collect();
        v[n - 1] = to;
        v
    }

    /// Reference values from R 4.5.3 (stats::loess, OpenBLAS):
    /// ```r
    /// x <- seq(0, 8, length = 200)
    /// y <- sin(1.3*x) + 0.1*cos(7*x) + x^2/20
    /// p <- predict(loess(y ~ x, span = .2), seq(0, 8, length.out = 1000))
    /// sprintf("%.17g", p[c(1, 2, 100, 250, 411, 412, 413, 500, 750, 999, 1000)]); which.min(p)  # 412
    /// ```
    #[test]
    fn test_loess_interpolate_matches_r() {
        let x = r_seq(0.0, 8.0, 200);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (1.3 * xi).sin() + 0.1 * (7.0 * xi).cos() + xi * xi / 20.0)
            .collect();
        let fine = r_seq(0.0, 8.0, 1000);
        let p = loess_interpolate(&x, &y, 0.2, &fine);

        let expected: [(usize, f64); 11] = [
            (0, 0.049211061489999722),
            (1, 0.058668475741219694),
            (99, 0.9079319977016852),
            (249, 0.73380593923255755),
            (410, -0.38976701379405831),
            (411, -0.38976857505013901),
            (412, -0.38944650108456852),
            (499, -0.12785104255704885),
            (749, 2.7757135872125951),
            (998, 2.3531962747589956),
            (999, 2.3524235193207317),
        ];
        for &(i, want) in &expected {
            let rel = (p[i] - want).abs() / want.abs();
            assert!(rel < 1e-12, "fine[{i}]: got {:.17e}, R {:.17e} (rel {rel:e})", p[i], want);
        }

        let argmin = p
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmin, 411, "R: which.min(p) = 412 (1-based)");
    }

    /// R's predLoess returns NA outside range(x) for surface = "interpolate".
    #[test]
    fn test_loess_interpolate_outside_range_is_nan() {
        let x = r_seq(0.0, 8.0, 200);
        let y: Vec<f64> = x.iter().map(|&xi| xi.cos()).collect();
        let p = loess_interpolate(&x, &y, 0.2, &[-1e-9, 0.0, 8.0, 8.0 + 1e-9, f64::NAN]);
        assert!(p[0].is_nan() && p[3].is_nan() && p[4].is_nan());
        assert!(p[1].is_finite() && p[2].is_finite());
    }
}


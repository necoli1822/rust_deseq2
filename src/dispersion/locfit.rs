//! One-dimensional port of the parts of locfit 1.5-9.12 that DESeq2's
//! `localDispersionFit` uses: `locfit(y ~ x, weights = w)` with the defaults
//! (local quadratic, nn = 0.7, tricube kernel, quasi-Gaussian family,
//! identity link, `ev = rbox()`, i.e. the adaptive tree with cut = 0.8) and
//! `predict()` on that fit.
//!
//! Every step follows the C code closely, including the order of floating
//! point operations, so that vertex values and predictions agree with R to
//! the last bit or close to it:
//!
//! - the parametric component (`compparcomp` in pcomp.c): a global weighted
//!   quadratic centred at the weighted mean of x. It is subtracted from the
//!   local coefficients at every vertex (`subparcomp`) and added back after
//!   interpolation (`addparcomp`). Inside the data range this cancels (the
//!   cubic Hermite interpolant reproduces quadratics), but outside it the
//!   prediction is the linear extrapolation of the residual plus the global
//!   quadratic, not a linear extrapolation of the local fit;
//! - the neighbourhood (`nbhd` / `nbhd1` in lf_nbhd.c) and the summation
//!   order of the local likelihood, which follows the data order;
//! - the single Newton step of `max_nr` (m_max.c) from `reginit`'s start,
//!   solved through `jacob_solve` with the scaled Jacobi eigen decomposition
//!   (`jacob_dec` with JAC_EIGD, `eig_dec` / `eig_solve` in m_eigen.c);
//! - the adaptive tree (`atree_start`, `atree_grow`, `atree_split`,
//!   `newsplit` in ev_atree.c / ev_main.c) and its interpolation
//!   (`atree_int`, `exvvalpv`, `rectcell_interp`, `hermite2` in ev_interp.c).

/// Number of local polynomial coefficients for deg = 2, d = 1 (`calcp`)
const P: usize = 3;

/// A vertex of the adaptive tree
#[derive(Debug, Clone)]
struct Vertex {
    /// Location (log mean)
    x: f64,
    /// Local fit minus the parametric component (`fp.coef[v]`)
    value: f64,
    /// Local slope minus the parametric component's slope (`fp.coef[nvm + v]`)
    slope: f64,
    /// Bandwidth (`fp.h[v]`)
    h: f64,
    /// Pseudo vertex: not fitted, interpolated from its parents (`evs.s[v]`)
    pseudo: bool,
    /// Parent vertices (`evs.lo[v]`, `evs.hi[v]`)
    lo: usize,
    hi: usize,
}

/// The parametric component (`paramcomp`): coefficients of the global fit in
/// the basis [1, x - xbar, (x - xbar)^2 / 2]
#[derive(Debug, Clone)]
struct ParComp {
    xbar: f64,
    coef: [f64; P],
}

/// A locfit fit evaluated on its adaptive tree
#[derive(Debug, Clone)]
pub(crate) struct Locfit {
    vertices: Vec<Vertex>,
    pc: Option<ParComp>,
    cut: f64,
}

/// Data of the fit (`lfdata`), in the caller's order
struct Data<'a> {
    x: &'a [f64],
    y: &'a [f64],
    w: &'a [f64],
    /// x is non-decreasing (`lfd->ord`, set in startlf)
    ord: bool,
}

/// locfit error conditions that R reports with `error()`
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LocfitError {
    /// `newsplit: out of vertex space`
    OutOfVertexSpace,
    /// `compparcomp: no points in dataset?`
    NoPoints,
}

impl std::fmt::Display for LocfitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocfitError::OutOfVertexSpace => write!(f, "locfit: newsplit: out of vertex space"),
            LocfitError::NoPoints => write!(f, "locfit: compparcomp: no points in dataset?"),
        }
    }
}

/// Tricube weight (`W` with WTCUB in weight.c)
fn tricube(u: f64) -> f64 {
    let u = u.abs();
    if u > 1.0 {
        return 0.0;
    }
    let u = 1.0 - u * u * u;
    u * u * u
}

/// Local quadratic basis at data point x around fitting point t
/// (`fitfun` with deg = 2, d = 1, no derivatives)
fn basis(x: f64, t: f64) -> [f64; P] {
    let dx = x - t;
    let f1 = 1.0 * dx / 1.0;
    [1.0, f1, f1 * dx / 2.0]
}

/// The basis differentiated once (`fitfun` with one derivative)
fn basis_deriv(x: f64, t: f64) -> [f64; P] {
    let dx = x - t;
    [0.0, 1.0, 1.0 * dx / 1.0]
}

fn innerprod(a: &[f64; P], b: &[f64; P]) -> f64 {
    let mut s = 0.0;
    for i in 0..P {
        s += a[i] * b[i];
    }
    s
}

/// Jacobi eigen decomposition of a symmetric matrix (`eig_dec` in m_eigen.c).
/// On return `x` holds the eigenvalues on its diagonal and `p` the eigenvectors.
fn eig_dec(x: &mut [f64; P * P], p: &mut [f64; P * P]) {
    let d = P;
    for i in 0..d {
        for j in 0..d {
            p[i * d + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _iter in 0..20 {
        let mut ms = false;
        for i in 0..d {
            for j in (i + 1)..d {
                if x[i * d + j] * x[i * d + j] > 1.0e-15 * (x[i * d + i] * x[j * d + j]).abs() {
                    let mut c = (x[j * d + j] - x[i * d + i]) / 2.0;
                    let mut s = -x[i * d + j];
                    let r = (c * c + s * s).sqrt();
                    c /= r;
                    s = ((1.0 - c) / 2.0).sqrt() * (2.0 * ((s > 0.0) as i32 as f64) - 1.0);
                    c = ((1.0 + c) / 2.0).sqrt();
                    for k in 0..d {
                        let u = x[i * d + k];
                        let v = x[j * d + k];
                        x[i * d + k] = u * c + v * s;
                        x[j * d + k] = v * c - u * s;
                    }
                    for k in 0..d {
                        let u = x[k * d + i];
                        let v = x[k * d + j];
                        x[k * d + i] = u * c + v * s;
                        x[k * d + j] = v * c - u * s;
                    }
                    x[i * d + j] = 0.0;
                    x[j * d + i] = 0.0;
                    for k in 0..d {
                        let u = p[k * d + i];
                        let v = p[k * d + j];
                        p[k * d + i] = u * c + v * s;
                        p[k * d + j] = v * c - u * s;
                    }
                    ms = true;
                }
            }
        }
        if !ms {
            return;
        }
    }
    log::warn!("locfit: eig_dec not converged");
}

/// Solve (X'WX) v = f1 in place, returning the rank: `jacob_dec` with
/// JAC_EIGD (eigen decomposition of the correlation-scaled matrix) followed
/// by `jacob_solve` / `eig_solve`, eigenvalues below 1e-8 times the largest
/// being skipped as in `e_tol`.
fn jacob_solve(z: &[f64; P * P], v: &mut [f64; P]) -> usize {
    let mut zz = *z;
    let mut dg = [0.0; P];
    for i in 0..P {
        dg[i] = if zz[i * (P + 1)] <= 0.0 { 0.0 } else { 1.0 / zz[i * (P + 1)].sqrt() };
    }
    for i in 0..P {
        for j in 0..P {
            zz[i * P + j] *= dg[i] * dg[j];
        }
    }
    let mut q = [0.0; P * P];
    eig_dec(&mut zz, &mut q);

    for i in 0..P {
        v[i] *= dg[i];
    }
    // eig_solve
    let mut mx = zz[0];
    for i in 1..P {
        if zz[i * (P + 1)] > mx {
            mx = zz[i * (P + 1)];
        }
    }
    let tol = 1.0e-8 * mx;
    let mut w = [0.0; P];
    for i in 0..P {
        w[i] = 0.0;
        for j in 0..P {
            w[i] += q[j * P + i] * v[j];
        }
    }
    let mut rank = 0;
    for i in 0..P {
        if zz[i * P + i] > tol {
            w[i] /= zz[i * (P + 1)];
            rank += 1;
        }
    }
    for i in 0..P {
        v[i] = 0.0;
        for j in 0..P {
            v[i] += q[i * P + j] * w[j];
        }
    }
    for i in 0..P {
        v[i] *= dg[i];
    }
    rank
}

/// Gaussian local likelihood with identity link at `coef` (`likereg` with
/// `famgaus`): returns (log likelihood, gradient, X'WX).
fn likereg(
    data: &Data,
    ind: &[usize],
    kw: &[f64],
    xs: &[[f64; P]],
    coef: &[f64; P],
) -> (f64, [f64; P], [f64; P * P]) {
    let mut lk = 0.0;
    let mut f1 = [0.0; P];
    let mut z = [0.0; P * P];
    for (m, &ii) in ind.iter().enumerate() {
        let x = &xs[m];
        let th = 0.0 + innerprod(coef, x);
        let pw = data.w[ii];
        let r = data.y[ii] - th;
        let zlik = -pw * r * r / 2.0;
        let zdll = pw * r;
        let zddll = pw;
        let ww = kw[m];
        lk += ww * zlik;
        for j in 0..P {
            f1[j] += x[j] * ww * zdll;
        }
        let c = ww * zddll;
        for i in 0..P {
            for j in 0..P {
                z[i * P + j] += c * x[i] * x[j];
            }
        }
    }
    (lk, f1, z)
}

/// Local fit at `t` over the points `ind` with kernel weights `kw`
/// (`locfit` -> `lfinit` / `reginit` -> `lfiter` / `max_nr`). For the
/// Gaussian family `likereg` stops the iteration after one Newton step.
/// Returns None when no point has positive weight (LF_NOPT).
fn local_fit(data: &Data, ind: &[usize], kw: &[f64], t: f64) -> Option<[f64; P]> {
    let xs: Vec<[f64; P]> = ind.iter().map(|&ii| basis(data.x[ii], t)).collect();

    // reginit
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let sb = 0.0;
    for (m, &ii) in ind.iter().enumerate() {
        let pw = data.w[ii];
        s1 += kw[m] * (pw * data.y[ii]);
        s0 += kw[m] * pw;
    }
    if s0 == 0.0 {
        return None;
    }
    let mut coef = [0.0; P];
    coef[0] = (s1 - sb) / s0;

    // max_nr: one Newton step, then likereg returns NR_BREAK
    let (f, mut f1, z) = likereg(data, ind, kw, &xs, &coef);
    let old = coef;
    let rank = jacob_solve(&z, &mut f1);
    let mut delta = f1;
    if rank == 0 {
        delta[0] = -f / delta[0];
    }
    let lambda = 1.0;
    for j in 0..P {
        coef[j] = old[j] + lambda * delta[j];
    }
    Some(coef)
}

/// k-th smallest of `di` (`kordstat`): only its value matters
fn kordstat(di: &[f64], k: usize) -> f64 {
    if k < 1 {
        return 0.0;
    }
    let mut v = di.to_vec();
    let (_, kth, _) = v.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
    *kth
}

/// Neighbourhood of `t`: the points with positive tricube weight, in
/// locfit's order, their weights and the bandwidth (`nbhd` / `nbhd1`)
fn nbhd(data: &Data, t: f64, nn: usize) -> (Vec<usize>, Vec<f64>, f64) {
    let n = data.x.len();
    let xd = data.x;
    let mut ind = Vec::new();
    let mut kw = Vec::new();

    if nn <= n && data.ord {
        // nbhd1: fast search on ordered data
        let x = t;
        let mut z;
        if x <= xd[0] {
            z = 0;
        } else if x >= xd[n - 1] {
            z = n - 1;
        } else {
            let (mut l, mut r) = (0usize, n - 1);
            while r - l > 1 {
                z = (r + l) / 2;
                if xd[z] > x {
                    r = z;
                } else {
                    l = z;
                }
            }
            z = if (x - xd[l]) > (xd[r] - x) { r } else { l };
        }
        let mut h = 0.0;
        if nn > 0 {
            let k = nn;
            let (mut l, mut r) = (z, z);
            if l == 0 {
                r = k - 1;
            }
            if r == n - 1 {
                l = n - k;
            }
            while r - l < k - 1 {
                if (x - xd[l - 1]) < (xd[r + 1] - x) {
                    l -= 1;
                } else {
                    r += 1;
                }
                if l == 0 {
                    r = k - 1;
                }
                if r == n - 1 {
                    l = n - k;
                }
            }
            h = x - xd[l];
            if h < xd[r] - x {
                h = xd[r] - x;
            }
        }
        h /= 1.0;

        let mut zi = z as isize;
        if xd[z] > x {
            zi -= 1;
        }
        // look left
        let mut i = zi;
        while i >= 0 {
            let iu = i as usize;
            let di = (x - xd[iu]) / 1.0;
            let w = weight_sph(di, h);
            if w > 0.0 {
                ind.push(iu);
                kw.push(w);
            } else {
                break;
            }
            i -= 1;
        }
        // look right
        for i in ((zi + 1) as usize)..n {
            let di = (xd[i] - x) / 1.0;
            let w = weight_sph(di, h);
            if w > 0.0 {
                ind.push(i);
                kw.push(w);
            } else {
                break;
            }
        }
        return (ind, kw, h);
    }

    let di: Vec<f64> = xd.iter().map(|&xi| ((xi - t) / 1.0).abs()).collect();
    // compbandwid
    let h = if nn == 0 {
        0.0
    } else if nn < n {
        kordstat(&di, nn).max(0.0)
    } else {
        let mut nnh = 0.0_f64;
        for &d in &di {
            nnh = nnh.max(d);
        }
        (nnh * ((nn as f64 / n as f64).ln() / 1.0).exp()).max(0.0)
    };
    for i in 0..n {
        let w = weight_sph(di[i], h);
        if w > 0.0 {
            ind.push(i);
            kw.push(w);
        }
    }
    (ind, kw, h)
}

/// Spherical kernel weight (`weightsph`)
fn weight_sph(di: f64, h: f64) -> f64 {
    if h == 0.0 {
        return if di == 0.0 { 1.0 } else { 0.0 };
    }
    tricube(di / h)
}

/// Linear-then-cubic Hermite basis (`hermite2` in ev_interp.c): cubic inside
/// the cell, linear extrapolation from the nearer vertex outside it
fn hermite2(x: f64, z: f64) -> [f64; 4] {
    if z == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let h = x / z;
    if h < 0.0 {
        return [1.0, 0.0, h, 0.0];
    }
    if h > 1.0 {
        return [0.0, 1.0, 0.0, h - 1.0];
    }
    let phi1 = h * h * (3.0 - 2.0 * h);
    [1.0 - phi1, phi1, h * (1.0 - h) * (1.0 - h), h * h * (h - 1.0)]
}

impl Locfit {
    /// Fit `locfit(y ~ x, weights = w, alpha = nn, ev = rbox(cut = cut))`
    /// (`startlf` -> `compparcomp`, `atree_start`)
    pub(crate) fn fit(x: &[f64], y: &[f64], w: &[f64], nn_alpha: f64, cut: f64) -> Result<Self, LocfitError> {
        let n = x.len();
        let ord = (1..n).all(|i| x[i] >= x[i - 1]);
        let data = Data { x, y, w, ord };

        // compparcomp: xbar is the prior-weighted mean of x, then a global fit
        // with unit kernel weights over all points
        let mut sw = 0.0;
        let mut xbar = 0.0;
        for i in 0..n {
            let wt = w[i];
            sw += wt;
            xbar += x[i] * wt;
        }
        xbar /= sw;
        let all: Vec<usize> = (0..n).collect();
        let ones = vec![1.0; n];
        let pc_coef = local_fit(&data, &all, &ones, xbar).ok_or(LocfitError::NoPoints)?;
        let pc = ParComp { xbar, coef: pc_coef };

        let mut lf = Locfit { vertices: Vec::new(), pc: Some(pc), cut };

        // set_flim: the data range, no expansion
        let mut mn = x[0];
        let mut mx = x[0];
        for &xi in &x[1..] {
            mx = mx.max(xi);
            mn = mn.min(xi);
        }

        // atree_guessnv: vertex space
        let a0 = if nn_alpha > 1.0 { 1.0 } else { 1.0 / nn_alpha };
        let cu = cut.min(1.0);
        let vc = 2usize;
        let nvm = ((5.0 * a0 / cu + 1.0) * vc as f64) as usize;

        // the k used for the nearest neighbour bandwidth (locfit.c)
        let k = (n as f64 * nn_alpha + 1e-12) as usize;

        for &xv in &[mn, mx] {
            let v = lf.fit_vertex(&data, xv, k);
            lf.vertices.push(Vertex { lo: 0, hi: 0, ..v });
        }
        lf.grow(&data, k, nvm, [0, 1], mn, mx)?;
        Ok(lf)
    }

    /// Fit at a new vertex (`procv` -> `procvraw` -> `subparcomp`)
    fn fit_vertex(&self, data: &Data, xv: f64, k: usize) -> Vertex {
        let (ind, kw, h) = nbhd(data, xv, k);
        let (mut value, mut slope) = match local_fit(data, &ind, &kw, xv) {
            Some(c) => (c[0], c[1]),
            // LF_NOPT: procv warns and leaves the coefficients at zero
            None => (0.0, 0.0),
        };
        if let Some(pc) = &self.pc {
            value -= innerprod(&pc.coef, &basis(xv, pc.xbar));
            slope -= innerprod(&pc.coef, &basis_deriv(xv, pc.xbar));
        }
        Vertex { x: xv, value, slope, h, pseudo: false, lo: 0, hi: 0 }
    }

    /// Does the cell (ce[0], ce[1]) need splitting? (`atree_split` for d = 1)
    fn split(&self, ce: [usize; 2], ll: f64, ur: f64, fl: (f64, f64)) -> Option<f64> {
        let mut hmin = 0.0;
        for &c in &ce {
            let h = self.vertices[c].h;
            if h > 0.0 && (hmin == 0.0 || h < hmin) {
                hmin = h;
            }
        }
        let le = (ur - ll) / 1.0;
        let score = if hmin == 0.0 { 2.0 * (ur - ll) / (fl.1 - fl.0) } else { le / hmin };
        if self.cut < score {
            Some(le)
        } else {
            None
        }
    }

    fn box_limits(&self) -> (f64, f64) {
        (self.vertices[0].x, self.vertices[1].x)
    }

    /// Recursively grow the tree (`atree_grow` and `newsplit` for d = 1)
    fn grow(&mut self, data: &Data, k: usize, nvm: usize, ce: [usize; 2], ll: f64, ur: f64) -> Result<(), LocfitError> {
        let fl = self.box_limits();
        let le = match self.split(ce, ll, ur, fl) {
            None => return Ok(()),
            Some(le) => le,
        };
        let (i0, i1) = (ce[1], ce[0]);
        let pv = le < self.cut * self.vertices[i0].h.min(self.vertices[i1].h);
        let nv = self.newsplit(data, k, nvm, i0, i1, pv)?;
        let mid = (ur + ll) / 2.0;
        self.grow(data, k, nvm, [ce[0], nv], ll, mid)?;
        self.grow(data, k, nvm, [nv, ce[1]], mid, ur)
    }

    /// Index of the vertex splitting (i0, i1), if it exists (`findpt`)
    fn findpt(&self, i0: usize, i1: usize) -> Option<usize> {
        let (i0, i1) = if i0 > i1 { (i1, i0) } else { (i0, i1) };
        ((i1 + 1)..self.vertices.len()).find(|&i| self.vertices[i].lo == i0 && self.vertices[i].hi == i1)
    }

    fn newsplit(&mut self, data: &Data, k: usize, nvm: usize, i0: usize, i1: usize, pv: bool) -> Result<usize, LocfitError> {
        if let Some(i) = self.findpt(i0, i1) {
            return Ok(i);
        }
        let (i0, i1) = if i0 > i1 { (i1, i0) } else { (i0, i1) };
        if self.vertices.len() == nvm {
            return Err(LocfitError::OutOfVertexSpace);
        }
        let xv = (self.vertices[i0].x + self.vertices[i1].x) / 2.0;
        let v = if pv {
            Vertex {
                x: xv,
                value: 0.0,
                slope: 0.0,
                h: (self.vertices[i0].h + self.vertices[i1].h) / 2.0,
                pseudo: true,
                lo: i0,
                hi: i1,
            }
        } else {
            Vertex { lo: i0, hi: i1, ..self.fit_vertex(data, xv, k) }
        };
        self.vertices.push(v);
        Ok(self.vertices.len() - 1)
    }

    /// Predict at x (`dointpoint` with ETREE: `atree_int` + `addparcomp`)
    pub(crate) fn predict(&self, x: f64) -> f64 {
        let fl = self.box_limits();
        let mut ce = [0usize, 1usize];
        let mut vv = [
            [self.vertices[0].value, self.vertices[0].slope],
            [self.vertices[1].value, self.vertices[1].slope],
        ];
        loop {
            let ll = self.vertices[ce[0]].x;
            let ur = self.vertices[ce[1]].x;
            if self.split(ce, ll, ur, fl).is_none() {
                break;
            }
            let h = ur - ll;
            let lo = (2.0 * (x - ll)) < h;
            let nv = self.findpt(ce[0], ce[1]).expect("locfit: Descend tree problem");
            let vn = &self.vertices[nv];
            let new = if vn.pseudo {
                // exvvalpv
                let (vl, vr) = (vv[0], vv[1]);
                [
                    (vl[0] + vr[0]) / 2.0 + h * (vl[1] - vr[1]) / 8.0,
                    1.5 * (vr[0] - vl[0]) / h - (vl[1] + vr[1]) / 4.0,
                ]
            } else {
                [vn.value, vn.slope]
            };
            if lo {
                ce[1] = nv;
                vv[1] = new;
            } else {
                ce[0] = nv;
                vv[0] = new;
            }
        }
        // rectcell_interp with nc = d + 1
        let ll = self.vertices[ce[0]].x;
        let ur = self.vertices[ce[1]].x;
        let mut phi = hermite2(x - ll, ur - ll);
        phi[2] *= ur - ll;
        phi[3] *= ur - ll;
        let mut f = phi[0] * vv[0][0] + phi[1] * vv[1][0] + phi[2] * vv[0][1] + phi[3] * vv[1][1];
        if let Some(pc) = &self.pc {
            f += innerprod(&pc.coef, &basis(x, pc.xbar));
        }
        f
    }
}

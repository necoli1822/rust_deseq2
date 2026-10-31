//! Port of R's L-BFGS-B (src/appl/lbfgsb.c, "L-BFGS-B (version 2.3)", an f2c
//! translation of the Fortran code by Zhu, Byrd, Lu-Chen and Nocedal).
//!
//! The reverse-communication interface is kept: [`Lbfgsb::setulb`] is called
//! repeatedly with a task string ("START", "FG...", "NEW_X", ...) exactly as
//! the driver `lbfgsb()` in R's optim.c does. The variables that the C code
//! keeps in `static` locals of `mainlb()` and `dcsrch()` are fields of
//! [`Lbfgsb`]. Arithmetic is written in the same order as the C code (R is
//! compiled without FMA contraction); the BLAS/LINPACK calls go through
//! [`super::blas`]. The printing routines `prn1lb`, `prn2lb`, `prn3lb` are
//! omitted (R only prints for `trace > 1`).
//!
//! Index conventions: the C code uses 1-based indices after its "Parameter
//! adjustments"; here arrays are 0-based Rust slices and every access is
//! written `a[i - 1]` (vectors) or through a column-major index helper
//! (matrices), keeping the C loop bounds. Integer index arrays (`indx`,
//! `iwhere`, `indx2`, `iorder`) store the C (1-based) values.

use super::blas::{daxpy, ddot, dpofa, dscal, dtrsl};

/// C macro `max(a, b)`: `(a < b) ? b : a`.
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// C macro `min(a, b)`: `(a > b) ? b : a`.
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if a > b {
        b
    } else {
        a
    }
}

/// Working storage and saved state of `setulb`/`mainlb`/`lnsrlb`/`dcsrch`.
pub(crate) struct Lbfgsb {
    n: usize,
    m: usize,
    // Work arrays partitioned from `wa` by setulb (all zero-initialised,
    // R allocates them with S_alloc).
    ws: Vec<f64>,
    wy: Vec<f64>,
    sy: Vec<f64>,
    ss: Vec<f64>,
    wt: Vec<f64>,
    wn: Vec<f64>,
    snd: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
    d: Vec<f64>,
    t: Vec<f64>,
    wa: Vec<f64>,
    // iwa partition
    indx: Vec<i32>,
    iwhere: Vec<i32>,
    indx2: Vec<i32>,
    csave: String,
    // mainlb statics: formerly lsave
    prjctd: bool,
    cnstnd: bool,
    boxed: bool,
    updatd: bool,
    // mainlb statics: formerly isave
    nintol: i32,
    iback: i32,
    nskip: i32,
    head: usize,
    col: usize,
    itail: usize,
    iter: i32,
    iupdat: usize,
    nint: i32,
    /// Total number of function and gradient evaluations (`isave[12]`).
    pub(crate) nfgv: i32,
    info: i32,
    ifun: i32,
    iword: i32,
    nfree: usize,
    nact: usize,
    ileave: usize,
    nenter: usize,
    // mainlb statics: formerly dsave
    theta: f64,
    fold: f64,
    tol: f64,
    dnorm: f64,
    epsmch: f64,
    gd: f64,
    stpmx: f64,
    sbgnrm: f64,
    stp: f64,
    gdold: f64,
    dtd: f64,
    xstep: f64,
    // `wrk` is a local of mainlb that is set before every use in one call
    wrk: bool,
    // dcsrch statics
    ls: Dcsrch,
}

#[derive(Default)]
struct Dcsrch {
    stage: i32,
    brackt: bool,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

/// Labels of `mainlb()` that execution can jump to.
#[derive(Clone, Copy)]
enum Label {
    L111,
    L222,
    L333,
    L555,
    L666,
    L777,
}

impl Lbfgsb {
    /// Allocate the working arrays (R: `wa`, `iwa` in optim.c `lbfgsb()`
    /// and the partition in `setulb()`).
    pub(crate) fn new(n: usize, m: usize) -> Self {
        Lbfgsb {
            n,
            m,
            ws: vec![0.0; m * n],
            wy: vec![0.0; m * n],
            sy: vec![0.0; m * m],
            ss: vec![0.0; m * m],
            wt: vec![0.0; m * m],
            wn: vec![0.0; 4 * m * m],
            snd: vec![0.0; 4 * m * m],
            z: vec![0.0; n],
            r: vec![0.0; n],
            d: vec![0.0; n],
            t: vec![0.0; n],
            wa: vec![0.0; 8 * m],
            indx: vec![0; n],
            iwhere: vec![0; n],
            indx2: vec![0; n],
            csave: String::new(),
            prjctd: false,
            cnstnd: false,
            boxed: false,
            updatd: false,
            nintol: 0,
            iback: 0,
            nskip: 0,
            head: 1,
            col: 0,
            itail: 0,
            iter: 0,
            iupdat: 0,
            nint: 0,
            nfgv: 0,
            info: 0,
            ifun: 0,
            iword: 0,
            nfree: n,
            nact: 0,
            ileave: 0,
            nenter: 0,
            theta: 1.0,
            fold: 0.0,
            tol: 0.0,
            dnorm: 0.0,
            epsmch: f64::EPSILON,
            gd: 0.0,
            stpmx: 0.0,
            sbgnrm: 0.0,
            stp: 0.0,
            gdold: 0.0,
            dtd: 0.0,
            xstep: 0.0,
            wrk: false,
            ls: Dcsrch::default(),
        }
    }

    /// R: `setulb()` -- one reverse-communication step of L-BFGS-B.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn setulb(
        &mut self,
        x: &mut [f64],
        l: &[f64],
        u: &[f64],
        nbd: &[i32],
        f: &mut f64,
        g: &mut [f64],
        factr: f64,
        pgtol: f64,
        task: &mut String,
    ) {
        // `csave` is a local of setulb in R, emptied at every call
        self.csave.clear();
        self.mainlb(x, l, u, nbd, f, g, factr, pgtol, task);
    }

    /// Reset of the limited-memory matrix after a failure
    /// ("refresh the lbfgs memory and restart the iteration").
    fn refresh(&mut self) {
        self.info = 0;
        self.col = 0;
        self.head = 1;
        self.theta = 1.0;
        self.iupdat = 0;
        self.updatd = false;
    }

    /// R: `mainlb()`.
    #[allow(clippy::too_many_arguments)]
    fn mainlb(
        &mut self,
        x: &mut [f64],
        l: &[f64],
        u: &[f64],
        nbd: &[i32],
        f: &mut f64,
        g: &mut [f64],
        factr: f64,
        pgtol: f64,
        task: &mut String,
    ) {
        let n = self.n;
        let m = self.m;
        let mut label;

        if task.starts_with("START") {
            self.epsmch = f64::EPSILON;
            self.fold = 0.0;
            self.dnorm = 0.0;
            self.gd = 0.0;
            self.sbgnrm = 0.0;
            self.stp = 0.0;
            self.xstep = 0.0;
            self.stpmx = 0.0;
            self.gdold = 0.0;
            self.dtd = 0.0;
            self.col = 0;
            self.head = 1;
            self.theta = 1.0;
            self.iupdat = 0;
            self.updatd = false;
            self.iback = 0;
            self.itail = 0;
            self.ifun = 0;
            self.iword = 0;
            self.nact = 0;
            self.ileave = 0;
            self.nenter = 0;
            self.iter = 0;
            self.nfgv = 0;
            self.nint = 0;
            self.nintol = 0;
            self.nskip = 0;
            self.nfree = n;
            self.tol = factr * self.epsmch;
            self.info = 0;
            errclb(n, m, factr, l, u, nbd, task, &mut self.info);
            if task.starts_with("ERROR") {
                return;
            }
            let (prjctd, cnstnd, boxed) = active(n, l, u, nbd, x, &mut self.iwhere);
            self.prjctd = prjctd;
            self.cnstnd = cnstnd;
            self.boxed = boxed;
            // Compute f0 and g0: return to the driver, re-enter at 111.
            *task = "FG_START".to_string();
            return;
        } else if task.starts_with("FG_LN") {
            label = Label::L666;
        } else if task.starts_with("NEW_X") {
            label = Label::L777;
        } else if task.starts_with("FG_ST") {
            label = Label::L111;
        } else if task.starts_with("STOP") {
            if task.len() >= 9 && &task[6..9] == "CPU" {
                x[..n].copy_from_slice(&self.t[..n]);
                g[..n].copy_from_slice(&self.r[..n]);
                *f = self.fold;
            }
            return;
        } else {
            *task = "FG_START".to_string();
            return;
        }

        loop {
            match label {
                Label::L111 => {
                    self.nfgv = 1;
                    self.sbgnrm = projgr(n, l, u, nbd, x, g);
                    if self.sbgnrm <= pgtol {
                        *task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".to_string();
                        return;
                    }
                    label = Label::L222;
                }
                Label::L222 => {
                    self.iword = -1;
                    if !self.cnstnd && self.col > 0 {
                        self.z[..n].copy_from_slice(&x[..n]);
                        self.wrk = self.updatd;
                        self.nint = 0;
                        label = Label::L333;
                        continue;
                    }
                    // Compute the Generalized Cauchy Point (GCP).
                    {
                        let (p, rest) = self.wa.split_at_mut(2 * m);
                        let (c, rest) = rest.split_at_mut(2 * m);
                        let (wbp, v) = rest.split_at_mut(2 * m);
                        self.info = cauchy(
                            n,
                            x,
                            l,
                            u,
                            nbd,
                            g,
                            &mut self.indx2,
                            &mut self.iwhere,
                            &mut self.t,
                            &mut self.d,
                            &mut self.z,
                            m,
                            &self.wy,
                            &self.ws,
                            &self.sy,
                            &self.wt,
                            self.theta,
                            self.col,
                            self.head,
                            p,
                            c,
                            wbp,
                            v,
                            &mut self.nint,
                            self.sbgnrm,
                            self.epsmch,
                        );
                    }
                    if self.info != 0 {
                        // singular triangular system detected; refresh the lbfgs memory.
                        self.refresh();
                        label = Label::L222;
                        continue;
                    }
                    self.nintol += self.nint;
                    // Count the entering and leaving variables for iter > 0;
                    // find the index set of free and active variables at the GCP.
                    self.wrk = freev(
                        n,
                        &mut self.nfree,
                        &mut self.indx,
                        &mut self.nenter,
                        &mut self.ileave,
                        &mut self.indx2,
                        &self.iwhere,
                        self.updatd,
                        self.cnstnd,
                        self.iter,
                    );
                    self.nact = n - self.nfree;
                    label = Label::L333;
                }
                Label::L333 => {
                    // If there are no free variables or B=theta*I, skip the
                    // subspace minimization.
                    if self.nfree == 0 || self.col == 0 {
                        label = Label::L555;
                        continue;
                    }
                    if self.wrk {
                        self.info = formk(
                            n,
                            self.nfree,
                            &self.indx,
                            self.nenter,
                            self.ileave,
                            &self.indx2,
                            self.iupdat,
                            self.updatd,
                            &mut self.wn,
                            &mut self.snd,
                            m,
                            &self.ws,
                            &self.wy,
                            &self.sy,
                            self.theta,
                            self.col,
                            self.head,
                        );
                    }
                    if self.info != 0 {
                        // nonpositive definiteness in Cholesky factorization;
                        self.refresh();
                        label = Label::L222;
                        continue;
                    }
                    // compute r=-Z'B(xcp-xk)-Z'g (using wa(2m+1)=W'(xcp-x) from 'cauchy').
                    self.info = cmprlb(
                        n,
                        m,
                        x,
                        g,
                        &self.ws,
                        &self.wy,
                        &self.sy,
                        &self.wt,
                        &self.z,
                        &mut self.r,
                        &mut self.wa,
                        &self.indx,
                        self.theta,
                        self.col,
                        self.head,
                        self.nfree,
                        self.cnstnd,
                    );
                    if self.info == 0 {
                        // call the direct method.
                        let (wv, _) = self.wa.split_at_mut(2 * m);
                        self.info = subsm(
                            n,
                            m,
                            self.nfree,
                            &self.indx,
                            l,
                            u,
                            nbd,
                            &mut self.z,
                            &mut self.r,
                            &self.ws,
                            &self.wy,
                            self.theta,
                            self.col,
                            self.head,
                            &mut self.iword,
                            wv,
                            &self.wn,
                        );
                    }
                    // L444
                    if self.info != 0 {
                        self.refresh();
                        label = Label::L222;
                        continue;
                    }
                    label = Label::L555;
                }
                Label::L555 => {
                    // Generate the search direction d:=z-x.
                    for i in 1..=n {
                        self.d[i - 1] = self.z[i - 1] - x[i - 1];
                    }
                    label = Label::L666;
                }
                Label::L666 => {
                    self.lnsrlb(x, l, u, nbd, f, g, task);
                    if self.info != 0 || self.iback >= 20 {
                        // restore the previous iterate.
                        x[..n].copy_from_slice(&self.t[..n]);
                        g[..n].copy_from_slice(&self.r[..n]);
                        *f = self.fold;
                        if self.col == 0 {
                            // abnormal termination.
                            if self.info == 0 {
                                self.info = -9;
                                // restore the actual number of f and g evaluations etc.
                                self.nfgv -= 1;
                                self.ifun -= 1;
                                self.iback -= 1;
                            }
                            *task = "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH".to_string();
                            self.iter += 1;
                            return;
                        } else {
                            // refresh the lbfgs memory and restart the iteration.
                            if self.info == 0 {
                                self.nfgv -= 1;
                            }
                            self.refresh();
                            *task = "RESTART_FROM_LNSRCH".to_string();
                            label = Label::L222;
                            continue;
                        }
                    } else if task.starts_with("FG_LN") {
                        // return to the driver for calculating f and g; reenter at 666.
                        return;
                    } else {
                        // calculate and print out the quantities related to the new X.
                        self.iter += 1;
                        self.sbgnrm = projgr(n, l, u, nbd, x, g);
                        return;
                    }
                }
                Label::L777 => {
                    // Test for termination.
                    if self.sbgnrm <= pgtol {
                        *task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".to_string();
                        return;
                    }
                    let d1 = cmax(self.fold.abs(), f.abs());
                    let ddum = cmax(d1, 1.0);
                    if self.fold - *f <= self.tol * ddum {
                        *task = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH".to_string();
                        if self.iback >= 10 {
                            self.info = -5;
                        }
                        return;
                    }
                    // Compute d=newx-oldx, r=newg-oldg, rr=y'y and dr=y's.
                    for i in 1..=n {
                        self.r[i - 1] = g[i - 1] - self.r[i - 1];
                    }
                    let rr = ddot(n, &self.r, &self.r);
                    let dr;
                    let ddum;
                    if self.stp == 1.0 {
                        dr = self.gd - self.gdold;
                        ddum = -self.gdold;
                    } else {
                        dr = (self.gd - self.gdold) * self.stp;
                        dscal(n, self.stp, &mut self.d);
                        ddum = -self.gdold * self.stp;
                    }
                    if dr <= self.epsmch * ddum {
                        // skip the L-BFGS update.
                        self.nskip += 1;
                        self.updatd = false;
                        label = Label::L222;
                        continue;
                    }
                    // Update the L-BFGS matrix.
                    self.updatd = true;
                    self.iupdat += 1;
                    matupd(
                        n,
                        m,
                        &mut self.ws,
                        &mut self.wy,
                        &mut self.sy,
                        &mut self.ss,
                        &self.d,
                        &self.r,
                        &mut self.itail,
                        self.iupdat,
                        &mut self.col,
                        &mut self.head,
                        &mut self.theta,
                        rr,
                        dr,
                        self.stp,
                        self.dtd,
                    );
                    // Form the upper half of the pds T = theta*SS + L*D^(-1)*L';
                    // Cholesky factorize T to J*J' with J' stored in the upper
                    // triangular of wt.
                    self.info = formt(m, &mut self.wt, &self.sy, &self.ss, self.col, self.theta);
                    if self.info != 0 {
                        self.refresh();
                    }
                    // L888: the end of the loop
                    label = Label::L222;
                }
            }
        }
    }

    /// R: `lnsrlb()` -- line search via `dcsrch`, safeguarded to stay in the box.
    #[allow(clippy::too_many_arguments)]
    fn lnsrlb(
        &mut self,
        x: &mut [f64],
        l: &[f64],
        u: &[f64],
        nbd: &[i32],
        f: &mut f64,
        g: &[f64],
        task: &mut String,
    ) {
        const STPMIN: f64 = 0.0;
        const FTOL: f64 = 0.001;
        const GTOL: f64 = 0.9;
        const XTOL: f64 = 0.1;
        let n = self.n;

        if !task.starts_with("FG_LN") {
            self.dtd = ddot(n, &self.d, &self.d);
            self.dnorm = self.dtd.sqrt();
            // Determine the maximum step length.
            self.stpmx = 1e10;
            if self.cnstnd {
                if self.iter == 0 {
                    self.stpmx = 1.0;
                } else {
                    for i in 1..=n {
                        let a1 = self.d[i - 1];
                        if nbd[i - 1] != 0 {
                            if a1 < 0.0 && nbd[i - 1] <= 2 {
                                let a2 = l[i - 1] - x[i - 1];
                                if a2 >= 0.0 {
                                    self.stpmx = 0.0;
                                } else if a1 * self.stpmx < a2 {
                                    self.stpmx = a2 / a1;
                                }
                            } else if a1 > 0.0 && nbd[i - 1] >= 2 {
                                let a2 = u[i - 1] - x[i - 1];
                                if a2 <= 0.0 {
                                    self.stpmx = 0.0;
                                } else if a1 * self.stpmx > a2 {
                                    self.stpmx = a2 / a1;
                                }
                            }
                        }
                    }
                }
            }
            if self.iter == 0 && !self.boxed {
                let d1 = 1.0 / self.dnorm;
                self.stp = cmin(d1, self.stpmx);
            } else {
                self.stp = 1.0;
            }
            self.t[..n].copy_from_slice(&x[..n]);
            self.r[..n].copy_from_slice(&g[..n]);
            self.fold = *f;
            self.ifun = 0;
            self.iback = 0;
            self.csave = "START".to_string();
        }
        // L556
        self.gd = ddot(n, g, &self.d);
        if self.ifun == 0 {
            self.gdold = self.gd;
            if self.gd >= 0.0 {
                // the directional derivative >=0. Line search is impossible.
                self.info = -4;
                return;
            }
        }
        let mut gd = self.gd;
        let mut stp = self.stp;
        let stpmx = self.stpmx;
        let mut csave = std::mem::take(&mut self.csave);
        self.ls.dcsrch(f, &mut gd, &mut stp, FTOL, GTOL, XTOL, STPMIN, stpmx, &mut csave);
        self.gd = gd;
        self.stp = stp;
        self.csave = csave;
        self.xstep = self.stp * self.dnorm;
        if !self.csave.starts_with("CONV") && !self.csave.starts_with("WARN") {
            *task = "FG_LNSRCH".to_string();
            self.ifun += 1;
            self.nfgv += 1;
            self.iback = self.ifun - 1;
            if self.stp == 1.0 {
                x[..n].copy_from_slice(&self.z[..n]);
            } else {
                for i in 1..=n {
                    x[i - 1] = self.stp * self.d[i - 1] + self.t[i - 1];
                }
            }
        } else {
            *task = "NEW_X".to_string();
        }
    }
}

/// R: `active()` -- initialise `iwhere` and project `x` onto the feasible set.
/// Returns `(prjctd, cnstnd, boxed)`.
fn active(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &mut [f64], iwhere: &mut [i32]) -> (bool, bool, bool) {
    let mut prjctd = false;
    let mut cnstnd = false;
    let mut boxed = true;
    // Project the initial x to the feasible set if necessary.
    for i in 1..=n {
        if nbd[i - 1] > 0 {
            if nbd[i - 1] <= 2 && x[i - 1] <= l[i - 1] {
                if x[i - 1] < l[i - 1] {
                    prjctd = true;
                    x[i - 1] = l[i - 1];
                }
            } else if nbd[i - 1] >= 2 && x[i - 1] >= u[i - 1] && x[i - 1] > u[i - 1] {
                prjctd = true;
                x[i - 1] = u[i - 1];
            }
        }
    }
    // Initialize iwhere and assign values to cnstnd and boxed.
    for i in 1..=n {
        if nbd[i - 1] != 2 {
            boxed = false;
        }
        if nbd[i - 1] == 0 {
            // this variable is always free
            iwhere[i - 1] = -1;
        } else {
            cnstnd = true;
            if nbd[i - 1] == 2 && u[i - 1] - l[i - 1] <= 0.0 {
                // this variable is always fixed
                iwhere[i - 1] = 3;
            } else {
                iwhere[i - 1] = 0;
            }
        }
    }
    (prjctd, cnstnd, boxed)
}

/// R: `bmv()` -- product of the 2m x 2m middle matrix of the compact L-BFGS
/// formula with a 2col vector `v`, returned in `p`. Returns `info`.
fn bmv(m: usize, sy: &[f64], wt: &[f64], col: usize, v: &[f64], p: &mut [f64]) -> i32 {
    let sy_ = |i: usize, j: usize| sy[(i - 1) + (j - 1) * m];
    if col == 0 {
        return 0;
    }
    // PART I: solve [ D^(1/2) O ; -L*D^(-1/2) J ] [p1;p2] = [v1;v2].
    // solve Jp2=v2+LD^(-1)v1.
    p[col] = v[col];
    for i in 2..=col {
        let i2 = col + i;
        let mut sum = 0.0;
        for k in 1..=i - 1 {
            sum += sy_(i, k) * v[k - 1] / sy_(k, k);
        }
        p[i2 - 1] = v[i2 - 1] + sum;
    }
    // Solve the triangular system
    if dtrsl(wt, 0, m, col, &mut p[col..2 * col], 11) != 0 {
        return 1;
    }
    // solve D^(1/2)p1=v1.
    for i in 1..=col {
        p[i - 1] = v[i - 1] / sy_(i, i).sqrt();
    }
    // PART II: solve [ -D^(1/2) D^(-1/2)*L' ; 0 J' ] [p1;p2] = [p1;p2].
    // solve J^Tp2=p2.
    if dtrsl(wt, 0, m, col, &mut p[col..2 * col], 1) != 0 {
        return 1;
    }
    // compute p1=-D^(-1/2)(p1-D^(-1/2)L'p2) =-D^(-1/2)p1 + D^(-1)L'p2.
    for i in 1..=col {
        p[i - 1] = -p[i - 1] / sy_(i, i).sqrt();
    }
    for i in 1..=col {
        let mut sum = 0.0;
        for k in i + 1..=col {
            sum += sy_(k, i) * p[col + k - 1] / sy_(i, i);
        }
        p[i - 1] += sum;
    }
    0
}

/// R: `cauchy()` -- the generalized Cauchy point. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn cauchy(
    n: usize,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
    iorder: &mut [i32],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    m: usize,
    wy: &[f64],
    ws: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    p: &mut [f64],
    c: &mut [f64],
    wbp: &mut [f64],
    v: &mut [f64],
    nint: &mut i32,
    sbgnrm: f64,
    epsmch: f64,
) -> i32 {
    let wy_ = |i: usize, j: usize| wy[(i - 1) + (j - 1) * n];
    let ws_ = |i: usize, j: usize| ws[(i - 1) + (j - 1) * n];

    if sbgnrm <= 0.0 {
        xcp[..n].copy_from_slice(&x[..n]);
        return 0;
    }
    let mut bnded = true;
    let mut nfree = n + 1;
    let mut nbreak = 0usize;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = col << 1;
    let mut f1 = 0.0;
    let mut tl = 0.0;
    let mut tu = 0.0;

    // We set p to zero and build it up as we determine d.
    for i in 1..=col2 {
        p[i - 1] = 0.0;
    }

    // In the following loop we determine for each variable its bound
    // status and its breakpoint, and update p accordingly.
    // Smallest breakpoint is identified.
    for i in 1..=n {
        let neggi = -g[i - 1];
        if iwhere[i - 1] != 3 && iwhere[i - 1] != -1 {
            // if x(i) is not a constant and has bounds,
            // compute the difference between x(i) and its bounds.
            if nbd[i - 1] <= 2 {
                tl = x[i - 1] - l[i - 1];
            }
            if nbd[i - 1] >= 2 {
                tu = u[i - 1] - x[i - 1];
            }
            // If a variable is close enough to a bound we treat it as at bound.
            let xlower = nbd[i - 1] <= 2 && tl <= 0.0;
            let xupper = nbd[i - 1] >= 2 && tu <= 0.0;
            // reset iwhere(i).
            iwhere[i - 1] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i - 1] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i - 1] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i - 1] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i - 1] != 0 && iwhere[i - 1] != -1 {
            d[i - 1] = 0.0;
        } else {
            d[i - 1] = neggi;
            f1 -= neggi * neggi;
            // calculate p := p - W'e_i* (g_i).
            for j in 1..=col {
                p[j - 1] += wy_(i, pointr) * neggi;
                p[col + j - 1] += ws_(i, pointr) * neggi;
                pointr = pointr % m + 1;
            }
            if nbd[i - 1] <= 2 && nbd[i - 1] != 0 && neggi < 0.0 {
                // x(i) + d(i) is bounded; compute t(i).
                nbreak += 1;
                iorder[nbreak - 1] = i as i32;
                t[nbreak - 1] = tl / (-neggi);
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else if nbd[i - 1] >= 2 && neggi > 0.0 {
                // x(i) + d(i) is bounded; compute t(i).
                nbreak += 1;
                iorder[nbreak - 1] = i as i32;
                t[nbreak - 1] = tu / neggi;
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else {
                // x(i) + d(i) is not bounded.
                nfree -= 1;
                iorder[nfree - 1] = i as i32;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }

    // The indices of the nonzero components of d are now stored
    // in iorder(1),...,iorder(nbreak) and iorder(nfree),...,iorder(n).
    // The smallest of the nbreak breakpoints is in t(ibkmin)=bkmin.
    if theta != 1.0 {
        // complete the initialization of p for theta not= one.
        dscal(col, theta, &mut p[col..]);
    }
    // Initialize GCP xcp = x.
    xcp[..n].copy_from_slice(&x[..n]);
    if nbreak == 0 && nfree == n + 1 {
        // is a zero vector, return with the initial xcp as GCP.
        return 0;
    }
    // Initialize c = W'(xcp - x) = 0.
    for j in 1..=col2 {
        c[j - 1] = 0.0;
    }

    // Initialize derivative f2.
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let info = bmv(m, sy, wt, col, p, v);
        if info != 0 {
            return info;
        }
        f2 -= ddot(col2, v, p);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    *nint = 1;

    // If there are no breakpoints, locate the GCP and return.
    let mut skip_to_999 = false;
    if nbreak > 0 {
        let mut nleft = nbreak;
        let mut iter = 1usize;
        let mut tj = 0.0;
        // ------------------- the beginning of the loop -------------------------
        loop {
            // Find the next smallest breakpoint;
            // compute dt = t(nleft) - t(nleft + 1).
            let tj0 = tj;
            let ibp: usize;
            if iter == 1 {
                // Since we already have the smallest breakpoint we need not do
                // heapsort yet. Often only one breakpoint is used and the
                // cost of heapsort is avoided.
                tj = bkmin;
                ibp = iorder[ibkmin - 1] as usize;
            } else {
                if iter == 2 {
                    // Replace the already used smallest breakpoint with the
                    // breakpoint numbered nbreak > nlast, before heapsort call.
                    if ibkmin != nbreak {
                        t[ibkmin - 1] = t[nbreak - 1];
                        iorder[ibkmin - 1] = iorder[nbreak - 1];
                    }
                }
                // Update heap structure of breakpoints (if iter=2, initialize heap).
                hpsolb(nleft, t, iorder, iter as i32 - 2);
                tj = t[nleft - 1];
                ibp = iorder[nleft - 1] as usize;
            }
            let dt = tj - tj0;

            // If a minimizer is within this interval, locate the GCP and return.
            if dtm < dt {
                break;
            }
            // Otherwise fix one variable and reset the corresponding component of d to zero.
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp - 1];
            d[ibp - 1] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = u[ibp - 1] - x[ibp - 1];
                xcp[ibp - 1] = u[ibp - 1];
                iwhere[ibp - 1] = 2;
            } else {
                zibp = l[ibp - 1] - x[ibp - 1];
                xcp[ibp - 1] = l[ibp - 1];
                iwhere[ibp - 1] = 1;
            }
            if nleft == 0 && nbreak == n {
                // all n variables are fixed, return with xcp as GCP.
                dtm = dt;
                skip_to_999 = true;
                break;
            }
            // Update the derivative information.
            *nint += 1;
            let dibp2 = dibp * dibp;
            // Update f1 and f2. temporarily set f1 and f2 for col=0.
            f1 += dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                // update c = c + dt*p.
                daxpy(col2, dt, p, c);
                // choose wbp, the row of W corresponding to the breakpoint encountered.
                let mut pointr = head;
                for j in 1..=col {
                    wbp[j - 1] = wy_(ibp, pointr);
                    wbp[col + j - 1] = theta * ws_(ibp, pointr);
                    pointr = pointr % m + 1;
                }
                // compute (wbp)Mc, (wbp)Mp, and (wbp)M(wbp)'.
                let info = bmv(m, sy, wt, col, wbp, v);
                if info != 0 {
                    return info;
                }
                let wmc = ddot(col2, c, v);
                let wmp = ddot(col2, p, v);
                let wmw = ddot(col2, wbp, v);
                // update p = p - dibp*wbp.
                daxpy(col2, -dibp, wbp, p);
                // complete updating f1 and f2 while col > 0.
                f1 += dibp * wmc;
                f2 += 2.0 * dibp * wmp - dibp2 * wmw;
            }
            let d1 = epsmch * f2_org;
            if f2 < d1 {
                f2 = d1;
            }
            if nleft > 0 {
                dtm = -f1 / f2;
                continue;
                // to repeat the loop for unsearched intervals.
            } else if bnded {
                f1 = 0.0;
                f2 = 0.0;
                dtm = 0.0;
            } else {
                dtm = -f1 / f2;
            }
            let _ = (f1, f2);
            break;
        }
        // ------------------- the end of the loop -------------------------------
    }
    if !skip_to_999 {
        // L888
        if dtm <= 0.0 {
            dtm = 0.0;
        }
        tsum += dtm;
        // Move free variables (i.e., the ones w/o breakpoints) and
        // the variables whose breakpoints haven't been reached.
        daxpy(n, tsum, d, xcp);
    }
    // L999: Update c = c + dtm*p = W'(x^c - x)
    // which will be used in computing r = Z'(B(x^c - x) + g).
    if col > 0 {
        daxpy(col2, dtm, p, c);
    }
    0
}

/// R: `cmprlb()` -- compute r=-Z'B(xcp-xk)-Z'g using wa(2m+1)=W'(xcp-x)
/// from `cauchy`. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn cmprlb(
    n: usize,
    m: usize,
    x: &[f64],
    g: &[f64],
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    wt: &[f64],
    z: &[f64],
    r: &mut [f64],
    wa: &mut [f64],
    indx: &[i32],
    theta: f64,
    col: usize,
    head: usize,
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    let wy_ = |i: usize, j: usize| wy[(i - 1) + (j - 1) * n];
    let ws_ = |i: usize, j: usize| ws[(i - 1) + (j - 1) * n];
    if !cnstnd && col > 0 {
        for i in 1..=n {
            r[i - 1] = -g[i - 1];
        }
    } else {
        for i in 1..=nfree {
            let k = indx[i - 1] as usize;
            r[i - 1] = -theta * (z[k - 1] - x[k - 1]) - g[k - 1];
        }
        {
            let (p, rest) = wa.split_at_mut(2 * m);
            let v = &rest[..2 * m];
            if bmv(m, sy, wt, col, v, p) != 0 {
                return -8;
            }
        }
        let mut pointr = head;
        for j in 1..=col {
            let a1 = wa[j - 1];
            let a2 = theta * wa[col + j - 1];
            for i in 1..=nfree {
                let k = indx[i - 1] as usize;
                r[i - 1] += wy_(k, pointr) * a1 + ws_(k, pointr) * a2;
            }
            pointr = pointr % m + 1;
        }
    }
    0
}

/// R: `errclb()` -- check the validity of the input data.
#[allow(clippy::too_many_arguments)]
fn errclb(n: usize, m: usize, factr: f64, l: &[f64], u: &[f64], nbd: &[i32], task: &mut String, info: &mut i32) {
    if n == 0 {
        *task = "ERROR: N .LE. 0".to_string();
    }
    if m == 0 {
        *task = "ERROR: M .LE. 0".to_string();
    }
    if factr < 0.0 {
        *task = "ERROR: FACTR .LT. 0".to_string();
    }
    for i in 1..=n {
        if nbd[i - 1] < 0 || nbd[i - 1] > 3 {
            *task = "ERROR: INVALID NBD".to_string();
            *info = -6;
        }
        if nbd[i - 1] == 2 && l[i - 1] > u[i - 1] {
            *task = "ERROR: NO FEASIBLE SOLUTION".to_string();
            *info = -7;
        }
    }
}

/// R: `formk()` -- form the LEL^T factorization of the indefinite matrix
/// K = [-D -Y'ZZ'Y/theta  L_a'-R_z' ; L_a -R_z  theta*S'AA'S]. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn formk(
    n: usize,
    nsub: usize,
    ind: &[i32],
    nenter: usize,
    ileave: usize,
    indx2: &[i32],
    iupdat: usize,
    updatd: bool,
    wn: &mut [f64],
    wn1: &mut [f64],
    m: usize,
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
) -> i32 {
    let m2d = 2 * m;
    let w = |i: usize, j: usize| (i - 1) + (j - 1) * m2d; // wn / wn1 index
    let wy_ = |i: usize, j: usize| wy[(i - 1) + (j - 1) * n];
    let ws_ = |i: usize, j: usize| ws[(i - 1) + (j - 1) * n];
    let sy_ = |i: usize, j: usize| sy[(i - 1) + (j - 1) * m];

    // Form the lower triangular part of
    //   WN1 = [Y' ZZ'Y   L_a'+R_z'] [L_a+R_z   S'AA'S   ]
    // where L_a is the strictly lower triangular part of S'AA'Y
    //       R_z is the upper triangular part of S'ZZ'Y.
    let upcl;
    if updatd {
        if iupdat > m {
            // shift old part of WN1.
            for jy in 1..m {
                let js = m + jy;
                let len = m - jy;
                for k in 0..len {
                    wn1[w(jy, jy) + k] = wn1[w(jy + 1, jy + 1) + k];
                }
                for k in 0..len {
                    wn1[w(js, js) + k] = wn1[w(js + 1, js + 1) + k];
                }
                for k in 0..m - 1 {
                    wn1[w(m + 1, jy) + k] = wn1[w(m + 2, jy + 1) + k];
                }
            }
        }
        // put new rows in blocks (1,1), (2,1) and (2,2).
        let pbegin = 1;
        let pend = nsub;
        let dbegin = nsub + 1;
        let dend = n;
        let iy = col;
        let is = m + col;
        let mut ipntr = head + col - 1;
        if ipntr > m {
            ipntr -= m;
        }
        let mut jpntr = head;
        for jy in 1..=col {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            // compute element jy of row 'col' of Y'ZZ'Y
            for k in pbegin..=pend {
                let k1 = ind[k - 1] as usize;
                temp1 += wy_(k1, ipntr) * wy_(k1, jpntr);
            }
            // compute elements jy of row 'col' of L_a and S'AA'S
            for k in dbegin..=dend {
                let k1 = ind[k - 1] as usize;
                temp2 += ws_(k1, ipntr) * ws_(k1, jpntr);
                temp3 += ws_(k1, ipntr) * wy_(k1, jpntr);
            }
            wn1[w(iy, jy)] = temp1;
            wn1[w(is, js)] = temp2;
            wn1[w(is, jy)] = temp3;
            jpntr = jpntr % m + 1;
        }
        // put new column in block (2,1).
        let jy = col;
        let mut jpntr = head + col - 1;
        if jpntr > m {
            jpntr -= m;
        }
        let mut ipntr = head;
        for i in 1..=col {
            let is = m + i;
            let mut temp3 = 0.0;
            // compute element i of column 'col' of R_z
            for k in pbegin..=pend {
                let k1 = ind[k - 1] as usize;
                temp3 += ws_(k1, ipntr) * wy_(k1, jpntr);
            }
            ipntr = ipntr % m + 1;
            wn1[w(is, jy)] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }
    // modify the old parts in blocks (1,1) and (2,2) due to changes
    // in the set of free variables.
    let mut ipntr = head;
    for iy in 1..=upcl {
        let is = m + iy;
        let mut jpntr = head;
        for jy in 1..=iy {
            let js = m + jy;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            let mut temp3 = 0.0;
            let mut temp4 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k - 1] as usize;
                temp1 += wy_(k1, ipntr) * wy_(k1, jpntr);
                temp2 += ws_(k1, ipntr) * ws_(k1, jpntr);
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1] as usize;
                temp3 += wy_(k1, ipntr) * wy_(k1, jpntr);
                temp4 += ws_(k1, ipntr) * ws_(k1, jpntr);
            }
            wn1[w(iy, jy)] = wn1[w(iy, jy)] + temp1 - temp3;
            wn1[w(is, js)] = wn1[w(is, js)] - temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    // modify the old parts in block (2,1).
    let mut ipntr = head;
    for is in m + 1..=m + upcl {
        let mut jpntr = head;
        for jy in 1..=upcl {
            let mut temp1 = 0.0;
            let mut temp3 = 0.0;
            for k in 1..=nenter {
                let k1 = indx2[k - 1] as usize;
                temp1 += ws_(k1, ipntr) * wy_(k1, jpntr);
            }
            for k in ileave..=n {
                let k1 = indx2[k - 1] as usize;
                temp3 += ws_(k1, ipntr) * wy_(k1, jpntr);
            }
            if is <= jy + m {
                wn1[w(is, jy)] += temp1 - temp3;
            } else {
                wn1[w(is, jy)] += -temp1 + temp3;
            }
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
    //                                 [-L_a +R_z        S'AA'S*theta]
    for iy in 1..=col {
        let is = col + iy;
        let is1 = m + iy;
        for jy in 1..=iy {
            let js = col + jy;
            let js1 = m + jy;
            wn[w(jy, iy)] = wn1[w(iy, jy)] / theta;
            wn[w(js, is)] = wn1[w(is1, js1)] * theta;
        }
        for jy in 1..iy {
            wn[w(jy, is)] = -wn1[w(is1, jy)];
        }
        for jy in iy..=col {
            wn[w(jy, is)] = wn1[w(is1, jy)];
        }
        wn[w(iy, iy)] += sy_(iy, iy);
    }
    // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
    //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
    // first Cholesky factor (1,1) block of wn to get LL'
    // with L' stored in the upper triangle of wn.
    if dpofa(wn, 0, m2d, col) != 0 {
        return -1;
    }
    // then form L^-1(-L_a'+R_z') in the (1,2) block.
    let col2 = col << 1;
    let mut bcol = vec![0.0; col];
    for js in col + 1..=col2 {
        let start = w(1, js);
        bcol.copy_from_slice(&wn[start..start + col]);
        // dtrsl's info is not checked here in the C code
        let _ = dtrsl(wn, 0, m2d, col, &mut bcol, 11);
        wn[start..start + col].copy_from_slice(&bcol);
    }
    // Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the
    // upper triangle of (2,2) block of wn.
    for is in col + 1..=col2 {
        for js in is..=col2 {
            let a = w(1, is);
            let b = w(1, js);
            let dot = ddot(col, &wn[a..a + col], &wn[b..b + col]);
            wn[w(is, js)] += dot;
        }
    }
    // Cholesky factorization of (2,2) block of wn.
    if dpofa(wn, w(col + 1, col + 1), m2d, col) != 0 {
        return -2;
    }
    0
}

/// R: `formt()` -- form T = theta*SS + L*D^(-1)*L' in the upper triangle of
/// `wt` and Cholesky factorize it. Returns `info` (-3 on failure).
fn formt(m: usize, wt: &mut [f64], sy: &[f64], ss: &[f64], col: usize, theta: f64) -> i32 {
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * m;
    for j in 1..=col {
        wt[ix(1, j)] = theta * ss[ix(1, j)];
    }
    for i in 2..=col {
        for j in i..=col {
            let k1 = if i > j { j } else { i } - 1;
            let mut ddum = 0.0;
            for k in 1..=k1 {
                ddum += sy[ix(i, k)] * sy[ix(j, k)] / sy[ix(k, k)];
            }
            wt[ix(i, j)] = ddum + theta * ss[ix(i, j)];
        }
    }
    if dpofa(wt, 0, m, col) != 0 {
        return -3;
    }
    0
}

/// R: `freev()` -- count the entering and leaving variables and find the
/// index set of free and active variables at the GCP. Returns `wrk`.
#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    nfree: &mut usize,
    indx: &mut [i32],
    nenter: &mut usize,
    ileave: &mut usize,
    indx2: &mut [i32],
    iwhere: &[i32],
    updatd: bool,
    cnstnd: bool,
    iter: i32,
) -> bool {
    *nenter = 0;
    *ileave = n + 1;
    if iter > 0 && cnstnd {
        // count the entering and leaving variables.
        for i in 1..=*nfree {
            let k = indx[i - 1] as usize;
            if iwhere[k - 1] > 0 {
                *ileave -= 1;
                indx2[*ileave - 1] = k as i32;
            }
        }
        for i in *nfree + 1..=n {
            let k = indx[i - 1] as usize;
            if iwhere[k - 1] <= 0 {
                *nenter += 1;
                indx2[*nenter - 1] = k as i32;
            }
        }
    }
    let wrk = *ileave < n + 1 || *nenter > 0 || updatd;
    // Find the index set of free and active variables at the GCP.
    *nfree = 0;
    let mut iact = n + 1;
    for i in 1..=n {
        if iwhere[i - 1] <= 0 {
            *nfree += 1;
            indx[*nfree - 1] = i as i32;
        } else {
            iact -= 1;
            indx[iact - 1] = i as i32;
        }
    }
    wrk
}

/// R: `hpsolb()` -- sort out the least element of t, and put the remaining
/// elements of t in a heap (Algorithm 232 of CACM, J. W. J. Williams).
fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [i32], iheap: i32) {
    if iheap == 0 {
        // Rearrange the elements t(1) to t(n) to form a heap.
        for k in 2..=n {
            let ddum = t[k - 1];
            let indxin = iorder[k - 1];
            // Add ddum to the heap.
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j - 1] {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }
    // Assign to 'out' the value of t(1), the least member of the heap,
    // and rearrange the remaining members to form a heap as
    // elements 1 to n-1 of t.
    if n > 1 {
        let mut i = 1;
        let out = t[0];
        let indxou = iorder[0];
        let ddum = t[n - 1];
        let indxin = iorder[n - 1];
        // Restore the heap
        loop {
            let mut j = i + i;
            if j < n {
                if t[j] < t[j - 1] {
                    j += 1;
                }
                if t[j - 1] < ddum {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;
        // Put the least member in t(n).
        t[n - 1] = out;
        iorder[n - 1] = indxou;
    }
}

/// R: `matupd()` -- update matrices WS and WY, and form the middle matrix in B.
#[allow(clippy::too_many_arguments)]
fn matupd(
    n: usize,
    m: usize,
    ws: &mut [f64],
    wy: &mut [f64],
    sy: &mut [f64],
    ss: &mut [f64],
    d: &[f64],
    r: &[f64],
    itail: &mut usize,
    iupdat: usize,
    col: &mut usize,
    head: &mut usize,
    theta: &mut f64,
    rr: f64,
    dr: f64,
    stp: f64,
    dtd: f64,
) {
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * m;
    // Set pointers for matrices WS and WY.
    if iupdat <= m {
        *col = iupdat;
        *itail = (*head + iupdat - 2) % m + 1;
    } else {
        *itail = *itail % m + 1;
        *head = *head % m + 1;
    }
    // Update matrices WS and WY.
    let c = (*itail - 1) * n;
    ws[c..c + n].copy_from_slice(&d[..n]);
    wy[c..c + n].copy_from_slice(&r[..n]);
    // Set theta=yy/ys.
    *theta = rr / dr;
    // Form the middle matrix in B.
    // update the upper triangle of SS, and the lower triangle of SY:
    if iupdat > m {
        // move old information
        for j in 1..*col {
            for k in 0..j {
                ss[ix(1, j) + k] = ss[ix(2, j + 1) + k];
            }
            for k in 0..*col - j {
                sy[ix(j, j) + k] = sy[ix(j + 1, j + 1) + k];
            }
        }
    }
    // add new information: the last row of SY and the last column of SS:
    let mut pointr = *head;
    for j in 1..*col {
        let pc = (pointr - 1) * n;
        sy[ix(*col, j)] = ddot(n, d, &wy[pc..pc + n]);
        ss[ix(j, *col)] = ddot(n, &ws[pc..pc + n], d);
        pointr = pointr % m + 1;
    }
    if stp == 1.0 {
        ss[ix(*col, *col)] = dtd;
    } else {
        ss[ix(*col, *col)] = stp * stp * dtd;
    }
    sy[ix(*col, *col)] = dr;
}

/// R: `projgr()` -- infinity norm of the projected gradient.
fn projgr(n: usize, l: &[f64], u: &[f64], nbd: &[i32], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.0;
    for i in 0..n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    let d1 = x[i] - u[i];
                    if gi < d1 {
                        gi = d1;
                    }
                }
            } else if nbd[i] <= 2 {
                let d1 = x[i] - l[i];
                if gi > d1 {
                    gi = d1;
                }
            }
        }
        let d1 = gi.abs();
        if sbgnrm < d1 {
            sbgnrm = d1;
        }
    }
    sbgnrm
}

/// R: `subsm()` -- subspace minimization along the unconstrained Newton
/// direction, backtracked into the box. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn subsm(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &[i32],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],
    d: &mut [f64],
    ws: &[f64],
    wy: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    iword: &mut i32,
    wv: &mut [f64],
    wn: &[f64],
) -> i32 {
    let wy_ = |i: usize, j: usize| wy[(i - 1) + (j - 1) * n];
    let ws_ = |i: usize, j: usize| ws[(i - 1) + (j - 1) * n];
    let ns = nsub;
    if ns == 0 {
        return 0;
    }
    // Compute wv = W'Zd.
    let mut pointr = head;
    for i in 1..=col {
        let mut temp1 = 0.0;
        let mut temp2 = 0.0;
        for j in 1..=ns {
            let k = ind[j - 1] as usize;
            temp1 += wy_(k, pointr) * d[j - 1];
            temp2 += ws_(k, pointr) * d[j - 1];
        }
        wv[i - 1] = temp1;
        wv[col + i - 1] = theta * temp2;
        pointr = pointr % m + 1;
    }
    // Compute wv:=K^(-1)wv.
    let m2 = m << 1;
    let col2 = col << 1;
    if dtrsl(wn, 0, m2, col2, &mut wv[..col2], 11) != 0 {
        return 1;
    }
    for i in 1..=col {
        wv[i - 1] = -wv[i - 1];
    }
    if dtrsl(wn, 0, m2, col2, &mut wv[..col2], 1) != 0 {
        return 1;
    }
    // Compute d = (1/theta)d + (1/theta**2)Z'W wv.
    let mut pointr = head;
    for jy in 1..=col {
        let js = col + jy;
        for i in 1..=ns {
            let k = ind[i - 1] as usize;
            d[i - 1] += wy_(k, pointr) * wv[jy - 1] / theta + ws_(k, pointr) * wv[js - 1];
        }
        pointr = pointr % m + 1;
    }
    for i in 1..=ns {
        d[i - 1] /= theta;
    }
    // Backtrack to the feasible region.
    let mut alpha = 1.0;
    let mut temp1 = alpha;
    let mut ibd = 0usize;
    for i in 1..=ns {
        let k = ind[i - 1] as usize;
        let dk = d[i - 1];
        if nbd[k - 1] != 0 {
            if dk < 0.0 && nbd[k - 1] <= 2 {
                let temp2 = l[k - 1] - x[k - 1];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k - 1] >= 2 {
                let temp2 = u[k - 1] - x[k - 1];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1.0 {
        let dk = d[ibd - 1];
        let k = ind[ibd - 1] as usize;
        if dk > 0.0 {
            x[k - 1] = u[k - 1];
            d[ibd - 1] = 0.0;
        } else if dk < 0.0 {
            x[k - 1] = l[k - 1];
            d[ibd - 1] = 0.0;
        }
    }
    for i in 1..=ns {
        let k = ind[i - 1] as usize;
        x[k - 1] += alpha * d[i - 1];
    }
    *iword = if alpha < 1.0 { 1 } else { 0 };
    0
}

impl Dcsrch {
    /// R: `dcsrch()` (MINPACK-2) -- find a step satisfying a sufficient
    /// decrease condition and a curvature condition.
    #[allow(clippy::too_many_arguments)]
    fn dcsrch(
        &mut self,
        f: &f64,
        g: &mut f64,
        stp: &mut f64,
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
        task: &mut String,
    ) {
        let f = *f;
        // Initialization block.
        if task.starts_with("START") {
            // Check the input arguments for errors.
            if *stp < stpmin {
                *task = "ERROR: STP .LT. STPMIN".to_string();
            }
            if *stp > stpmax {
                *task = "ERROR: STP .GT. STPMAX".to_string();
            }
            if *g >= 0.0 {
                *task = "ERROR: INITIAL G .GE. ZERO".to_string();
            }
            if ftol < 0.0 {
                *task = "ERROR: FTOL .LT. ZERO".to_string();
            }
            if gtol < 0.0 {
                *task = "ERROR: GTOL .LT. ZERO".to_string();
            }
            if xtol < 0.0 {
                *task = "ERROR: XTOL .LT. ZERO".to_string();
            }
            if stpmin < 0.0 {
                *task = "ERROR: STPMIN .LT. ZERO".to_string();
            }
            if stpmax < stpmin {
                *task = "ERROR: STPMAX .LT. STPMIN".to_string();
            }
            // Exit if there are errors on input.
            if task.starts_with("ERROR") {
                return;
            }
            // Initialize local variables.
            self.brackt = false;
            self.stage = 1;
            self.finit = f;
            self.ginit = *g;
            self.gtest = ftol * self.ginit;
            self.width = stpmax - stpmin;
            self.width1 = self.width / 0.5;
            self.stx = 0.0;
            self.fx = self.finit;
            self.gx = self.ginit;
            self.sty = 0.0;
            self.fy = self.finit;
            self.gy = self.ginit;
            self.stmin = 0.0;
            self.stmax = *stp + *stp * 4.0;
            *task = "FG".to_string();
            return;
        }
        // If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
        // algorithm enters the second stage.
        let ftest = self.finit + *stp * self.gtest;
        if self.stage == 1 && f <= ftest && *g >= 0.0 {
            self.stage = 2;
        }
        // Test for warnings.
        if self.brackt && (*stp <= self.stmin || *stp >= self.stmax) {
            *task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS".to_string();
        }
        if self.brackt && self.stmax - self.stmin <= xtol * self.stmax {
            *task = "WARNING: XTOL TEST SATISFIED".to_string();
        }
        if *stp == stpmax && f <= ftest && *g <= self.gtest {
            *task = "WARNING: STP = STPMAX".to_string();
        }
        if *stp == stpmin && (f > ftest || *g >= self.gtest) {
            *task = "WARNING: STP = STPMIN".to_string();
        }
        // Test for convergence.
        if f <= ftest && g.abs() <= gtol * (-self.ginit) {
            *task = "CONVERGENCE".to_string();
        }
        // Test for termination.
        if task.starts_with("WARN") || task.starts_with("CONV") {
            return;
        }
        // A modified function is used to predict the step during the
        // first stage if a lower function value has been obtained but
        // the decrease is not sufficient.
        if self.stage == 1 && f <= self.fx && f > ftest {
            // Define the modified function and derivative values.
            let fm = f - *stp * self.gtest;
            let mut fxm = self.fx - self.stx * self.gtest;
            let mut fym = self.fy - self.sty * self.gtest;
            let gm = *g - self.gtest;
            let mut gxm = self.gx - self.gtest;
            let mut gym = self.gy - self.gtest;
            // Call dcstep to update stx, sty, and to compute the new step.
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                stp,
                fm,
                gm,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
            // Reset the function and derivative values for f.
            self.fx = fxm + self.stx * self.gtest;
            self.fy = fym + self.sty * self.gtest;
            self.gx = gxm + self.gtest;
            self.gy = gym + self.gtest;
        } else {
            // Call dcstep to update stx, sty, and to compute the new step.
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                stp,
                f,
                *g,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
        }
        // Decide if a bisection step is needed.
        if self.brackt {
            if (self.sty - self.stx).abs() >= self.width1 * 0.66 {
                *stp = self.stx + (self.sty - self.stx) * 0.5;
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        // Set the minimum and maximum steps allowed for stp.
        if self.brackt {
            self.stmin = cmin(self.stx, self.sty);
            self.stmax = cmax(self.stx, self.sty);
        } else {
            self.stmin = *stp + (*stp - self.stx) * 1.1;
            self.stmax = *stp + (*stp - self.stx) * 4.0;
        }
        // Force the step to be within the bounds stpmax and stpmin.
        if *stp < stpmin {
            *stp = stpmin;
        }
        if *stp > stpmax {
            *stp = stpmax;
        }
        // If further progress is not possible, let stp be the best
        // point obtained during the search (kept in the C form).
        #[allow(clippy::nonminimal_bool)]
        let no_progress = (self.brackt && (*stp <= self.stmin || *stp >= self.stmax))
            || (self.brackt && (self.stmax - self.stmin <= xtol * self.stmax));
        if no_progress {
            *stp = self.stx;
        }
        // Obtain another function and derivative.
        *task = "FG".to_string();
    }
}

/// R: `dcstep()` (MINPACK-2) -- safeguarded step for the line search and
/// update of the interval containing a step that satisfies the conditions.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        // First case: A higher function value. The minimum is bracketed.
        // If the cubic step is closer to stx than the quadratic step, the
        // cubic step is taken, otherwise the average of the cubic and
        // quadratic steps is taken.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let d1 = cmax(theta.abs(), dx.abs());
        let s = cmax(d1, dp.abs());
        let d1 = theta / s;
        let mut gamm = s * (d1 * d1 - *dx / s * (dp / s)).sqrt();
        if *stp < *stx {
            gamm = -gamm;
        }
        let p = gamm - *dx + theta;
        let q = gamm - *dx + gamm + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + *dx / ((*fx - fp) / (*stp - *stx) + *dx) / 2.0 * (*stp - *stx);
        if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        *brackt = true;
    } else if sgnd < 0.0 {
        // Second case: A lower function value and derivatives of opposite
        // sign. The minimum is bracketed. If the cubic step is farther from
        // stp than the secant step, the cubic step is taken, otherwise the
        // secant step is taken.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let d1 = cmax(theta.abs(), dx.abs());
        let s = cmax(d1, dp.abs());
        let d1 = theta / s;
        let mut gamm = s * (d1 * d1 - *dx / s * (dp / s)).sqrt();
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm - dp + gamm + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        // Third case: A lower function value, derivatives of the same sign,
        // and the magnitude of the derivative decreases.
        // The cubic step is computed only if the cubic tends to infinity
        // in the direction of the step or if the minimum of the cubic
        // is beyond stp. Otherwise the cubic step is defined to be the
        // secant step.
        let theta = (*fx - fp) * 3.0 / (*stp - *stx) + *dx + dp;
        let d1 = cmax(theta.abs(), dx.abs());
        let s = cmax(d1, dp.abs());
        // The case gamm = 0 only arises if the cubic does not tend
        // to infinity in the direction of the step.
        let d1 = theta / s;
        let d1 = d1 * d1 - *dx / s * (dp / s);
        let mut gamm = if d1 < 0.0 { 0.0 } else { s * d1.sqrt() };
        if *stp > *stx {
            gamm = -gamm;
        }
        let p = gamm - dp + theta;
        let q = gamm + (*dx - dp) + gamm;
        let r = p / q;
        let stpc = if r < 0.0 && gamm != 0.0 {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + dp / (dp - *dx) * (*stx - *stp);
        if *brackt {
            // A minimizer has been bracketed. If the cubic step is
            // closer to stp than the secant step, the cubic step is
            // taken, otherwise the secant step is taken.
            let mut sf = if (stpc - *stp).abs() < (stpq - *stp).abs() { stpc } else { stpq };
            let d1 = *stp + (*sty - *stp) * 0.66;
            if *stp > *stx {
                sf = cmin(d1, sf);
            } else {
                sf = cmax(d1, sf);
            }
            stpf = sf;
        } else {
            // A minimizer has not been bracketed. If the cubic step is
            // farther from stp than the secant step, the cubic step is
            // taken, otherwise the secant step is taken.
            let mut sf = if (stpc - *stp).abs() > (stpq - *stp).abs() { stpc } else { stpq };
            sf = cmin(stpmax, sf);
            sf = cmax(stpmin, sf);
            stpf = sf;
        }
    } else {
        // Fourth case: A lower function value, derivatives of the
        // same sign, and the magnitude of the derivative does not
        // decrease. If the minimum is not bracketed, the step is either
        // stpmin or stpmax, otherwise the cubic step is taken.
        if *brackt {
            let theta = (fp - *fy) * 3.0 / (*sty - *stp) + *dy + dp;
            let d1 = cmax(theta.abs(), dy.abs());
            let s = cmax(d1, dp.abs());
            let d1 = theta / s;
            let mut gamm = s * (d1 * d1 - *dy / s * (dp / s)).sqrt();
            if *stp > *sty {
                gamm = -gamm;
            }
            let p = gamm - dp + theta;
            let q = gamm - dp + gamm + *dy;
            let r = p / q;
            let stpc = *stp + r * (*sty - *stp);
            stpf = stpc;
        } else if *stp > *stx {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    // Update the interval which contains a minimizer.
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    // Compute the new step.
    *stp = stpf;
}

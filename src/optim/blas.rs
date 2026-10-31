//! The BLAS level-1 and LINPACK routines R's `lbfgsb.c` calls
//! (`ddot`, `daxpy`, `dpofa`, `dtrsl`), reproducing the arithmetic of the
//! libraries R 4.5.3 is linked against in the reference environment.
//!
//! * BLAS is OpenBLAS 0.3.34 (DYNAMIC_ARCH, Haswell kernels on the reference
//!   machine). Its `daxpy` computes `y[i] = fma(a, x[i], y[i])`; its `ddot`
//!   (unit strides) sums blocks of four products left to right without FMA and
//!   the remaining `n mod 4` products with FMA, which was verified bit for bit
//!   against the library for n < 16. For n >= 16 OpenBLAS uses a vectorised
//!   kernel with several partial sums that is not reproduced (the tail model is
//!   applied to the whole vector); L-BFGS-B only calls `ddot` with the number
//!   of parameters and with `col`, `2*col` <= 10, so DESeq2 designs with fewer
//!   than 16 coefficients are exact.
//! * LINPACK `dpofa` is R's modified copy (src/appl/dpofa.f), which declares a
//!   matrix singular when `s <= 1e-14 * |a(j,j)|` instead of `s <= 0`.
//!
//! Matrices are column-major with leading dimension `lda`; `off` is the
//! 0-based position of element (1,1), so that a sub-block such as
//! `&wn[col+1 + (col+1)*wn_dim1]` in the C code is `(wn, off)`.

/// BLAS `ddot` with unit strides (OpenBLAS summation order, see module docs).
pub(crate) fn ddot(n: usize, x: &[f64], y: &[f64]) -> f64 {
    let mut dot = 0.0f64;
    let mut i = 0;
    while i + 4 <= n {
        let p0 = y[i] * x[i];
        let p1 = y[i + 1] * x[i + 1];
        let p2 = y[i + 2] * x[i + 2];
        let p3 = y[i + 3] * x[i + 3];
        dot += p0;
        dot += p1;
        dot += p2;
        dot += p3;
        i += 4;
    }
    while i < n {
        dot = y[i].mul_add(x[i], dot);
        i += 1;
    }
    dot
}

/// BLAS `daxpy` with unit strides: `y := a*x + y` (fused, as OpenBLAS).
pub(crate) fn daxpy(n: usize, a: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        y[i] = a.mul_add(x[i], y[i]);
    }
}

/// BLAS `dscal` with unit stride: `x := a*x`.
pub(crate) fn dscal(n: usize, a: f64, x: &mut [f64]) {
    for v in x.iter_mut().take(n) {
        *v *= a;
    }
}

/// LINPACK `dpofa` as modified in R (src/appl/dpofa.f): Cholesky factor of
/// the symmetric positive definite matrix whose upper triangle is stored in
/// `a`, overwriting it with R where A = R'R. Returns `info` (0 = success,
/// otherwise the order of the leading minor that is not positive definite).
pub(crate) fn dpofa(a: &mut [f64], off: usize, lda: usize, n: usize) -> usize {
    let idx = |i: usize, j: usize| off + (i - 1) + (j - 1) * lda;
    const EPS: f64 = 1e-14;
    for j in 1..=n {
        let mut s = 0.0f64;
        for k in 1..j {
            let dot = {
                let ck = idx(1, k);
                let cj = idx(1, j);
                ddot(k - 1, &a[ck..ck + (k - 1)], &a[cj..cj + (k - 1)])
            };
            let mut t = a[idx(k, j)] - dot;
            t /= a[idx(k, k)];
            a[idx(k, j)] = t;
            s += t * t;
        }
        s = a[idx(j, j)] - s;
        if s <= EPS * a[idx(j, j)].abs() {
            return j;
        }
        a[idx(j, j)] = s.sqrt();
    }
    0
}

/// LINPACK `dtrsl`: solve `T x = b` or `T' x = b` for triangular `T`
/// (stored at `t[off..]`, leading dimension `ldt`), overwriting `b`.
/// `job`: 00 = T lower, 01 = T upper, 10 = T' with T lower, 11 = T' with T
/// upper. Returns `info` (0, or the index of a zero diagonal element).
pub(crate) fn dtrsl(t: &[f64], off: usize, ldt: usize, n: usize, b: &mut [f64], job: i32) -> usize {
    let idx = |i: usize, j: usize| off + (i - 1) + (j - 1) * ldt;
    for info in 1..=n {
        if t[idx(info, info)] == 0.0 {
            return info;
        }
    }
    let mut case = 1;
    if job % 10 != 0 {
        case = 2;
    }
    if (job % 100) / 10 != 0 {
        case += 2;
    }
    // b is 1-based in the Fortran; b[j-1] here
    match case {
        1 => {
            // T x = b, T lower triangular
            b[0] /= t[idx(1, 1)];
            for j in 2..=n {
                let temp = -b[j - 2];
                let c = idx(j, j - 1);
                daxpy(n - j + 1, temp, &t[c..c + (n - j + 1)], &mut b[j - 1..n]);
                b[j - 1] /= t[idx(j, j)];
            }
        }
        2 => {
            // T x = b, T upper triangular
            b[n - 1] /= t[idx(n, n)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let temp = -b[j];
                let c = idx(1, j + 1);
                daxpy(j, temp, &t[c..c + j], &mut b[0..j]);
                b[j - 1] /= t[idx(j, j)];
            }
        }
        3 => {
            // T' x = b, T lower triangular
            b[n - 1] /= t[idx(n, n)];
            for jj in 2..=n {
                let j = n - jj + 1;
                let c = idx(j + 1, j);
                let d = ddot(jj - 1, &t[c..c + (jj - 1)], &b[j..j + (jj - 1)]);
                b[j - 1] -= d;
                b[j - 1] /= t[idx(j, j)];
            }
        }
        _ => {
            // T' x = b, T upper triangular
            b[0] /= t[idx(1, 1)];
            for j in 2..=n {
                let c = idx(1, j);
                let d = ddot(j - 1, &t[c..c + (j - 1)], &b[0..j - 1]);
                b[j - 1] -= d;
                b[j - 1] /= t[idx(j, j)];
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpofa_uses_r_tolerance() {
        // R: .C(dpofa) on matrix(c(1,1,1,1+e),2) gives info=2 for e=1e-14
        // and info=0 for e=2e-14.
        let mut a = vec![1.0, 1.0, 1.0, 1.0 + 1e-14];
        assert_eq!(dpofa(&mut a, 0, 2, 2), 2);
        let mut a = vec![1.0, 1.0, 1.0, 1.0 + 2e-14];
        assert_eq!(dpofa(&mut a, 0, 2, 2), 0);
    }

    #[test]
    fn dtrsl_solves_both_ways() {
        // upper triangular R = [[2, 1], [0, 3]] column-major
        let r = vec![2.0, 0.0, 1.0, 3.0];
        let mut b = vec![4.0, 6.0];
        dtrsl(&r, 0, 2, 2, &mut b, 1); // R x = b
        assert_eq!(b, vec![1.0, 2.0]);
        let mut b = vec![2.0, 7.0];
        dtrsl(&r, 0, 2, 2, &mut b, 11); // R' x = b
        assert_eq!(b, vec![1.0, 2.0]);
    }
}

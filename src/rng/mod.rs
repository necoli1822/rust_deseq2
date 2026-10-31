//! R-compatible Random Number Generator
//!
//! This module implements R's Mersenne Twister RNG with R's specific seeding
//! algorithm to produce identical random sequences as R's set.seed().

/// R's Mersenne Twister RNG
///
/// This implementation exactly matches R's Mersenne-Twister RNG as defined in RNG.c.
/// R uses a 624-element state array with Matsumoto and Nishimura's algorithm.
pub struct RMersenneTwister {
    state: [u32; 624],
    index: usize,
}

impl RMersenneTwister {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908B0DF;
    const UPPER_MASK: u32 = 0x80000000;
    const LOWER_MASK: u32 = 0x7FFFFFFF;

    /// Create a new RNG with the same seed as R's set.seed()
    ///
    /// R's set.seed() uses a Linear Congruential Generator (LCG) with
    /// multiplier 69069 to initialize the Mersenne-Twister state.
    /// This is NOT the standard MT init_genrand!
    pub fn new(seed: u32) -> Self {
        let mut mt = RMersenneTwister {
            state: [0; Self::N],
            index: Self::N,
        };
        mt.r_init_seed(seed);
        mt
    }

    /// Initialize MT state using R's exact algorithm from RNG.c
    ///
    /// R's set.seed() uses an LCG: x_{n+1} = 69069 * x_n + 1 (mod 2^32)
    /// to generate the initial MT state values.
    ///
    /// From R's RNG.c Setseed() function:
    /// - Run LCG 50 times to "warm up"
    /// - Discard one more LCG value
    /// - Then use next 624 LCG values to initialize MT state
    fn r_init_seed(&mut self, seed: u32) {
        let mut r_i1: u32 = seed;

        // Warm up the LCG with 50 iterations (R does this)
        for _ in 0..50 {
            r_i1 = r_i1.wrapping_mul(69069).wrapping_add(1);
        }

        // R discards one more value before using for MT state
        r_i1 = r_i1.wrapping_mul(69069).wrapping_add(1);

        // Initialize MT state with next 624 LCG values
        for i in 0..Self::N {
            r_i1 = r_i1.wrapping_mul(69069).wrapping_add(1);
            self.state[i] = r_i1;
        }

        self.index = Self::N;
    }

    /// Standard MT initialization (for reference, not used by R's set.seed)
    #[allow(dead_code)]
    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..Self::N {
            let prev = self.state[i - 1];
            self.state[i] = (1812433253_u32)
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = Self::N;
    }

    /// Generate the next 624 words of the state array
    fn generate_numbers(&mut self) {
        for i in 0..Self::N {
            let y = (self.state[i] & Self::UPPER_MASK)
                  | (self.state[(i + 1) % Self::N] & Self::LOWER_MASK);
            self.state[i] = self.state[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                self.state[i] ^= Self::MATRIX_A;
            }
        }
        self.index = 0;
    }

    /// Generate a random 32-bit integer
    fn next_u32(&mut self) -> u32 {
        if self.index >= Self::N {
            self.generate_numbers();
        }

        let mut y = self.state[self.index];
        self.index += 1;

        // Tempering transformation
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;

        y
    }

    /// Generate uniform random number in [0, 1)
    ///
    /// R uses: (double)u * 2.3283064365386963e-10 which is 1/(2^32)
    /// With correction to avoid 0 and 1
    pub fn runif(&mut self) -> f64 {
        // R's formula from RNG.c: return 2.3283064365386963e-10 * (double) I
        // where I is the 32-bit random integer
        // But R actually uses: fixup(2.3283064365386963e-10 * (double) genrand_int32())
        // The fixup ensures values are in (0, 1) not [0, 1)
        let u = self.next_u32();
        let result = u as f64 * 2.3283064365386963e-10;

        // R's fixup(): keep strictly inside (0, 1), using i2_32m1 = 1/(2^32 - 1)
        const I2_32M1: f64 = 2.328306437080797e-10;
        if result <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if 1.0 - result <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        result
    }

    /// Standard normal deviate by R's default "Inversion" method (norm_rand in snorm.c)
    ///
    /// A single uniform is not precise enough, so R combines two:
    ///   u = unif_rand(); u = (int)(BIG * u) + unif_rand(); qnorm(u / BIG), BIG = 2^27
    pub fn rnorm(&mut self) -> f64 {
        const BIG: f64 = 134217728.0;
        let u = self.runif();
        let u = (BIG * u).trunc() + self.runif();
        qnorm(u / BIG)
    }

    /// Generate chi-squared random number with given degrees of freedom
    /// Chi-squared(df) = Gamma(df/2, 2)
    pub fn rchisq(&mut self, df: f64) -> f64 {
        self.rgamma(df / 2.0, 2.0)
    }

    /// Standard exponential deviate exactly as R's exp_rand() (sexp.c):
    /// Ahrens & Dieter (1972), algorithm SA, not -log(U)
    pub fn rexp(&mut self) -> f64 {
        // q[k-1] = sum(log(2)^k / k!), k = 1..16, copied verbatim from sexp.c
        // (q[0] is ln 2; kept as R's literal so the table reads as R's)
        #[allow(clippy::approx_constant)]
        const Q: [f64; 16] = [
            0.6931471805599453,
            0.9333736875190459,
            0.9888777961838675,
            0.9984959252914960040,
            0.9998292811061389,
            0.9999833164100727,
            0.9999985508193775,
            0.9999998906925558,
            0.9999999924734159,
            0.9999999995283275,
            0.9999999999728814,
            0.9999999999985598,
            0.9999999999999289,
            0.9999999999999968,
            0.9999999999999999,
            1.0000000000000000,
        ];
        let mut a = 0.0;
        let mut u = self.runif();
        while u <= 0.0 || u >= 1.0 {
            u = self.runif();
        }
        loop {
            u += u;
            if u > 1.0 {
                break;
            }
            a += Q[0];
        }
        u -= 1.0;
        if u <= Q[0] {
            return a + u;
        }
        let mut i = 0;
        let mut ustar = self.runif();
        let mut umin = ustar;
        loop {
            ustar = self.runif();
            if umin > ustar {
                umin = ustar;
            }
            i += 1;
            if u <= Q[i] {
                break;
            }
        }
        a + umin * Q[0]
    }

    /// Generate gamma random number using R's exact algorithm from rgamma.c
    ///
    /// R uses:
    /// - GS algorithm (Ahrens-Dieter 1974) for shape < 1
    /// - GD algorithm (Ahrens-Dieter 1982) for shape >= 1
    pub fn rgamma(&mut self, a: f64, scale: f64) -> f64 {
        // R's rgamma.c uses this truncated value, which enters the squeeze test
        const SQRT32: f64 = 5.656854;
        const EXP_M1: f64 = 0.36787944117144232; // exp(-1)

        // Coefficients for GD algorithm
        const Q1: f64 = 0.04166669;
        const Q2: f64 = 0.02083148;
        const Q3: f64 = 0.00801191;
        const Q4: f64 = 0.00144121;
        const Q5: f64 = -7.388e-5;
        const Q6: f64 = 2.4511e-4;
        const Q7: f64 = 2.424e-4;

        const A1: f64 = 0.3333333;
        const A2: f64 = -0.250003;
        const A3: f64 = 0.2000062;
        const A4: f64 = -0.1662921;
        const A5: f64 = 0.1423657;
        const A6: f64 = -0.1367177;
        const A7: f64 = 0.1233795;

        if a < 1.0 {
            // GS algorithm for shape < 1
            let e = 1.0 + EXP_M1 * a;
            loop {
                let p = e * self.runif();
                let x;
                if p >= 1.0 {
                    x = -((e - p) / a).ln();
                    if self.rexp() >= (1.0 - a) * x.ln() {
                        return scale * x;
                    }
                } else {
                    x = (p.ln() / a).exp();
                    if self.rexp() >= x {
                        return scale * x;
                    }
                }
            }
        }

        // GD algorithm for shape >= 1
        let s2 = a - 0.5;
        let s = s2.sqrt();
        let d = SQRT32 - s * 12.0;

        // Step 2: t = standard normal, x = (s, 1/2)-normal
        let t = self.rnorm();
        let x = s + 0.5 * t;
        let ret_val = x * x;

        // Immediate acceptance
        if t >= 0.0 {
            return scale * ret_val;
        }

        // Step 3: squeeze acceptance
        let u = self.runif();
        if d * u <= t * t * t {
            return scale * ret_val;
        }

        // Step 4: calculate q0, b, si, c
        let r = 1.0 / a;
        let q0 = ((((((Q7 * r + Q6) * r + Q5) * r + Q4) * r + Q3) * r + Q2) * r + Q1) * r;

        let (b, si, c) = if a <= 3.686 {
            (0.463 + s + 0.178 * s2, 1.235, 0.195 / s - 0.079 + 0.16 * s)
        } else if a <= 13.022 {
            (1.654 + 0.0076 * s2, 1.68 / s + 0.275, 0.062 / s + 0.024)
        } else {
            (1.77, 0.75, 0.1515 / s)
        };

        // Step 5-7: quotient test if x > 0
        if x > 0.0 {
            let v = t / (s + s);
            let q = if v.abs() <= 0.25 {
                q0 + 0.5 * t * t * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
            } else {
                q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln()
            };

            if (1.0 - u).ln() <= q {
                return scale * ret_val;
            }
        }

        // Step 8-11: rejection loop
        loop {
            let e = self.rexp();
            let mut u = self.runif();
            u = u + u - 1.0;
            let t = if u < 0.0 { b - si * e } else { b + si * e };

            if t >= -0.71874483771719 {
                let v = t / (s + s);
                let q = if v.abs() <= 0.25 {
                    q0 + 0.5 * t * t * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
                } else {
                    q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln()
                };

                if q > 0.0 {
                    let w = q.exp_m1();
                    if c * u.abs() <= w * (e - 0.5 * t * t).exp() {
                        let x = s + 0.5 * t;
                        return scale * x * x;
                    }
                }
            }
        }
    }
}

/// Inverse normal CDF (quantile function), lower tail, non-log p
///
/// A literal translation of R's qnorm5() (src/nmath/qnorm.c, R 4.5), Wichura's
/// AS 241 with R's coefficients and branch structure: norm_rand() feeds every
/// uniform through it, so any difference shows up in the simulated draws.
fn qnorm(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // |p - 0.5| <= .425  <==> 0.075 <= p <= 0.925
        let r = 0.180625 - q * q;
        return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
            + 45921.953931549871457)
            * r
            + 13731.693765509461125)
            * r
            + 1971.5909503065514427)
            * r
            + 133.14166789178437745)
            * r
            + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }

    // closer than 0.075 from {0, 1}: lp = log(min(p, 1 - p)), with 1 - p as R_DT_CIv(p)
    let lp = if q > 0.0 { (0.5 - p + 0.5).ln() } else { p.ln() };
    let mut r = (-lp).sqrt();
    let val = if r <= 5.0 {
        r += -1.6;
        (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else if r <= 27.0 {
        r += -5.0;
        (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    } else {
        // r > 27 needs min(p, 1 - p) < exp(-729), below the smallest uniform R can return
        let s2 = -2.0 * lp;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut x2 = s2 - (two_pi * s2).ln();
        if r < 36000.0 {
            x2 = s2 - (two_pi * x2).ln() - 2.0 / (2.0 + x2);
        }
        x2.sqrt()
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from R 4.5 (set.seed(2), default RNG kinds), printed with %.17g
    fn assert_close(got: f64, want: f64) {
        assert!((got - want).abs() <= 1e-15 * want.abs().max(1.0), "got {:.17}, want {:.17}", got, want);
    }

    #[test]
    fn test_r_rng_runif() {
        let mut rng = RMersenneTwister::new(2);
        for want in [0.18488225992769003, 0.70237403595820069, 0.57332633482292295] {
            assert_close(rng.runif(), want);
        }
    }

    #[test]
    fn test_r_rng_rnorm() {
        let mut rng = RMersenneTwister::new(2);
        for want in [-0.89691454662498138, 0.18484918464674249, 1.5878453312088232] {
            assert_close(rng.rnorm(), want);
        }
    }

    #[test]
    fn test_r_rng_rexp() {
        let mut rng = RMersenneTwister::new(2);
        for want in [1.8653524405414108, 0.40474807191640139, 0.14665266964584589] {
            assert_close(rng.rexp(), want);
        }
    }

    #[test]
    fn test_r_rng_rchisq() {
        let cases: [(f64, [f64; 4]); 3] = [
            (1.0, [0.095825183658292332, 0.92149495036514861, 0.85630277508525954, 0.5546726566079011]),
            (2.0, [0.13379913584698364, 0.10229738282570668, 0.040281973368143688, 1.1855441251226946]),
            (3.0, [0.60839875872378502, 0.53883407524968197, 0.37812323397130138, 2.2591307791793858]),
        ];
        for (df, wants) in cases {
            let mut rng = RMersenneTwister::new(2);
            for want in wants {
                assert_close(rng.rchisq(df), want);
            }
        }
    }

    #[test]
    fn test_r_rng_long_sequence() {
        // set.seed(2); a <- rchisq(1e4, 2); b <- rnorm(1e4, 0, 0.5); c(a[1e4], b[1e4])
        let mut rng = RMersenneTwister::new(2);
        let a: Vec<f64> = (0..10_000).map(|_| rng.rchisq(2.0)).collect();
        let b: Vec<f64> = (0..10_000).map(|_| 0.5 * rng.rnorm()).collect();
        assert_close(a[9_999], 1.830613728945319);
        assert_close(b[9_999], -0.67501397818342734);
    }
}

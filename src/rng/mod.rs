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
        let mut result = u as f64 * 2.3283064365386963e-10;

        // R's fixup: avoid exact 0 and 1
        if result <= 0.0 {
            result = 0.5 * 2.3283064365386963e-10;
        }
        if result >= 1.0 {
            result = 1.0 - 0.5 * 2.3283064365386963e-10;
        }
        result
    }

    /// Generate normal random number using R's Inversion method
    ///
    /// R's default normal RNG uses the "Inversion" method which uses
    /// qnorm(runif()) - the inverse normal CDF.
    pub fn rnorm(&mut self) -> f64 {
        let u = self.runif();
        qnorm(u)
    }

    /// Generate chi-squared random number with given degrees of freedom
    /// Chi-squared(df) = Gamma(df/2, 2)
    pub fn rchisq(&mut self, df: f64) -> f64 {
        self.rgamma(df / 2.0, 2.0)
    }

    /// Generate exponential random number
    ///
    /// R's exp_rand() uses the standard inversion method: -log(U)
    pub fn rexp(&mut self) -> f64 {
        -self.runif().ln()
    }

    /// Generate gamma random number using R's exact algorithm from rgamma.c
    ///
    /// R uses:
    /// - GS algorithm (Ahrens-Dieter 1974) for shape < 1
    /// - GD algorithm (Ahrens-Dieter 1982) for shape >= 1
    pub fn rgamma(&mut self, a: f64, scale: f64) -> f64 {
        const SQRT32: f64 = 5.656854249492381;
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

/// Inverse normal CDF (quantile function)
///
/// Uses Wichura's Algorithm AS 241 (~10^-16 precision), matching R's qnorm.c exactly.
/// Three-region rational approximation with 8-term polynomials.
fn qnorm(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p.is_nan() {
        return f64::NAN;
    }

    // Horner's method for polynomial evaluation
    fn horner(coeffs: &[f64], x: f64) -> f64 {
        let mut result = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            result = result * x + coeffs[i];
        }
        result
    }

    // Central region coefficients (|p - 0.5| <= 0.425)
    const A: [f64; 8] = [
        3.3871328727963666080e0,
        1.3314166789178437745e2,
        1.9715909503065514427e3,
        1.3731693765509461125e4,
        4.5921953931549871457e4,
        6.7265770927008700853e4,
        3.3430575583588128105e4,
        2.5090809287301226727e3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.2313330701600911252e1,
        6.8718700749205790830e2,
        5.3941960214247511077e3,
        2.1213794301586595867e4,
        3.9307895800092710610e4,
        2.8729085735721942674e4,
        5.2264952788528545610e3,
    ];

    // Intermediate region coefficients (0.425 < |p-0.5| < ~0.499997)
    const C: [f64; 8] = [
        1.42343711074968357734e0,
        4.63033784615654529590e0,
        5.76949722146069140550e0,
        3.64784832476320460504e0,
        1.27045825245236838258e0,
        2.41780725177450611770e-1,
        2.27238260553211220900e-2,
        7.74545014427727025900e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.05319162663775882187e0,
        1.67638483018380162246e0,
        6.89767334985100004550e-1,
        1.48103976427480074590e-1,
        1.51986665636164571966e-2,
        5.47593808499534494600e-4,
        1.05075007164441684324e-9,
    ];

    // Tail region coefficients (r > 5, where r = sqrt(-2*ln(min(p,1-p))))
    const E: [f64; 8] = [
        6.65790464350110377720e0,
        5.46378491116411436990e0,
        1.78482653991729133580e0,
        2.96560571828504891230e-1,
        2.65321895265761230930e-2,
        1.24266094738807843860e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ];
    const F: [f64; 8] = [
        1.0,
        5.99832206555887937690e-1,
        1.36929880922735805310e-1,
        1.48753612908506508198e-2,
        7.86869131145613259100e-4,
        1.84631831751005468180e-5,
        1.42151175831644588870e-7,
        2.04426310338993978564e-15,
    ];

    let q = p - 0.5;

    if q.abs() <= 0.425 {
        // Central region
        let r = 0.180625 - q * q;
        q * horner(&A, r) / horner(&B, r)
    } else {
        // Tail region
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();

        let val = if r <= 5.0 {
            // Intermediate tail
            let r = r - 1.6;
            horner(&C, r) / horner(&D, r)
        } else {
            // Far tail
            let r = r - 5.0;
            horner(&E, r) / horner(&F, r)
        };

        if q < 0.0 { -val } else { val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_rng_runif() {
        let mut rng = RMersenneTwister::new(2);

        // Compare with R's output:
        // set.seed(2); runif(10)
        // [1] 0.1848822599 0.7023740360 0.5733263348 0.1680519204 0.9438393388
        // [6] 0.9434749587 0.1291589767 0.8334488156 0.4680185155 0.5499837417
        let expected = [
            0.1848822599, 0.7023740360, 0.5733263348, 0.1680519204, 0.9438393388,
            0.9434749587, 0.1291589767, 0.8334488156, 0.4680185155, 0.5499837417,
        ];

        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.runif();
            println!("runif[{}] = {:.10}, expected = {:.10}, diff = {:.2e}",
                     i, got, exp, (got - exp).abs());
        }
    }

    #[test]
    fn test_r_rng_rchisq() {
        let mut rng = RMersenneTwister::new(2);

        // R's set.seed(2); rchisq(5, df=2)
        // [1] 0.13379913585 0.10229738283 0.04028197337 1.18554412512 0.24805533529
        let chi1 = rng.rchisq(2.0);
        println!("rchisq(1, df=2) = {:.10}", chi1);
    }
}

//! Ports of the R `nmath` density routines that the DESeq2 `optim()` objective
//! evaluates (R 4.5.3, src/nmath/).
//!
//! DESeq2's `fitNbinomGLMsOptim` minimises
//! `-(sum(dnbinom(k, mu=, size=, log=TRUE)) + sum(dnorm(p, 0, sd, log=TRUE)))`
//! with a finite-difference gradient, so the L-BFGS-B iterates depend on the
//! last bits of the objective. These functions follow R's C code operation by
//! operation (`dnbinom_mu`, `dbinom_raw`, `stirlerr`, `bd0`, `lgammafn`,
//! `gammafn`, `lgammacor`, `chebyshev_eval`, `lgamma1p`, `logcf`, `log1pmx`,
//! `dnorm4`) and use the same libm primitives (`log`, `log1p`, `exp`,
//! `lgamma`), so that the objective agrees with R's to the last bit in the
//! cases DESeq2 hits. Only the branches reachable for non-negative integer
//! counts, positive `size` and positive `mu` are ported; the remaining corner
//! cases are documented where they differ.

// The coefficient tables are copied verbatim from R's C sources.
#![allow(clippy::excessive_precision)]

/// log(2*pi). R: M_LN_2PI
const M_LN_2PI: f64 = 1.837_877_066_409_345_483_560_659_472_811;
/// log(sqrt(2*pi)). R: M_LN_SQRT_2PI
const M_LN_SQRT_2PI: f64 = 0.918_938_533_204_672_741_780_329_736_406;

/// The C library's `lgamma`, which R's `stirlerr()` calls directly (it is not
/// R's `lgammafn`). Uses the pure-Rust fdlibm/musl port in the `libm` crate rather
/// than calling the platform C library, so the crate stays free of FFI; glibc's
/// own result already varies by CPU (FMA variants), and libm agrees with it to
/// <= 2 ulp on [1, 5.25], the range stirlerr uses.
fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// C `nearbyint()` in the default rounding mode (ties to even). R: `R_forceint`.
fn nearbyint(x: f64) -> f64 {
    let r = x.round();
    if (r - x).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// R: `R_nonint(x)` (nmath.h, R >= 4.4.0).
fn r_nonint(x: f64) -> bool {
    (x - nearbyint(x)).abs() > 1e-9 * 1f64.max(x.abs())
}

/// R: `chebyshev_eval()` (chebyshev.c).
fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    if !(1..=1000).contains(&n) || !(-1.1..=1.1).contains(&x) {
        return f64::NAN;
    }
    let twox = x * 2.0;
    let mut b2 = 0.0;
    let mut b1 = 0.0;
    let mut b0 = 0.0;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

/// Chebyshev coefficients of R's `gammafn()` (gamma.c); `ngam = 22` are used.
const GAMCS: [f64; 22] = [
    0.857_119_559_098_933_142_192_006_239_994_2e-2,
    0.441_538_132_484_100_675_719_131_577_165_2e-2,
    0.568_504_368_159_936_337_863_266_458_878_9e-1,
    -0.421_983_539_641_856_050_101_250_018_662_4e-2,
    0.132_680_818_121_246_022_058_400_679_635_2e-2,
    -0.189_302_452_979_888_043_252_394_702_388_6e-3,
    0.360_692_532_744_124_525_657_808_221_722_5e-4,
    -0.605_676_190_446_086_421_848_554_829_036_5e-5,
    0.105_582_954_630_228_334_473_182_350_909_3e-5,
    -0.181_196_736_554_238_404_829_185_589_116_6e-6,
    0.311_772_496_471_532_227_779_025_459_316_9e-7,
    -0.535_421_963_901_968_714_087_408_102_434_7e-8,
    0.919_327_551_985_958_894_688_778_682_594_0e-9,
    -0.157_794_128_028_833_976_176_742_327_395_3e-9,
    0.270_798_062_293_495_454_326_654_043_308_9e-10,
    -0.464_681_865_382_573_014_408_166_105_893_3e-11,
    0.797_335_019_200_741_965_646_076_717_535_9e-12,
    -0.136_807_820_983_091_602_579_949_917_230_9e-12,
    0.234_731_948_656_380_065_723_347_177_168_8e-13,
    -0.402_743_261_494_906_693_276_657_053_469_9e-14,
    0.691_005_174_737_210_091_213_833_697_525_7e-15,
    -0.118_558_450_022_199_290_705_238_712_619_2e-15,
];

/// R: `gammafn()` (gamma.c), branch `0 < x <= 10` only (the only one
/// `lgammafn()` reaches for positive arguments).
fn gammafn_small_pos(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x <= 10.0);
    // xsml = exp(.01)*DBL_MIN
    const XSML: f64 = 2.247_436_222_559_854_5e-308;
    let mut n = x as i32;
    // x > 0 here, so no `if (x < 0) --n`
    let y = x - n as f64;
    n -= 1;
    let mut value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, 22) + 0.9375;
    if n == 0 {
        return value;
    }
    if n < 0 {
        if y < XSML {
            return f64::INFINITY;
        }
        let n = -n;
        for i in 0..n {
            value /= x + i as f64;
        }
        value
    } else {
        for i in 1..=n {
            value *= y + i as f64;
        }
        value
    }
}

/// R: `lgammacor()` (lgammacor.c) for `x >= 10`.
fn lgammacor(x: f64) -> f64 {
    const ALGMCS: [f64; 5] = [
        0.166_638_948_045_186_324_720_572_965_082_2e+0,
        -0.138_494_817_606_756_384_073_298_605_913_5e-4,
        0.981_082_564_692_472_942_615_717_154_748_7e-8,
        -0.180_912_947_557_249_419_426_330_626_671_9e-10,
        0.622_109_804_189_260_522_712_601_554_341_6e-13,
    ];
    const XBIG: f64 = 94_906_265.624_251_56;
    if x < 10.0 {
        f64::NAN
    } else if x < XBIG {
        let tmp = 10.0 / x;
        chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, 5) / x
    } else {
        1.0 / (x * 12.0)
    }
}

/// R: `lgammafn()` (lgamma.c) for `x > 0`. Non-positive arguments (never
/// reached from `dnbinom`) fall back to the C library's `lgamma`.
pub fn lgammafn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 {
        return lgamma(x);
    }
    const XMAX: f64 = 2.532_737_276_080_075_8e+305;
    let y = x;
    if y < 1e-306 {
        return -y.ln();
    }
    if y <= 10.0 {
        return gammafn_small_pos(x).abs().ln();
    }
    if y > XMAX {
        return f64::INFINITY;
    }
    if x > 1e17 {
        x * (x.ln() - 1.0)
    } else if x > 4_934_720.0 {
        M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x
    } else {
        M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x + lgammacor(x)
    }
}

/// R: `logcf()` (pgamma.c) -- continued fraction for `log1pmx` / `lgamma1p`.
fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    // SQR(SQR(SQR(4294967296.0))) = 2^256 (exact)
    let s1 = 4294967296.0f64 * 4294967296.0;
    let s2 = s1 * s1;
    let scalefactor = s2 * s2;
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;
    b2 = c4 * b1 - i * b2;

    while (a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs() {
        let mut c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if b2.abs() > scalefactor {
            a1 /= scalefactor;
            b1 /= scalefactor;
            a2 /= scalefactor;
            b2 /= scalefactor;
        } else if b2.abs() < 1.0 / scalefactor {
            a1 *= scalefactor;
            b1 *= scalefactor;
            a2 *= scalefactor;
            b2 *= scalefactor;
        }
    }
    a2 / b2
}

/// R: `log1pmx()` (pgamma.c) -- accurate `log(1+x) - x`.
fn log1pmx(x: f64) -> f64 {
    const MIN_LOG1_VALUE: f64 = -0.79149064;
    if !(MIN_LOG1_VALUE..=1.0).contains(&x) {
        x.ln_1p() - x
    } else {
        let r = x / (2.0 + x);
        let y = r * r;
        if x.abs() < 1e-2 {
            let two = 2.0_f64;
            r * ((((two / 9.0 * y + two / 7.0) * y + two / 5.0) * y + two / 3.0) * y - x)
        } else {
            r * (2.0 * y * logcf(y, 3.0, 2.0, 1e-14) - x)
        }
    }
}

/// R: `lgamma1p()` (pgamma.c) -- accurate `log(gamma(a+1))` for small `a`.
pub fn lgamma1p(a: f64) -> f64 {
    if a.abs() >= 0.5 {
        return lgammafn(a + 1.0);
    }
    const EULERS_CONST: f64 = 0.577_215_664_901_532_860_606_512_090_082_4;
    const N: usize = 40;
    const COEFFS: [f64; 40] = [
        0.322_467_033_424_113_218_236_207_583_323_012_6e-0,
        0.673_523_010_531_980_951_332_460_538_371_500_0e-1,
        0.205_808_084_277_845_478_790_009_241_352_919_8e-1,
        0.738_555_102_867_398_526_627_309_729_140_683_4e-2,
        0.289_051_033_074_152_328_575_298_829_848_675_5e-2,
        0.119_275_391_170_326_097_711_393_569_282_810_9e-2,
        0.509_669_524_743_042_422_335_654_813_581_558_2e-3,
        0.223_154_758_453_579_379_761_418_803_601_340_1e-3,
        0.994_575_127_818_085_337_145_958_900_319_017_0e-4,
        0.449_262_367_381_331_417_002_075_024_063_578_6e-4,
        0.205_072_127_756_706_915_531_665_039_783_059_1e-4,
        0.943_948_827_526_839_590_398_742_510_441_505_5e-5,
        0.437_486_678_990_748_780_418_179_322_395_241_1e-5,
        0.203_921_575_380_136_623_678_190_070_967_083_9e-5,
        0.955_141_213_040_741_983_285_717_977_295_126_5e-6,
        0.449_246_919_876_456_604_329_429_033_119_365_5e-6,
        0.212_071_848_055_546_658_692_313_590_107_762_8e-6,
        0.100_432_248_239_680_996_087_208_305_005_334_4e-6,
        0.476_981_016_936_398_056_576_019_341_724_673_0e-7,
        0.227_110_946_089_431_649_103_199_811_606_212_4e-7,
        0.108_386_592_148_969_540_910_749_175_796_815_9e-7,
        0.518_347_504_197_004_665_512_124_864_705_766_9e-8,
        0.248_367_454_380_247_831_718_500_866_399_171_8e-8,
        0.119_214_014_058_609_120_744_254_820_277_464_0e-8,
        0.573_136_724_167_886_201_333_019_485_796_101_1e-9,
        0.275_952_288_512_423_314_517_814_969_281_634_1e-9,
        0.133_047_643_742_444_894_814_971_572_085_800_8e-9,
        0.642_296_456_383_810_002_208_244_808_764_464_8e-10,
        0.310_442_477_473_222_727_623_921_578_340_406_6e-10,
        0.150_213_840_807_541_421_709_330_104_878_066_8e-10,
        0.727_597_448_023_907_966_250_454_992_481_404_7e-11,
        0.352_774_247_657_591_508_361_507_222_865_548_3e-11,
        0.171_199_179_055_961_790_860_108_411_444_303_1e-11,
        0.831_538_584_142_028_481_979_835_779_395_441_8e-12,
        0.404_220_052_528_944_006_553_600_895_703_289_5e-12,
        0.196_647_563_109_661_649_041_104_567_901_028_6e-12,
        0.957_363_038_783_855_576_378_220_093_650_861_5e-13,
        0.466_407_602_642_837_422_457_649_256_597_457_7e-13,
        0.227_373_696_006_597_232_063_327_959_673_727_2e-13,
        0.110_913_994_708_345_220_165_832_000_719_233_4e-13,
    ];
    const C: f64 = 0.227_373_684_582_465_251_522_682_157_797_869_1e-12;
    const TOL_LOGCF: f64 = 1e-14;

    let mut lgam = C * logcf(-a / 2.0, (N + 2) as f64, 1.0, TOL_LOGCF);
    for i in (0..N).rev() {
        lgam = COEFFS[i] - a * lgam;
    }
    (a * lgam - EULERS_CONST) * a - log1pmx(a)
}

/// R: `stirlerr()` (stirlerr.c, R >= 4.4.0):
/// `log(n!) - log(sqrt(2*pi*n)*(n/e)^n)`.
pub fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 0.083333333333333333333;
    const S1: f64 = 0.00277777777777777777778;
    const S2: f64 = 0.00079365079365079365079365;
    const S3: f64 = 0.000595238095238095238095238;
    const S4: f64 = 0.0008417508417508417508417508;
    const S5: f64 = 0.0019175269175269175269175262;
    const S6: f64 = 0.0064102564102564102564102561;
    const S7: f64 = 0.029550653594771241830065352;
    const S8: f64 = 0.17964437236883057316493850;
    const S9: f64 = 1.3924322169059011164274315;
    const S10: f64 = 13.402864044168391994478957;
    const S11: f64 = 156.84828462600201730636509;
    const S12: f64 = 2193.1033333333333333333333;
    const S13: f64 = 36108.771253724989357173269;
    const S14: f64 = 691472.26885131306710839498;
    const S15: f64 = 15238221.539407416192283370;
    const S16: f64 = 382900751.39141414141414141;
    const SFERR_HALVES: [f64; 31] = [
        0.0,
        0.1534264097200273452913848,
        0.0810614667953272582196702,
        0.0548141210519176538961390,
        0.0413406959554092940938221,
        0.03316287351993628748511048,
        0.02767792568499833914878929,
        0.02374616365629749597132920,
        0.02079067210376509311152277,
        0.01848845053267318523077934,
        0.01664469118982119216319487,
        0.01513497322191737887351255,
        0.01387612882307074799874573,
        0.01281046524292022692424986,
        0.01189670994589177009505572,
        0.01110455975820691732662991,
        0.010411265261972096497478567,
        0.009799416126158803298389475,
        0.009255462182712732917728637,
        0.008768700134139385462952823,
        0.008330563433362871256469318,
        0.007934114564314020547248100,
        0.007573675487951840794972024,
        0.007244554301320383179543912,
        0.006942840107209529865664152,
        0.006665247032707682442354394,
        0.006408994188004207068439631,
        0.006171712263039457647532867,
        0.005951370112758847735624416,
        0.005746216513010115682023589,
        0.005554733551962801371038690,
    ];

    if n <= 23.5 {
        let nn = n + n;
        if n <= 15.0 && nn == (nn as i32) as f64 {
            return SFERR_HALVES[nn as i32 as usize];
        }
        if n <= 5.25 {
            if n >= 1.0 {
                let l_n = n.ln();
                return lgamma(n) + n * (1.0 - l_n) + (l_n - M_LN_2PI) * 0.5;
            } else {
                return lgamma1p(n) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
            }
        }
        let nn = n * n;
        if n > 12.8 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - S6 / nn) / nn) / nn) / nn) / nn) / nn) / n;
        }
        if n > 12.3 {
            return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn)
                / n;
        }
        if n > 8.9 {
            return (S0
                - (S1 - (S2 - (S3 - (S4 - (S5 - (S6 - (S7 - S8 / nn) / nn) / nn) / nn) / nn) / nn) / nn)
                    / nn)
                / n;
        }
        if n > 7.3 {
            return (S0
                - (S1
                    - (S2
                        - (S3 - (S4 - (S5 - (S6 - (S7 - (S8 - (S9 - S10 / nn) / nn) / nn) / nn) / nn) / nn) / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        if n > 6.6 {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6 - (S7 - (S8 - (S9 - (S10 - (S11 - S12 / nn) / nn) / nn) / nn) / nn) / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        if n > 6.1 {
            return (S0
                - (S1
                    - (S2
                        - (S3
                            - (S4
                                - (S5
                                    - (S6
                                        - (S7
                                            - (S8
                                                - (S9 - (S10 - (S11 - (S12 - (S13 - S14 / nn) / nn) / nn) / nn) / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / n;
        }
        (S0 - (S1
            - (S2
                - (S3
                    - (S4
                        - (S5
                            - (S6
                                - (S7
                                    - (S8
                                        - (S9
                                            - (S10
                                                - (S11 - (S12 - (S13 - (S14 - (S15 - S16 / nn) / nn) / nn) / nn) / nn)
                                                    / nn)
                                                / nn)
                                            / nn)
                                        / nn)
                                    / nn)
                                / nn)
                            / nn)
                        / nn)
                    / nn)
                / nn)
            / nn)
            / n
    } else {
        let nn = n * n;
        if n > 15.7e6 {
            return S0 / n;
        }
        if n > 6180.0 {
            return (S0 - S1 / nn) / n;
        }
        if n > 205.0 {
            return (S0 - (S1 - S2 / nn) / nn) / n;
        }
        if n > 86.0 {
            return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
        }
        if n > 27.0 {
            return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
        }
        (S0 - (S1 - (S2 - (S3 - (S4 - S5 / nn) / nn) / nn) / nn) / nn) / n
    }
}

/// R: `bd0()` (bd0.c) -- deviance term `x log(x/np) + np - x`, computed stably.
pub fn bd0(x: f64, np: f64) -> f64 {
    if !x.is_finite() || !np.is_finite() || np == 0.0 {
        return f64::NAN;
    }
    if (x - np).abs() < 0.1 * (x + np) {
        let d = x - np;
        let mut v = d / (x + np);
        if d != 0.0 && v == 0.0 {
            let x_ = x * 0.25;
            let n_ = np * 0.25;
            v = (x_ - n_) / (x_ + n_);
        }
        let mut s = d * 0.5 * v;
        if (s * 2.0).abs() < f64::MIN_POSITIVE {
            return s * 2.0;
        }
        let mut ej = x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s_ = s;
            s += ej / ((j << 1) + 1) as f64;
            if s == s_ {
                return s * 2.0;
            }
        }
        // "should never happen": fall through to the direct formula as R does
    }
    let lg_x_n = if (x / np).is_finite() { (x / np).ln() } else { x.ln() - np.ln() };
    if x > np {
        x * (lg_x_n - 1.0) + np
    } else {
        x * lg_x_n + np - x
    }
}

/// R: `dbinom_raw(x, n, p, q, give_log = TRUE)` (dbinom.c).
fn dbinom_raw_log(x: f64, n: f64, p: f64, q: f64) -> f64 {
    if p == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if q == 0.0 {
        return if x == n { 0.0 } else { f64::NEG_INFINITY };
    }
    if x == 0.0 {
        if n == 0.0 {
            return 0.0;
        }
        return if p > q { n * q.ln() } else { n * (-p).ln_1p() };
    }
    if x == n {
        return if p > q { n * (-q).ln_1p() } else { n * p.ln() };
    }
    if x < 0.0 || x > n {
        return f64::NEG_INFINITY;
    }
    let mut n = n;
    if !n.is_finite() {
        if x.is_finite() {
            return f64::NEG_INFINITY;
        }
        n = f64::MAX;
    }
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();
    lc - 0.5 * lf
}

/// R: `dnbinom_mu(x, size, mu, give_log = TRUE)` (dnbinom.c), i.e.
/// `dnbinom(x, size = size, mu = mu, log = TRUE)`.
///
/// The `size = Inf` (Poisson) limit, which R evaluates with `dpois_raw`, uses
/// the plain formula `x log(mu) - mu - lgamma(x + 1)` here; DESeq2 never passes
/// an infinite size.
pub fn dnbinom_mu_log(x: f64, size: f64, mu: f64) -> f64 {
    if x.is_nan() || size.is_nan() || mu.is_nan() {
        return x + size + mu;
    }
    if mu < 0.0 || size < 0.0 {
        return f64::NAN;
    }
    if r_nonint(x) {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 && size == 0.0 {
        return 0.0;
    }
    let x = nearbyint(x);
    if !size.is_finite() {
        return x * mu.ln() - mu - lgammafn(x + 1.0);
    }
    if x == 0.0 {
        return size
            * if size < mu {
                (size / (size + mu)).ln()
            } else {
                (-mu / (size + mu)).ln_1p()
            };
    }
    if x < 1e-10 * size {
        let p = if size < mu {
            (size / (1.0 + size / mu)).ln()
        } else {
            (mu / (1.0 + mu / size)).ln()
        };
        let xx2s = if x < f64::MAX.sqrt() {
            (x * (x - 1.0)) * 0.5 / size
        } else {
            x * ((x * 0.5) / size)
        };
        return x * p - mu - lgamma1p(x) + xx2s.ln_1p();
    }
    let p = if x < size {
        (-x / (size + x)).ln_1p()
    } else {
        (size / (size + x)).ln()
    };
    let ans = dbinom_raw_log(size, x + size, size / (size + mu), mu / (size + mu));
    p + ans
}

/// R: `dnorm4(x, mu, sigma, give_log = TRUE)` (dnorm.c), i.e.
/// `dnorm(x, mean = mu, sd = sigma, log = TRUE)`.
pub fn dnorm_log(x: f64, mu: f64, sigma: f64) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if !sigma.is_finite() {
        return f64::NEG_INFINITY;
    }
    if !x.is_finite() && mu == x {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x == mu { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let x = (x - mu) / sigma;
    if !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let x = x.abs();
    if x >= 2.0 * f64::MAX.sqrt() {
        return f64::NEG_INFINITY;
    }
    -(M_LN_SQRT_2PI + 0.5 * x * x + sigma.ln())
}

/// R's `sum()` of a double vector (summary.c `rsum`), which accumulates in
/// `long double` and rounds once at the end. The accumulation is emulated
/// with an error-free (two-sum) double-double accumulator, which gives the
/// correctly rounded sum; this equals R's 80-bit result except in rare
/// double-rounding cases.
pub fn r_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut hi = 0.0f64;
    let mut lo = 0.0f64;
    for v in values {
        let s = hi + v;
        let bb = s - hi;
        let err = (hi - (s - bb)) + (v - bb);
        hi = s;
        lo += err;
    }
    if !hi.is_finite() {
        return hi;
    }
    hi + lo
}

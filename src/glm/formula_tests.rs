//! Tests of the formula parser and model matrices against R 4.5.3 / DESeq2 1.50.2

use super::*;

/// The colData of the R check (validation notes): batch b1/b2 alternating,
/// condition A/B/C in blocks of 4, genotype WT/KO (WT reference), numeric x and z
fn metadata() -> SampleMetadata {
    let ids: Vec<String> = (1..=12).map(|i| format!("s{}", i)).collect();
    let mut m = SampleMetadata::new(ids);
    m.add_condition("batch", (0..12).map(|i| if i % 2 == 0 { "b1" } else { "b2" }.to_string()).collect()).unwrap();
    m.add_condition("condition", (0..12).map(|i| ["A", "B", "C"][i / 4].to_string()).collect()).unwrap();
    m.add_condition("genotype", (0..12).map(|i| if i < 6 { "WT" } else { "KO" }.to_string()).collect()).unwrap();
    let x = [-0.626, 0.184, -0.836, 1.595, 0.33, -0.82, 0.487, 0.738, 0.576, -0.305, 1.512, 0.39];
    let z = [-0.621, -2.215, 1.125, -0.045, -0.016, 0.944, 0.821, 0.594, 0.919, 0.782, 0.075, -1.989];
    for (name, v) in [("x", x), ("z", z)] {
        m.add_condition(name, v.iter().map(|a| a.to_string()).collect()).unwrap();
        m.add_continuous(name, v.to_vec()).unwrap();
    }
    m
}

/// resultsNames, rank and sum(mm * row index) as printed by R 4.5.3 / DESeq2 1.50.2
/// for model.matrix(f, colData) with the renaming of fitNbinomGLMs()
fn check(f: &str, expected_names: &str, rank: usize, weighted_sum: f64) {
    let meta = metadata();
    let refs = HashMap::from([("genotype".to_string(), "WT".to_string())]);
    let formula = Formula::parse(f).unwrap();
    let frame = ModelFrame::new(&formula, &meta, &refs).unwrap().evaluate(&formula).unwrap();
    let ModelMatrix { matrix: mm, column_names, .. } = model_matrix(&formula, &frame).unwrap();
    let names = deseq_coefficient_names(&formula, &column_names, &frame);
    assert_eq!(names.join(","), expected_names, "names of {}", f);
    assert_eq!(super::super::design::qr_rank(&mm), rank, "rank of {}", f);
    let s: f64 = (0..mm.nrows()).map(|i| (i + 1) as f64 * mm.row(i).sum()).sum();
    assert!((s - weighted_sum).abs() < 1e-9, "sum of {}: {} vs {}", f, s, weighted_sum);
}

#[test]
fn test_model_matrix_matches_r() {
    check("~ batch + condition", "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A", 4, 188.0);
    let inter = "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,batchb2.conditionB,batchb2.conditionC";
    check("~ batch + condition + batch:condition", inter, 6, 224.0);
    check("~ batch * condition", inter, 6, 224.0);
    check(
        "~ condition:batch + batch + condition",
        "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,conditionB.batchb2,conditionC.batchb2",
        6,
        224.0,
    );
    let nested = "Intercept,condition_B_vs_A,condition_C_vs_A,conditionA.batchb2,conditionB.batchb2,conditionC.batchb2";
    check("~ condition + batch:condition", nested, 6, 188.0);
    check("~ condition / batch", nested, 6, 188.0);
    check(
        "~ batch:condition",
        "Intercept,batchb1.conditionA,batchb2.conditionA,batchb1.conditionB,batchb2.conditionB,batchb1.conditionC,batchb2.conditionC",
        6,
        156.0,
    );
    for f in ["~ 0 + condition", "~ condition - 1", "~ condition + 0", "~ -1 + condition"] {
        check(f, "conditionA,conditionB,conditionC", 3, 78.0);
    }
    check("~ 0 + batch + condition", "batchb1,batchb2,conditionB,conditionC", 4, 146.0);
    check(
        "~ 0 + batch:condition",
        "batchb1.conditionA,batchb2.conditionA,batchb1.conditionB,batchb2.conditionB,batchb1.conditionC,batchb2.conditionC",
        6,
        78.0,
    );
    check("~ 0 + x + condition", "x,conditionA,conditionB,conditionC", 4, 111.103);
    check("~ x + condition + x:condition", "Intercept,x,condition_B_vs_A,condition_C_vs_A,x.conditionB,x.conditionC", 6, 208.592);
    check("~ condition + condition:x", "Intercept,condition_B_vs_A,condition_C_vs_A,conditionA.x,conditionB.x,conditionC.x", 6, 179.103);
    check("~ x * z", "Intercept,x,z,x.z", 4, 110.795787);
    check("~ x:z", "Intercept,x.z", 2, 70.417787);
    check(
        "~ batch * condition * genotype",
        "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,genotype_KO_vs_WT,batchb2.conditionB,batchb2.conditionC,\
         batchb2.genotypeKO,conditionB.genotypeKO,conditionC.genotypeKO,batchb2.conditionB.genotypeKO,batchb2.conditionC.genotypeKO",
        8,
        398.0,
    );
    check(
        "~ (batch + condition + genotype)^2",
        "Intercept,batch_b2_vs_b1,condition_B_vs_A,condition_C_vs_A,genotype_KO_vs_WT,batchb2.conditionB,batchb2.conditionC,\
         batchb2.genotypeKO,conditionB.genotypeKO,conditionC.genotypeKO",
        8,
        368.0,
    );
    check("~ batch * condition - batch", "Intercept,condition_B_vs_A,condition_C_vs_A,batchb2.conditionA,batchb2.conditionB,batchb2.conditionC", 6, 188.0);
    check("~ 1", "Intercept", 1, 78.0);
    check("~ 0 + condition:x", "conditionA.x,conditionB.x,conditionC.x", 3, 33.103);
    check(
        "~ genotype + condition %in% genotype",
        "Intercept,genotype_KO_vs_WT,genotypeWT.conditionB,genotypeKO.conditionB,genotypeWT.conditionC,genotypeKO.conditionC",
        4,
        203.0,
    );
    check("~ x + batch:x", "Intercept,x,x.batchb2", 3, 120.465);
    check(
        "~ 0 + condition + condition:batch",
        "conditionA,conditionB,conditionC,conditionA.batchb2,conditionB.batchb2,conditionC.batchb2",
        6,
        120.0,
    );
}

#[test]
fn test_formula_terms() {
    let f = Formula::parse("~ batch * condition * genotype").unwrap();
    assert_eq!(f.variables, vec!["batch", "condition", "genotype"]);
    assert_eq!(
        f.term_labels(),
        vec!["batch", "condition", "genotype", "batch:condition", "batch:genotype", "condition:genotype", "batch:condition:genotype"]
    );
    assert!(f.intercept && f.has_interactions());
    let f = Formula::parse("condition:batch + batch").unwrap();
    assert_eq!(f.term_labels(), vec!["batch", "condition:batch"]);
    assert!(!Formula::parse("~ 0 + a").unwrap().intercept);
    assert!(Formula::parse("~ a - 1 + 1").unwrap().intercept);
    assert!(Formula::parse("~ x[1]").is_err());
    assert!(Formula::parse("y ~ x").is_err());
    assert!(Formula::parse("~ a +").is_err());
    assert!(Formula::parse("~ (a + b").is_err());
    assert_eq!(Formula::parse("~ `my var` + b").unwrap().variables, vec!["`my var`", "b"]);
    assert_eq!(Formula::parse("~ `my var` + b").unwrap().all_vars, vec!["my var", "b"]);
}

#[test]
fn test_make_names() {
    let cases = [
        ("", "X"), ("1a", "X1a"), (".1a", "X.1a"), ("a b", "a.b"), ("if", "if."), ("a:b", "a.b"), ("_x", "X_x"),
        (".x", ".x"), ("TRUE", "TRUE."), ("a-b", "a.b"), ("é", "é"),
    ];
    for (input, expected) in cases {
        assert_eq!(make_names(input), expected, "make.names({:?})", input);
    }
}

/// R cleanContrast(): column choice for character contrasts with and without intercept
#[test]
fn test_contrast_vectors() {
    use super::super::design::contrast_vector;
    let meta = metadata();
    let refs = HashMap::new();
    let design = |f: &str| formula_design(&Formula::parse(f).unwrap(), &meta, &refs).unwrap().1;
    // ~ 0 + batch + condition: batchb1, batchb2, conditionB, conditionC
    let info = design("~ 0 + batch + condition");
    assert_eq!(contrast_vector(&info, "condition", "C", "A").unwrap(), vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(contrast_vector(&info, "condition", "A", "C").unwrap(), vec![0.0, 0.0, 0.0, -1.0]);
    assert_eq!(contrast_vector(&info, "condition", "C", "B").unwrap(), vec![0.0, 0.0, -1.0, 1.0]);
    assert_eq!(contrast_vector(&info, "batch", "b2", "b1").unwrap(), vec![-1.0, 1.0, 0.0, 0.0]);
    // ~ 0 + condition: the first variable uses both columns
    let info = design("~ 0 + condition");
    assert_eq!(contrast_vector(&info, "condition", "B", "A").unwrap(), vec![-1.0, 1.0, 0.0]);
    // with an intercept: the renamed main-effect columns
    let info = design("~ batch * condition");
    assert_eq!(contrast_vector(&info, "condition", "C", "B").unwrap(), vec![0.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
    assert!(contrast_vector(&info, "condition", "D", "A").is_err());
    assert!(contrast_vector(&info, "genotype", "KO", "WT").is_err());
}

#[test]
fn test_full_rank_and_reduced() {
    let meta = metadata();
    let refs = HashMap::new();
    let err = formula_design(&Formula::parse("~ batch:condition").unwrap(), &meta, &refs).unwrap_err();
    assert!(format!("{}", err).contains("not full rank"));
    let (mm, info) = formula_design(&Formula::parse("~ batch * condition").unwrap(), &meta, &refs).unwrap();
    assert_eq!(mm.ncols(), 6);
    assert_eq!(info.factor_columns["batch"], vec![1]);
    assert_eq!(info.factor_columns["condition"], vec![2, 3]);
    assert!(info.interaction_present && info.has_intercept);
    let full_vars = info.design_vars.clone();
    let red = reduced_design(&Formula::parse("~ batch + condition").unwrap(), &full_vars, 6, &meta, &refs).unwrap();
    assert_eq!(red.ncols(), 4);
    assert!(reduced_design(&Formula::parse("~ genotype").unwrap(), &full_vars, 6, &meta, &refs).is_err());
    assert!(reduced_design(&Formula::parse("~ batch * condition").unwrap(), &full_vars, 6, &meta, &refs).is_err());
}

// ---- R reference model matrices of formula functions ----

#[path = "formula_r_cases.rs"]
mod r_cases;
use r_cases::{R_CASES, R_ERRORS};

/// The colData of gen_cases.R: cond A/B/C in blocks of 4, batch b1/b2 alternating, x and
/// age numeric (--continuous), dose 1/2/3 numeric, geno with levels needing
/// make.names(), flag logical, `my var` a factor with a non-syntactic name
fn metadata_r() -> SampleMetadata {
    let ids: Vec<String> = (1..=12).map(|i| format!("s{}", i)).collect();
    let mut m = SampleMetadata::new(ids);
    let s = |v: Vec<&str>| v.into_iter().map(String::from).collect::<Vec<String>>();
    m.add_condition("cond", (0..12).map(|i| ["A", "B", "C"][i / 4].to_string()).collect()).unwrap();
    m.add_condition("batch", (0..12).map(|i| ["b1", "b2"][i % 2].to_string()).collect()).unwrap();
    m.add_condition("geno", (0..12).map(|i| ["wild type", "KO-2", "1x"][i % 3].to_string()).collect()).unwrap();
    m.add_condition("flag", (0..12).map(|i| if i < 6 { "TRUE" } else { "FALSE" }.to_string()).collect()).unwrap();
    m.add_condition("my var", s(vec!["p"; 6]).into_iter().chain(s(vec!["q"; 6])).collect()).unwrap();
    let x = [-1.104, -0.105, -1.362, 1.636, 0.075, -1.343, 0.27, 0.579, 0.379, -0.708, 1.533, 0.15];
    let age = [30.7, 35.4, 20.5, 35.3, 54.8, 33.6, 39.3, 44.0, 39.7, 27.4, 53.1, 46.7];
    let dose: Vec<f64> = (0..12).map(|i| (i % 3 + 1) as f64).collect();
    for (name, v) in [("x", x.to_vec()), ("age", age.to_vec()), ("dose", dose)] {
        m.add_condition(name, v.iter().map(|a| a.to_string()).collect()).unwrap();
        m.add_continuous(name, v).unwrap();
    }
    m
}

/// colnames(model.matrix(f, colData)), DESeq2's resultsNames and every value of the
/// model matrix, against R for factor(), relevel(), ordered(), as.numeric(), log(),
/// scale(), I(), poly(), ns(), bs(), offset(), logical and non-syntactic columns and
/// their interactions; values to 1e-14 relative
#[test]
fn test_r_model_matrices_of_functions() {
    let meta = metadata_r();
    let refs = HashMap::new();
    let mut worst = (0.0f64, "");
    for (f, raw, names, values) in R_CASES {
        let formula = Formula::parse(f).unwrap_or_else(|e| panic!("{}: {}", f, e));
        let frame = ModelFrame::new(&formula, &meta, &refs)
            .and_then(|fr| fr.evaluate(&formula))
            .unwrap_or_else(|e| panic!("{}: {}", f, e));
        let mm = model_matrix(&formula, &frame).unwrap();
        assert_eq!(mm.column_names.join("|"), *raw, "colnames of {}", f);
        assert_eq!(deseq_coefficient_names(&formula, &mm.column_names, &frame).join(","), *names, "resultsNames of {}", f);
        let (n, p) = mm.matrix.dim();
        assert_eq!(values.len(), n * p, "size of {}", f);
        for j in 0..p {
            for i in 0..n {
                let r = values[j * n + i];
                let d = (mm.matrix[[i, j]] - r).abs() / r.abs().max(1.0);
                if d > worst.0 {
                    worst = (d, f);
                }
                assert!(d <= 1e-14, "{} [{}, {}]: {} vs R {}", f, i, j, mm.matrix[[i, j]], r);
            }
        }
    }
    eprintln!("largest relative difference to R: {:e} ({})", worst.0, worst.1);
}

/// Formulas R cannot turn into a model matrix are errors here too
#[test]
fn test_r_formula_errors() {
    let meta = metadata_r();
    let refs = HashMap::new();
    for (f, r_message) in R_ERRORS {
        let result = Formula::parse(f).and_then(|formula| {
            ModelFrame::new(&formula, &meta, &refs).and_then(|fr| fr.evaluate(&formula)).map(|_| ())
        });
        let err = result.expect_err(f);
        eprintln!("{} -> {} (R: {})", f, err, r_message);
    }
}

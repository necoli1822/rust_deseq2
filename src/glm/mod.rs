//! Generalized Linear Model fitting for negative binomial data

mod design;
mod fitting;
pub mod formula;
mod negative_binomial;

pub use design::{check_full_rank, contrast_vector, create_design_matrix, create_extended_design_matrix, get_contrast_index, DesignInfo};
pub use formula::{formula_design, make_names, reduced_design, Formula};
pub use fitting::{fit_glm, refit_glm_with_prior, refit_glm_with_prior_counts, fit_single_gene, fit_single_gene_weighted, GlmFitParams, GlmFitResult};
pub(crate) use fitting::gene_weights;
pub use negative_binomial::{nb_log_likelihood, nb_mean, nb_variance, nb_weight, MAX_LFC_BETA, MIN_MU};

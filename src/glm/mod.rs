//! Generalized Linear Model fitting for negative binomial data

mod design;
mod fitting;
mod negative_binomial;

pub use design::{check_full_rank, create_design_matrix, create_extended_design_matrix, get_contrast_index, DesignInfo};
pub use fitting::{fit_glm, refit_glm_with_prior, fit_single_gene, GlmFitParams, GlmFitResult};
pub use negative_binomial::{nb_log_likelihood, nb_mean, nb_variance, nb_weight, MAX_LFC_BETA, MIN_MU};

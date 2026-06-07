using Test
using NoLimits

# ── Sharded test runner ─────────────────────────────────────────────────────
#
# Every test file is tagged with a shard group (1..N_GROUPS). CI runs one group
# per matrix job (set via the TEST_GROUP env var) so the suite finishes in
# wall-clock ≈ slowest-group time instead of the full sequential ~100 min.
# Codecov merges the per-shard `lcov.info` uploads back into one report.
#
#   ENV["TEST_GROUP"] unset / "all" / "0"  → run every file (local default)
#   ENV["TEST_GROUP"] == "3"               → run only the files tagged group 3
#
# Group numbers were assigned by greedy longest-processing-time bin-packing of
# the per-file runtimes recorded in TEST_RUNTIMES.md, so each group is ≈ equal
# wall-clock. To rebalance after large timing changes, re-measure and reassign;
# the (file, group) table below is the single source of truth.
#
# NOTE: `include` evaluates each file in this module's global scope regardless
# of the enclosing loop, so top-level definitions in test files behave exactly
# as with the previous flat list of `include` calls.

const TEST_FILES = [
    # (file,                                    group)
    ("softtrees_tests.jl",                       1),
    ("ad_softtree.jl",                           6),
    ("ad_flow.jl",                               2),
    ("ad_random_effects.jl",                     4),
    ("ad_random_effects_values.jl",              1),
    ("ad_fixed_prede.jl",                        2),
    ("ad_differential_equation.jl",              6),
    ("ad_ode_solve_basic.jl",                    3),
    ("ad_ode_solve_richer.jl",                   5),
    ("ad_misc.jl",                               1),
    ("ad_model_full.jl",                         6),
    ("helpers_tests.jl",                         2),
    ("parameters_tests.jl",                      4),
    ("transform_tests.jl",                       1),
    ("fixed_effects_tests.jl",                   2),
    ("splines_tests.jl",                         6),
    ("covariates_tests.jl",                      6),
    ("random_effects_tests.jl",                  3),
    ("prede_tests.jl",                           5),
    ("differential_equation_tests.jl",           4),
    ("ode_solve_tests.jl",                       3),
    ("formulas_tests.jl",                        5),
    ("initialde_tests.jl",                       3),
    ("model_macro_tests.jl",                     2),
    ("model_tests.jl",                           3),
    ("equation_display_tests.jl",                4),
    ("data_model_tests.jl",                      2),
    ("identifiability_tests.jl",                 6),
    ("data_model_ode_tests.jl",                  4),
    ("summaries_data_model_tests.jl",            5),
    ("summaries_model_tests.jl",                 1),
    ("summaries_fit_uq_tests.jl",                1),
    ("summaries_parameter_comparison_tests.jl",  5),
    ("compact_show_tests.jl",                    6),
    ("data_simulation_tests.jl",                 5),
    ("ode_callbacks_tests.jl",                   3),
    ("plot_cache_tests.jl",                      4),
    ("plotting_functions_tests.jl",              2),
    ("vpc_tests.jl",                             3),
    ("plot_observation_distributions_tests.jl",  6),
    ("residual_plots_tests.jl",                  5),
    ("plot_random_effects_tests.jl",             5),
    ("random_effect_new_plots_tests.jl",         2),
    ("re_covariate_usage_tests.jl",              2),
    ("estimation_common_tests.jl",               3),
    # integration_* share fixtures; keep contiguous and in this order
    ("integration_no_re.jl",                     4),
    ("integration_simple_re.jl",                 4),
    ("integration_plotting.jl",                  4),
    ("estimation_mcmc_tests.jl",                 6),
    ("estimation_mcmc_re_tests.jl",              6),
    ("estimation_laplace_tests.jl",              6),
    ("estimation_focei_tests.jl",                6),
    ("estimation_pooled_tests.jl",               3),
    ("estimation_hutchinson_tests.jl",           1),
    ("estimation_mcem_tests.jl",                 1),
    ("estimation_mcem_is_tests.jl",              1),
    ("estimation_saem_tests.jl",                 2),
    ("saem_options_unit_tests.jl",               4),
    ("saem_adaptive_mh_tests.jl",                2),
    ("estimation_saem_autodetect_tests.jl",      4),
    ("estimation_saem_suffstats_tests.jl",       5),
    ("saem_saemixmh_tests.jl",                   3),
    ("estimation_multistart_tests.jl",           2),
    ("estimation_ghquadrature_tests.jl",         4),
    ("uq_tests.jl",                              5),
    ("uq_edge_cases_tests.jl",                   6),
    ("uq_plotting_tests.jl",                     4),
    ("hmm_continuous_tests.jl",                  5),
    ("hmm_discrete_time_tests.jl",               1),
    ("hmm_estimation_method_matrix_tests.jl",    3),
    ("hmm_mv_discrete_tests.jl",                 4),
    ("hmm_mv_continuous_tests.jl",               6),
    ("stickbreak_parameter_tests.jl",            6),
    ("stickbreak_transform_tests.jl",            3),
    ("stickbreak_uq_tests.jl",                   5),
    ("stickbreak_uq_natural_extension_tests.jl", 1),
    ("ad_stickbreak_hmm.jl",                     1),
    ("continuous_transition_matrix_tests.jl",    2),
    ("datasets_tests.jl",                        1),
    ("accessors_tests.jl",                       1),
    ("estimation_mle_tests.jl",                  1),
    ("estimation_map_tests.jl",                  4),
    ("estimation_vi_tests.jl",                   4),
    ("estimation_laplace_fit_tests.jl",          1),
    ("estimation_laplace_map_tests.jl",          5),
    ("estimation_cv_tests.jl",                   5),
    ("markov_discrete_time_tests.jl",            3),
    ("markov_continuous_time_tests.jl",          4),
    ("saem_schedule_tests.jl",                   4),
    ("saem_multichain_tests.jl",                 2),
    ("saem_sa_anneal_tests.jl",                  6),
    ("saem_var_lb_tests.jl",                     5),
    ("saem_anneal_ebe_tests.jl",                 3),
    ("saem_mstep_sa_on_params_tests.jl",         3),
    ("logit_scale_parameter_tests.jl",           5),
    ("logit_scale_transform_tests.jl",           3),
    ("logit_scale_uq_tests.jl",                  2),
    ("serialization_tests.jl",                   3),
    ("coverage_gap_tests.jl",                    2),
]

const _RAW_GROUP = get(ENV, "TEST_GROUP", "all")
const _TARGET_GROUP = (_RAW_GROUP == "all" || _RAW_GROUP == "0") ? nothing : parse(Int, _RAW_GROUP)

if _TARGET_GROUP === nothing
    @info "Running ALL test groups ($(length(TEST_FILES)) files). Set ENV[\"TEST_GROUP\"]=1..6 to shard."
else
    _n = count(((_, g),) -> g == _TARGET_GROUP, TEST_FILES)
    @info "Running test group $_TARGET_GROUP ($_n files). Unset TEST_GROUP to run everything."
end

# Each file runs inside its own @testset so that an error thrown while loading
# one file (outside any inner @test) is recorded as a failure but does NOT abort
# the remaining files in the shard. `include` still evaluates in this module's
# global scope, so top-level definitions leak across files exactly as before.
@testset "NoLimits" begin
    for (file, group) in TEST_FILES
        (_TARGET_GROUP === nothing || group == _TARGET_GROUP) || continue
        @testset "$file" begin
            include(file)
        end
    end
end

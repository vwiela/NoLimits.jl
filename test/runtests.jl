using Test
using NoLimits

# ── Weight-balanced sharded test runner ──────────────────────────────────────
#
# Each file is tagged with an approximate single-process runtime (seconds; see
# TEST_RUNTIMES.md). At load time the files are packed into `TEST_GROUPS` shards
# of roughly equal total weight via greedy longest-processing-time bin-packing,
# and the shard selected by `TEST_GROUP` is run. This lets the SAME file drive
# different shard counts (e.g. an 8-way no-coverage PR gate and a 10-way
# instrumented coverage run) just by changing the env vars.
#
#   TEST_GROUP  unset / "all" / "0"   → run every file (local default)
#   TEST_GROUP  == "3"                → run only shard 3 of TEST_GROUPS
#   TEST_GROUPS == "8" (default)      → total number of shards
#
# `include` evaluates each file in this module's global scope regardless of the
# enclosing loop, so top-level definitions behave as in a flat include list.

const TEST_FILES = [
    # (file,                                    approx_seconds)
    ("softtrees_tests.jl",                       1),
    ("ad_softtree.jl",                           1),
    ("ad_flow.jl",                               3),
    ("ad_random_effects.jl",                     8),
    ("ad_random_effects_values.jl",              4),
    ("ad_fixed_prede.jl",                        8),
    ("ad_differential_equation.jl",              3),
    ("ad_ode_solve_basic.jl",                    8),
    ("ad_ode_solve_richer.jl",                   4),
    ("ad_misc.jl",                               1),
    ("ad_model_full.jl",                         6),
    ("helpers_tests.jl",                         1),
    ("parameters_tests.jl",                      1),
    ("transform_tests.jl",                       3),
    ("fixed_effects_tests.jl",                   5),
    ("splines_tests.jl",                         1),
    ("covariates_tests.jl",                      1),
    ("random_effects_tests.jl",                  1),
    ("prede_tests.jl",                           1),
    ("differential_equation_tests.jl",           1),
    ("ode_solve_tests.jl",                       2),
    ("formulas_tests.jl",                        1),
    ("initialde_tests.jl",                       1),
    ("model_macro_tests.jl",                     1),
    ("model_tests.jl",                           3),
    ("equation_display_tests.jl",                2),
    ("data_model_tests.jl",                      9),
    ("identifiability_tests.jl",                 33),
    ("data_model_ode_tests.jl",                  5),
    ("summaries_data_model_tests.jl",            1),
    ("summaries_model_tests.jl",                 1),
    ("summaries_fit_uq_tests.jl",                57),
    ("summaries_parameter_comparison_tests.jl",  4),
    ("compact_show_tests.jl",                    5),
    ("data_simulation_tests.jl",                 8),
    ("ode_callbacks_tests.jl",                   5),
    ("plot_cache_tests.jl",                      58),
    ("plotting_functions_tests.jl",              72),
    ("vpc_tests.jl",                             17),
    ("plot_observation_distributions_tests.jl",  15),
    ("residual_plots_tests.jl",                  26),
    ("plot_random_effects_tests.jl",             196),
    ("random_effect_new_plots_tests.jl",         83),
    ("re_covariate_usage_tests.jl",              19),
    ("estimation_common_tests.jl",               11),
    ("integration_no_re.jl",                     107),
    ("integration_simple_re.jl",                 215),
    ("integration_plotting.jl",                  26),
    ("estimation_mcmc_tests.jl",                 51),
    ("estimation_mcmc_re_tests.jl",              78),
    ("estimation_laplace_tests.jl",              133),
    ("estimation_focei_tests.jl",                250),
    ("estimation_pooled_tests.jl",               351),
    ("estimation_hutchinson_tests.jl",           12),
    ("estimation_mcem_tests.jl",                 104),
    ("estimation_mcem_is_tests.jl",              33),
    ("estimation_saem_tests.jl",                 368),
    ("saem_options_unit_tests.jl",               104),
    ("saem_adaptive_mh_tests.jl",                31),
    ("estimation_saem_autodetect_tests.jl",      3),
    ("estimation_saem_suffstats_tests.jl",       46),
    ("saem_saemixmh_tests.jl",                   30),
    ("estimation_multistart_tests.jl",           104),
    ("estimation_ghquadrature_tests.jl",         176),
    ("uq_tests.jl",                              97),
    ("uq_edge_cases_tests.jl",                   208),
    ("uq_plotting_tests.jl",                     1),
    ("hmm_continuous_tests.jl",                  20),
    ("hmm_discrete_time_tests.jl",               15),
    ("hmm_estimation_method_matrix_tests.jl",    161),
    ("hmm_mv_discrete_tests.jl",                 15),
    ("hmm_mv_continuous_tests.jl",               12),
    ("stickbreak_parameter_tests.jl",            4),
    ("stickbreak_transform_tests.jl",            1),
    ("stickbreak_uq_tests.jl",                   9),
    ("stickbreak_uq_natural_extension_tests.jl", 23),
    ("ad_stickbreak_hmm.jl",                     9),
    ("continuous_transition_matrix_tests.jl",    2),
    ("datasets_tests.jl",                        3),
    ("accessors_tests.jl",                       150),
    ("estimation_mle_tests.jl",                  80),
    ("estimation_map_tests.jl",                  47),
    ("estimation_vi_tests.jl",                   13),
    ("estimation_laplace_fit_tests.jl",          410),
    ("estimation_laplace_map_tests.jl",          297),
    ("estimation_cv_tests.jl",                   63),
    ("markov_discrete_time_tests.jl",            81),
    ("markov_continuous_time_tests.jl",          95),
    ("saem_schedule_tests.jl",                   28),
    ("saem_multichain_tests.jl",                 38),
    ("saem_sa_anneal_tests.jl",                  104),
    ("saem_var_lb_tests.jl",                     130),
    ("saem_anneal_ebe_tests.jl",                 67),
    ("saem_mstep_sa_on_params_tests.jl",         41),
    ("logit_scale_parameter_tests.jl",           2),
    ("logit_scale_transform_tests.jl",           1),
    ("logit_scale_uq_tests.jl",                  2),
    ("serialization_tests.jl",                   125),
    ("coverage_gap_tests.jl",                    160),]

const _RAW_GROUP  = get(ENV, "TEST_GROUP", "all")
# "none" runs zero files: used by the CI `prepare` job, which calls Pkg.test only
# to precompile + cache the test environment (Pkg.test precompiles before running
# runtests.jl, so the cache is seeded even though no test file is included).
const _TARGET     = (_RAW_GROUP == "all" || _RAW_GROUP == "0") ? nothing :
                    _RAW_GROUP == "none" ? -1 :
                    parse(Int, _RAW_GROUP)
const _NGROUPS    = parse(Int, get(ENV, "TEST_GROUPS", "8"))

# integration_* files share fixtures and must stay together, in order.
const _PINNED = ["integration_no_re.jl", "integration_simple_re.jl", "integration_plotting.jl"]

# Greedy LPT bin-packing → Dict(file => shard 1..ngroups). The pinned group is
# packed as a single unit so its files land in the same shard.
function _assign_shards(files, ngroups)
    units = Vector{Tuple{Int, Vector{String}}}()   # (weight, members)
    pinned_w = 0; pinned_seen = false
    for (f, w) in files
        if f in _PINNED
            pinned_w += w
            pinned_seen && continue
            push!(units, (0, copy(_PINNED)))         # placeholder weight, fixed below
            pinned_seen = true
        else
            push!(units, (w, [f]))
        end
    end
    pinned_seen && (units[findfirst(u -> u[2] == _PINNED, units)] = (pinned_w, _PINNED))
    order = sortperm(units; by = u -> (-u[1], u[2][1]))   # heaviest first, name-tiebreak
    loads = zeros(Int, ngroups)
    assign = Dict{String, Int}()
    for idx in order
        w, members = units[idx]
        b = argmin(loads); loads[b] += w
        for m in members; assign[m] = b; end
    end
    return assign
end

const _SHARD = _TARGET === nothing ? nothing : _assign_shards(TEST_FILES, _NGROUPS)

if _TARGET === nothing
    @info "Running ALL $(length(TEST_FILES)) test files (set TEST_GROUP=1..$_NGROUPS to shard)."
else
    _n = count(((f, _),) -> _SHARD[f] == _TARGET, TEST_FILES)
    @info "Running shard $_TARGET of $_NGROUPS ($_n files)."
end

@testset "NoLimits" begin
    for (file, _) in TEST_FILES
        (_TARGET === nothing || _SHARD[file] == _TARGET) || continue
        @testset "$file" begin
            include(file)
        end
    end
end

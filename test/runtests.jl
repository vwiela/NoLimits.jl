using Test

# This is ONE CI job (no GitHub sharding), but it runs the suite as a handful of
# sequential `julia` subprocesses ("batches") rather than a single process.
# Reason: every distinct `@Model` in the suite emits type-specialized native
# code that Julia never frees within a process. Running all ~97 files in one
# process accumulates enough compiled code to exhaust RAM and stall (~50 min,
# 0% CPU). Splitting into batches caps per-process memory by exiting between
# batches. Each subprocess includes fixtures.jl fresh (lazy/memoized), so the
# only added cost is repeated `using NoLimits` (~tens of seconds per batch).
#
# Set NL_BATCHES to override the batch count (default below).

const TEST_FILES = [
    "softtrees_tests.jl",
    "ad_softtree.jl",
    "ad_flow.jl",
    "ad_random_effects.jl",
    "ad_random_effects_values.jl",
    "ad_fixed_prede.jl",
    "ad_differential_equation.jl",
    "ad_ode_solve_basic.jl",
    "ad_ode_solve_richer.jl",
    "ad_misc.jl",
    "ad_model_full.jl",
    "helpers_tests.jl",
    "parameters_tests.jl",
    "simplechains_nn_tests.jl",
    "transform_tests.jl",
    "fixed_effects_tests.jl",
    "splines_tests.jl",
    "covariates_tests.jl",
    "random_effects_tests.jl",
    "prede_tests.jl",
    "differential_equation_tests.jl",
    "ode_solve_tests.jl",
    "formulas_tests.jl",
    "initialde_tests.jl",
    "model_macro_tests.jl",
    "model_tests.jl",
    "equation_display_tests.jl",
    "data_model_tests.jl",
    "identifiability_tests.jl",
    "data_model_ode_tests.jl",
    "summaries_data_model_tests.jl",
    "summaries_model_tests.jl",
    "summaries_fit_uq_tests.jl",
    "summaries_parameter_comparison_tests.jl",
    "compact_show_tests.jl",
    "data_simulation_tests.jl",
    "ode_callbacks_tests.jl",
    "plot_cache_tests.jl",
    "plotting_functions_tests.jl",
    "vpc_tests.jl",
    "plot_observation_distributions_tests.jl",
    "residual_plots_tests.jl",
    "plot_random_effects_tests.jl",
    "random_effect_new_plots_tests.jl",
    "re_covariate_usage_tests.jl",
    "estimation_common_tests.jl",
    "integration_no_re.jl",
    "integration_simple_re.jl",
    "integration_plotting.jl",
    "estimation_mcmc_tests.jl",
    "estimation_mcmc_re_tests.jl",
    "estimation_laplace_tests.jl",
    "estimation_focei_tests.jl",
    "estimation_pooled_tests.jl",
    "estimation_hutchinson_tests.jl",
    "estimation_mcem_tests.jl",
    "estimation_mcem_is_tests.jl",
    "estimation_saem_tests.jl",
    "saem_options_unit_tests.jl",
    "saem_adaptive_mh_tests.jl",
    "estimation_saem_autodetect_tests.jl",
    "estimation_saem_suffstats_tests.jl",
    "saem_saemixmh_tests.jl",
    "estimation_multistart_tests.jl",
    "estimation_ghquadrature_tests.jl",
    "uq_tests.jl",
    "uq_edge_cases_tests.jl",
    "uq_plotting_tests.jl",
    "hmm_continuous_tests.jl",
    "hmm_discrete_time_tests.jl",
    "hmm_estimation_method_matrix_tests.jl",
    "hmm_mv_discrete_tests.jl",
    "hmm_mv_continuous_tests.jl",
    "stickbreak_parameter_tests.jl",
    "stickbreak_transform_tests.jl",
    "stickbreak_uq_tests.jl",
    "stickbreak_uq_natural_extension_tests.jl",
    "ad_stickbreak_hmm.jl",
    "continuous_transition_matrix_tests.jl",
    "datasets_tests.jl",
    "accessors_tests.jl",
    "estimation_mle_tests.jl",
    "estimation_map_tests.jl",
    "estimation_vi_tests.jl",
    "estimation_cv_tests.jl",
    "markov_discrete_time_tests.jl",
    "markov_continuous_time_tests.jl",
    "saem_schedule_tests.jl",
    "saem_multichain_tests.jl",
    "saem_sa_anneal_tests.jl",
    "saem_var_lb_tests.jl",
    "saem_anneal_ebe_tests.jl",
    "saem_mstep_sa_on_params_tests.jl",
    "logit_scale_parameter_tests.jl",
    "logit_scale_transform_tests.jl",
    "logit_scale_uq_tests.jl",
    "serialization_tests.jl",
    "coverage_gap_tests.jl",
    # Enzyme regression tests (merged from enzyme-compat). proxy = always-on,
    # ForwardDiff-only structural/numeric invariants; smoke = opt-in real Enzyme
    # gradients, no-op unless NOLIMITS_TEST_ENZYME=true (+ Julia>=1.12.5 + Enzyme).
    "enzyme_compat_proxy_tests.jl",
    "enzyme_smoke_tests.jl",
]

# --- Orchestrate sequential subprocess batches -----------------------------

const N_BATCHES = parse(Int, get(ENV, "NL_BATCHES", "6"))

# Contiguous split of TEST_FILES into N_BATCHES near-equal chunks (order
# preserved). Batches run sequentially, so the split only bounds per-process
# memory — balance isn't needed for wall-clock.
function _chunks(items, n)
    n = min(n, length(items))
    q, r = divrem(length(items), n)
    out = Vector{eltype(items)}[]
    i = 1
    for b in 1:n
        len = q + (b <= r ? 1 : 0)
        push!(out, items[i:(i + len - 1)])
        i += len
    end
    out
end

# Propagate the parent's relevant flags to each child so `Pkg.test` semantics
# (coverage, --check-bounds=auto) carry into the subprocesses.
#
# -O0 is the runtime lever: the suite is COMPILE-bound (577 distinct @Models,
# each forcing fresh type-specialized codegen), and tests use tiny data +
# maxiters<=3, so execution speed is irrelevant while LLVM optimization time
# dominates. -O0 cut a heavy 3-file batch from 358s (-O2) to 109s (~3.3x); -O1
# only reached 236s and the GitHub runner still exceeded the 120 min timeout at
# -O1. -O0 disables fma/muladd contraction, which can nudge optimizer
# trajectories on tiny degenerate-prone problems (one Laplace warm-start fit no
# longer converges); the affected test is convergence-gated rather than loosened.
# -O0 (unfused) is consistent across arm64/x86. Skipped under coverage (that
# workflow isn't latency-bound and instrumentation dominates anyway).
function _child_flags()
    o = Base.JLOptions()
    flags = String["--color=yes"]
    # check-bounds: 1=yes, 2=no, 0=default(auto) → leave unset
    o.check_bounds == 1 && push!(flags, "--check-bounds=yes")
    o.check_bounds == 2 && push!(flags, "--check-bounds=no")
    # code-coverage: 1=user, 2=all (Pkg.test sets coverage=true → user)
    o.code_coverage == 1 && push!(flags, "--code-coverage=user")
    o.code_coverage == 2 && push!(flags, "--code-coverage=all")
    if o.code_coverage == 0
        push!(flags, "-O0")
        push!(flags, "--min-optlevel=0")
    end
    flags
end

const _BATCHES = _chunks(TEST_FILES, N_BATCHES)
const _PROJECT = dirname(Base.active_project())
const _BATCH_SCRIPT = joinpath(@__DIR__, "run_batch.jl")

let failed = String[]
    for (i, batch) in enumerate(_BATCHES)
        @info "=== Test batch $i/$(length(_BATCHES)) ($(length(batch)) files) ===" files=batch
        cmd = `$(Base.julia_cmd()) $(_child_flags()) --project=$(_PROJECT) $(_BATCH_SCRIPT) $(batch)`
        ok = success(pipeline(cmd; stdout=stdout, stderr=stderr))
        ok || push!(failed, "batch $i: " * join(batch, ", "))
    end
    if !isempty(failed)
        error("Test batches failed:\n  " * join(failed, "\n  "))
    end
    @info "All $(length(_BATCHES)) test batches passed."
end

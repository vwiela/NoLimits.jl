export FOCEI
export FOCEIMAP

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using ForwardDiff
using LinearAlgebra
using Distributions
using SpecialFunctions: trigamma

# =====================================================================================
# FOCEI / FOCE estimation.
#
# FOCEI is the Laplace approximation with the inner negative Hessian of the log-joint
# replaced by the Gauss-Newton / expected-information form
#
#     𝓗_i  =  Σ_j Jᵢⱼᵀ ℐ(φᵢⱼ) Jᵢⱼ  −  ∇²_η log π_i(η̂_i)
#
# where
#   * φᵢⱼ           are the parameters of observation j's outcome distribution,
#   * Jᵢⱼ = ∂φᵢⱼ/∂η is obtained by one (first-order) ForwardDiff Jacobian through the
#                    formula evaluation (and the ODE solve, if any),
#   * ℐ(φ)          is the closed-form Fisher information of the outcome family
#                    (the "expected information"), evaluated in Distributions.jl's
#                    native parameterisation, and
#   * −∇²_η log π   is the exact curvature of the random-effects prior (= Ω⁻¹ for
#                    Normal/MvNormal random effects).
#
# Everything else — the empirical-Bayes inner solve, batching, the (q/2)·log(2π)
# constant, the Cholesky/log-det, caching and the result accessors — is shared with the
# Laplace implementation.  Only the Hessian builder differs, which drops the per-subject
# Hessian from second-order to first-order AD.
#
# FOCE (no interaction) freezes the dispersion-type parameters at the random-effects
# prior mean mᵢ = mean(πᵢ) (= 0 for centred Normal random effects, the classical case)
# and drops their η-sensitivity, while the location/structural parameters and their
# Jacobian stay at the conditional mode η̂.
# =====================================================================================

# -------------------------------------------------------------------------------------
# 1. Fisher-information registry
#
# For each supported outcome family we provide
#   _focei_params(d)                -> Vector of the distribution parameters (flat reals)
#   _focei_expected_information(d)  -> Matrix, the Fisher information ℐ(φ) in params order
#   _focei_dispersion_indices(d)    -> indices of _focei_params that FOCE freezes
#   _focei_is_supported(d)          -> Bool
#
# The Jacobian carries every link / transform / ODE dependence, so ℐ is stored purely in
# the native parameterisation.  Distributions without a registered ℐ are rejected up
# front (no numeric fallback by design).
# -------------------------------------------------------------------------------------

_focei_is_supported(::Any) = false
_focei_dispersion_indices(::Any) = Int[]
_focei_params(d) = error("FOCEI: distribution $(typeof(d)) is not supported (no closed-form Fisher information). Use Laplace instead, or remove this outcome.")
_focei_expected_information(d) = error("FOCEI: no closed-form Fisher information for $(typeof(d)). Use Laplace instead.")
_focei_paramcount(d) = length(_focei_params(d))

# --- location-scale families with a frozen dispersion parameter (the "I" in FOCEI) ---

# Normal(μ, σ): ℐ = diag(1/σ², 2/σ²)
_focei_is_supported(::Normal) = true
_focei_params(d::Normal) = [d.μ, d.σ]
_focei_paramcount(::Normal) = 2
_focei_dispersion_indices(::Normal) = [2]
function _focei_expected_information(d::Normal)
    iv = inv(d.σ * d.σ)
    z = zero(iv)
    return [iv z; z 2iv]
end

# LogNormal(μ, σ): same information as Normal in (μ, σ) — invariant to the y↦exp data map.
_focei_is_supported(::LogNormal) = true
_focei_params(d::LogNormal) = [d.μ, d.σ]
_focei_paramcount(::LogNormal) = 2
_focei_dispersion_indices(::LogNormal) = [2]
function _focei_expected_information(d::LogNormal)
    iv = inv(d.σ * d.σ)
    z = zero(iv)
    return [iv z; z 2iv]
end

# Laplace(μ, b): ℐ = diag(1/b², 1/b²)   (Distributions stores the scale as field θ)
_focei_is_supported(::Distributions.Laplace) = true
_focei_params(d::Distributions.Laplace) = [d.μ, d.θ]
_focei_paramcount(::Distributions.Laplace) = 2
_focei_dispersion_indices(::Distributions.Laplace) = [2]
function _focei_expected_information(d::Distributions.Laplace)
    ib = inv(d.θ * d.θ)
    z = zero(ib)
    return [ib z; z ib]
end

# Cauchy(μ, σ): ℐ = diag(1/2σ², 1/2σ²)
_focei_is_supported(::Cauchy) = true
_focei_params(d::Cauchy) = [d.μ, d.σ]
_focei_paramcount(::Cauchy) = 2
_focei_dispersion_indices(::Cauchy) = [2]
function _focei_expected_information(d::Cauchy)
    ih = inv(2 * d.σ * d.σ)
    z = zero(ih)
    return [ih z; z ih]
end

# --- single-parameter families (FOCE ≡ FOCEI; no dispersion to freeze) ---

# Exponential(θ) [scale]: ℐ = 1/θ²
_focei_is_supported(::Exponential) = true
_focei_params(d::Exponential) = [d.θ]
_focei_paramcount(::Exponential) = 1
_focei_expected_information(d::Exponential) = fill(inv(d.θ * d.θ), 1, 1)

# Poisson(λ): ℐ = 1/λ
_focei_is_supported(::Poisson) = true
_focei_params(d::Poisson) = [d.λ]
_focei_paramcount(::Poisson) = 1
_focei_expected_information(d::Poisson) = fill(inv(d.λ), 1, 1)

# Bernoulli(p): ℐ = 1/(p(1-p))
_focei_is_supported(::Bernoulli) = true
_focei_params(d::Bernoulli) = [d.p]
_focei_paramcount(::Bernoulli) = 1
_focei_expected_information(d::Bernoulli) = fill(inv(d.p * (one(d.p) - d.p)), 1, 1)

# Binomial(n, p): n fixed; ℐ_pp = n/(p(1-p)).  Only p is a differentiated parameter.
_focei_is_supported(::Binomial) = true
_focei_params(d::Binomial) = [d.p]
_focei_paramcount(::Binomial) = 1
_focei_expected_information(d::Binomial) = fill(d.n * inv(d.p * (one(d.p) - d.p)), 1, 1)

# Geometric(p): ℐ = 1/(p²(1-p))
_focei_is_supported(::Geometric) = true
_focei_params(d::Geometric) = [d.p]
_focei_paramcount(::Geometric) = 1
_focei_expected_information(d::Geometric) = fill(inv(d.p * d.p * (one(d.p) - d.p)), 1, 1)

# --- two-parameter families without a clean location/dispersion split (FOCE ≡ FOCEI) ---

# Gamma(α, θ) [shape, scale]: ℐ = [[ψ'(α), 1/θ], [1/θ, α/θ²]]
_focei_is_supported(::Gamma) = true
_focei_params(d::Gamma) = [d.α, d.θ]
_focei_paramcount(::Gamma) = 2
function _focei_expected_information(d::Gamma)
    a = d.α
    θ = d.θ
    iθ = inv(θ)
    return [trigamma(a) iθ; iθ a*iθ*iθ]
end

# Beta(α, β): ℐ = [[ψ'(α)-ψ'(α+β), -ψ'(α+β)], [-ψ'(α+β), ψ'(β)-ψ'(α+β)]]
_focei_is_supported(::Beta) = true
_focei_params(d::Beta) = [d.α, d.β]
_focei_paramcount(::Beta) = 2
function _focei_expected_information(d::Beta)
    tab = trigamma(d.α + d.β)
    iaa = trigamma(d.α) - tab
    ibb = trigamma(d.β) - tab
    return [iaa -tab; -tab ibb]
end

# --- multivariate normal outcome ---

# MvNormal(μ, Σ): mean block Σ⁻¹, covariance block ½·tr(Σ⁻¹ Eₐ Σ⁻¹ E_b) in vech(Σ)
# coordinates, zero cross block.  The vech ordering matches `_vech` (column-major upper
# triangle).
_focei_is_supported(::MvNormal) = true
function _focei_params(d::MvNormal)
    return vcat(collect(mean(d)), _vech(Matrix(cov(d))))
end
function _focei_paramcount(d::MvNormal)
    k = length(d)
    return k + k * (k + 1) ÷ 2
end
# Symmetric single-entry basis matrices in vech (upper-triangle, column-major) order.
function _focei_vech_basis(k::Int)
    bases = Matrix{Float64}[]
    for j in 1:k
        for i in 1:j
            E = zeros(Float64, k, k)
            E[i, j] += 1.0
            E[j, i] += 1.0
            i == j && (E[i, j] = 1.0)  # diagonal entry appears once
            push!(bases, E)
        end
    end
    return bases
end
function _focei_expected_information(d::MvNormal)
    Σ = Matrix(cov(d))
    P = inv(Σ)
    k = size(Σ, 1)
    nv = k * (k + 1) ÷ 2
    bases = _focei_vech_basis(k)
    T = eltype(P)
    Fcov = [0.5 * tr(P * bases[a] * P * bases[b]) for a in 1:nv, b in 1:nv]
    Z1 = zeros(T, k, nv)
    Z2 = zeros(T, nv, k)
    return [P Z1; Z2 Fcov]
end

# -------------------------------------------------------------------------------------
# 2. Per-observation distribution collection (mirrors `_loglikelihood_individual`,
#    returning the distribution objects instead of summing log-densities).
# -------------------------------------------------------------------------------------

function _focei_collect_obs_dists!(out::Vector, dm::DataModel, idx::Int, θ, η_ind, cache::_LLCache)
    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]
    if η_ind isa NamedTuple
        η_ind = ComponentArray(η_ind)
    end

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = cache.helpers,
            model_funs = cache.model_funs,
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end
        T = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        u0_T = eltype(u0) === T ? u0 : T.(u0)
        prob = remake(prob; u0 = u0_T, p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs, cache.ode_kwargs, (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs, cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return false
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    obs_cols = dm.config.obs_cols
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ?
               _varying_at(dm, ind, i, _get_col(dm.df, dm.config.time_col)[obs_rows]) :
               vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for col in obs_cols
            y = getfield(obs_series, col)[i]
            y === missing && continue
            push!(out, getproperty(obs, col))
        end
    end
    return true
end

# Returns a Vector of the outcome distributions for every (non-missing) observation in
# the batch, in a fixed order.
function _focei_obs_dists_batch(dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
                                b, const_cache::LaplaceConstantsCache, cache::_LLCache)
    out = Vector{Any}()
    for i in batch_info.inds
        η_ind = _build_eta_ind(dm, i, batch_info, b, const_cache, θ_re)
        ok = _focei_collect_obs_dists!(out, dm, i, θ_re, η_ind, cache)
        ok || return Vector{Any}()
    end
    return out
end

# Stacked outcome-distribution parameters Φ(b) — the Jacobian target.
function _focei_obs_params_batch(dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
                                 b, const_cache::LaplaceConstantsCache, cache::_LLCache)
    dists = _focei_obs_dists_batch(dm, batch_info, θ_re, b, const_cache, cache)
    T = promote_type(eltype(θ_re), eltype(b))
    isempty(dists) && return Vector{T}()
    return reduce(vcat, (_focei_params(d) for d in dists))
end

struct _FOCEIPhi{DM, INFO, TH, CC, CA}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
end
@inline (f::_FOCEIPhi)(b) = _focei_obs_params_batch(f.dm, f.info, f.θ, b, f.const_cache, f.cache)

struct _FOCEIPriorLogf{DM, INFO, TH, CC, CA}
    dm::DM
    info::INFO
    θ::TH
    const_cache::CC
    cache::CA
end
@inline (f::_FOCEIPriorLogf)(b) = _re_prior_logf_batch(f.dm, f.info, f.θ, b, f.const_cache, f.cache)

# -------------------------------------------------------------------------------------
# 3. Prior mean mᵢ in b-space (FOCE freezes dispersion parameters here).
# -------------------------------------------------------------------------------------

function _focei_prior_mean_b(dm::DataModel, batch_info::_LaplaceBatchInfo, θ_re::ComponentArray,
                             const_cache::LaplaceConstantsCache, cache::_LLCache)
    nb = batch_info.n_b
    T = eltype(θ_re)
    m = zeros(T, nb)
    model_funs = cache.model_funs
    helpers = cache.helpers
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    re_cache = dm.re_group_info.laplace_cache
    for (ri, re) in enumerate(re_cache.re_names)
        info = batch_info.re_info[ri]
        isempty(info.map.levels) && continue
        for (li, level_id) in enumerate(info.map.levels)
            idx = info.map.level_to_index[level_id]
            idx == 0 && continue
            rep_idx = info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ_re, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            r = info.ranges[idx]
            mv = mean(dist)
            if info.is_scalar
                m[first(r)] = mv isa AbstractVector ? mv[1] : mv
            else
                for (k, rr) in enumerate(r)
                    m[rr] = mv[k]
                end
            end
        end
    end
    return m
end

# -------------------------------------------------------------------------------------
# 4. The FOCEI / FOCE negative Hessian of the log-joint at (θ, b).
#    Returns the positive-(semi)definite precision 𝓗 = Σ Jⱼᵀ ℐ(φⱼ) Jⱼ − ∇²_b log π.
# -------------------------------------------------------------------------------------

function _focei_data_term(J, dists_b::Vector, dists_m, interaction::Bool, T::Type)
    nb = size(J, 2)
    isempty(dists_b) && return zeros(T, nb, nb)
    terms = Matrix{T}[]
    offset = 0
    for (j, d) in enumerate(dists_b)
        dj = _focei_paramcount(d)
        Jj = J[offset+1:offset+dj, :]
        offset += dj
        disp = _focei_dispersion_indices(d)
        if !interaction && !isempty(disp)
            # FOCE: dispersion parameters frozen at the prior mean; their η-sensitivity
            # is dropped.  ℐ of the supported location-scale families depends only on the
            # dispersion parameters, so evaluating ℐ at the prior-mean distribution gives
            # the correct (location-independent) weight.
            Ij = _focei_expected_information(dists_m[j])
            mask = Diagonal([k in disp ? 0.0 : 1.0 for k in 1:dj])
            Jj = mask * Jj
        else
            Ij = _focei_expected_information(d)
        end
        push!(terms, transpose(Jj) * Ij * Jj)
    end
    return sum(terms)
end

# Robustness wrapper (mirrors `_laplace_logf_batch`): an ODE solve or distribution
# construction that throws on an unstable parameter region the optimiser steps into
# (e.g. a negative state driving a fractional power, or a negative scale) is turned into
# a NaN Hessian. That makes the Cholesky/log-det fail and the marginal -Inf, so the outer
# optimiser backtracks instead of crashing the whole fit — matching Laplace's behaviour.
function _focei_negH_batch(dm::DataModel, batch_info::_LaplaceBatchInfo, θ, b,
                           const_cache::LaplaceConstantsCache, cache::_LLCache; interaction::Bool)
    try
        return _focei_negH_batch_impl(dm, batch_info, θ, b, const_cache, cache; interaction=interaction)
    catch err
        if err isa LinearAlgebra.PosDefException || err isa LinearAlgebra.SingularException ||
           err isa DomainError || err isa ArgumentError
            nb = batch_info.n_b
            return fill(convert(promote_type(eltype(θ), eltype(b)), NaN), nb, nb)
        end
        rethrow(err)
    end
end

function _focei_negH_batch_impl(dm::DataModel, batch_info::_LaplaceBatchInfo, θ, b,
                                const_cache::LaplaceConstantsCache, cache::_LLCache; interaction::Bool)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
    nb = batch_info.n_b
    T = promote_type(eltype(θ_re), eltype(b))
    nb == 0 && return zeros(T, 0, 0)

    dists_b = _focei_obs_dists_batch(dm, batch_info, θ_re, b, const_cache, cache)
    if isempty(dists_b)
        Hdata = zeros(T, nb, nb)
    else
        Φ = _FOCEIPhi(dm, batch_info, θ_re, const_cache, cache)
        J = ForwardDiff.jacobian(Φ, b)
        # Guard against an ODE solve that succeeds at the base point but reports failure
        # under the AD seed (length mismatch): signal a non-finite Hessian so the marginal
        # becomes -Inf and the outer optimiser backtracks out of the unstable region.
        D_expected = sum(_focei_paramcount, dists_b; init=0)
        size(J, 1) == D_expected || return fill(convert(T, NaN), nb, nb)
        dists_m = nothing
        if !interaction && any(d -> !isempty(_focei_dispersion_indices(d)), dists_b)
            m_b = _focei_prior_mean_b(dm, batch_info, θ_re, const_cache, cache)
            dists_m = _focei_obs_dists_batch(dm, batch_info, θ_re, m_b, const_cache, cache)
        end
        Hdata = _focei_data_term(J, dists_b, dists_m, interaction, T)
    end

    Λ = -ForwardDiff.hessian(_FOCEIPriorLogf(dm, batch_info, θ_re, const_cache, cache), b)
    return Hdata .+ Λ
end

# -------------------------------------------------------------------------------------
# 5. Hessian-builder plug-in for the shared Laplace objective/gradient machinery.
#
# FOCEI reuses Laplace's optimised marginal objective and trace-estimator gradient
# (config/buffer caching, the implicit db*/dθ term, batching, threading) verbatim,
# swapping only the inner Hessian via `_build_hess_b`.  The builder returns H = ∇²_b log f
# so that the downstream `negH = -H`; FOCEI's negH IS 𝓗 = Σ Jᵀ ℐ(φ) J − ∇²log π, hence it
# returns -𝓗.  Differentiating this first-order-AD Hessian for the log-determinant
# gradient is one AD order cheaper than Laplace's exact second-order Hessian, so the
# FOCEI marginal and its gradient are computed through the same fast code path.
# -------------------------------------------------------------------------------------

struct _FOCEIHess <: _HessMode
    interaction::Bool
end

@inline function _build_hess_b(m::_FOCEIHess, dm::DataModel, batch_info::_LaplaceBatchInfo,
                               θ, b, const_cache::LaplaceConstantsCache, cache::_LLCache,
                               ad_cache::Union{Nothing, LaplaceADCache}, bi::Int;
                               ctx::AbstractString="")
    return -_focei_negH_batch(dm, batch_info, θ, b, const_cache, cache; interaction=m.interaction)
end

# -------------------------------------------------------------------------------------
# 6. Method types.
# -------------------------------------------------------------------------------------

"""
    FOCEI(; interaction=true, optimizer, optim_kwargs, adtype, inner_*, multistart_*,
            jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, theta_tol,
            lb, ub, ignore_model_bounds) <: FittingMethod

First-Order Conditional Estimation with Interaction for random-effects models.

`FOCEI` is the Laplace approximation with the inner negative Hessian of the log-joint
replaced by the expected-information form `Σ Jᵀ ℐ(φ) J − ∇²log π`, where `J = ∂φ/∂η` is a
first-order Jacobian of the outcome-distribution parameters and `ℐ(φ)` is the closed-form
Fisher information of the outcome family.  This drops the per-subject Hessian from
second-order to first-order automatic differentiation and yields a positive-definite
curvature by construction.

Set `interaction=false` for FOCE, which freezes dispersion-type parameters (e.g. a
residual-error standard deviation) at the random-effects prior mean and ignores their
dependence on the random effects.

Supported outcome families: `Normal`, `LogNormal`, `Laplace`, `Cauchy`, `Exponential`,
`Poisson`, `Bernoulli`, `Binomial`, `Geometric`, `Gamma`, `Beta`, `MvNormal`.  Hidden
Markov / Markov outcome models and any family without a registered Fisher information are
not supported — use [`Laplace`](@ref) for those.

Keyword arguments mirror [`Laplace`](@ref); the only addition is `interaction::Bool=true`.
"""
struct FOCEI{O, K, A, IO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
    ignore_model_bounds::Bool
    interaction::Bool
end

FOCEI(; interaction::Bool=true,
       optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
       optim_kwargs=NamedTuple(),
       adtype=Optimization.AutoForwardDiff(),
       inner_options=nothing,
       hessian_options=nothing,
       cache_options=nothing,
       multistart_options=nothing,
       inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
       inner_kwargs=NamedTuple(),
       inner_adtype=Optimization.AutoForwardDiff(),
       inner_grad_tol=:auto,
       multistart_n=50,
       multistart_k=10,
       multistart_grad_tol=inner_grad_tol,
       multistart_max_rounds=1,
       multistart_sampling=:lhs,
       jitter=1e-6,
       max_tries=6,
       jitter_growth=10.0,
       adaptive_jitter=true,
       jitter_scale=1e-6,
       theta_tol=0.0,
       lb=nothing,
       ub=nothing,
       ignore_model_bounds=false) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    hess = hessian_options === nothing ? LaplaceHessianOptions(jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, true, false, 8) : hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    FOCEI(optimizer, optim_kwargs, adtype, inner, hess, cache, ms, lb, ub, ignore_model_bounds, interaction)
end

"""
    FOCEIMAP(; interaction=true, ...) <: FittingMethod

FOCEI with MAP-regularised fixed effects: identical to [`FOCEI`](@ref) but adds the
log-prior of the fixed effects to the outer objective.  Requires a prior on at least one
free fixed effect.  See [`FOCEI`](@ref) for keyword arguments.
"""
struct FOCEIMAP{O, K, A, IO, HO, CO, MS, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    hessian::HO
    cache::CO
    multistart::MS
    lb::L
    ub::U
    ignore_model_bounds::Bool
    interaction::Bool
end

FOCEIMAP(; interaction::Bool=true,
          optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
          optim_kwargs=NamedTuple(),
          adtype=Optimization.AutoForwardDiff(),
          inner_options=nothing,
          hessian_options=nothing,
          cache_options=nothing,
          multistart_options=nothing,
          inner_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
          inner_kwargs=NamedTuple(),
          inner_adtype=Optimization.AutoForwardDiff(),
          inner_grad_tol=:auto,
          multistart_n=50,
          multistart_k=10,
          multistart_grad_tol=inner_grad_tol,
          multistart_max_rounds=5,
          multistart_sampling=:lhs,
          jitter=1e-6,
          max_tries=6,
          jitter_growth=10.0,
          adaptive_jitter=true,
          jitter_scale=1e-6,
          theta_tol=0.0,
          lb=nothing,
          ub=nothing,
          ignore_model_bounds=false) = begin
    inner = inner_options === nothing ? LaplaceInnerOptions(inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) : inner_options
    hess = hessian_options === nothing ? LaplaceHessianOptions(jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, true, false, 8) : hessian_options
    cache = cache_options === nothing ? LaplaceCacheOptions(theta_tol) : cache_options
    ms = multistart_options === nothing ? LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol, multistart_max_rounds, multistart_sampling) : multistart_options
    FOCEIMAP(optimizer, optim_kwargs, adtype, inner, hess, cache, ms, lb, ub, ignore_model_bounds, interaction)
end

# -------------------------------------------------------------------------------------
# 7. Distribution guard.
# -------------------------------------------------------------------------------------

function _focei_validate_distributions(dm::DataModel; ode_args::Tuple, ode_kwargs::NamedTuple)
    fe = dm.model.fixed.fixed
    θ0 = get_θ0_untransformed(fe)
    θ_re = _symmetrize_psd_params(θ0, fe)
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
    for info in batch_infos
        info.n_b == 0 && continue
        b0 = zeros(Float64, info.n_b)
        dists = _focei_obs_dists_batch(dm, info, θ_re, b0, const_cache, ll_cache)
        isempty(dists) && continue
        for d in dists
            _focei_is_supported(d) ||
                error("FOCEI does not support outcome distribution $(typeof(d)). " *
                      "Supported families: Normal, LogNormal, Laplace, Cauchy, Exponential, " *
                      "Poisson, Bernoulli, Binomial, Geometric, Gamma, Beta, MvNormal. " *
                      "Use Laplace for hidden-Markov / unsupported outcomes.")
        end
        return nothing  # families are fixed by the formula; one observed batch suffices
    end
    return nothing
end

# -------------------------------------------------------------------------------------
# 8. Fit drivers.
# -------------------------------------------------------------------------------------

function _fit_focei(dm::DataModel, method, is_map::Bool, args, fit_kwargs;
                    constants::NamedTuple,
                    constants_re::NamedTuple,
                    penalty::NamedTuple,
                    ode_args::Tuple,
                    ode_kwargs::NamedTuple,
                    serialization::SciMLBase.EnsembleAlgorithm,
                    rng::AbstractRNG,
                    theta_0_untransformed::Union{Nothing, ComponentArray},
                    store_data_model::Bool)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("FOCEI requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("FOCEI requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("FOCEI requires at least one free fixed effect. Remove constants or add a fixed/random effect.")

    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    if is_map
        priors = get_priors(fe)
        for name in fixed_names
            hasproperty(priors, name) || error("FOCEIMAP requires priors on all fixed effects. Missing prior for $(name).")
            getfield(priors, name) isa Priorless && error("FOCEIMAP requires priors on all fixed effects. Priorless for $(name).")
        end
    end

    transform = get_transform(fe)
    θ0_t = transform(θ0_u)
    inv_transform = get_inverse_transform(fe)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)

    _focei_validate_distributions(dm; ode_args=ode_args, ode_kwargs=ode_kwargs)

    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = serialization isa SciMLBase.EnsembleThreads ?
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
    n_batches = length(batch_infos)
    Tθ = eltype(θ0_t)
    bstar_cache = _LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                   fill(Tθ(NaN), n_batches),
                                   [Vector{Tθ}() for _ in 1:n_batches],
                                   falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)

    θ0_free_t = θ0_t[free_names]
    axs_free = getaxes(θ0_free_t)
    axs_full = getaxes(θ_const_t)

    hmode = _FOCEIHess(method.interaction)
    has_penalty = !isempty(keys(penalty))
    T0 = eltype(θ0_free_t)
    obj_cache = _LaplaceObjCache{T0, ComponentArray}(nothing, T0(Inf),
                    ComponentArray(zeros(T0, length(θ0_free_t)), axs_free), false)

    function obj_only(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        cached_obj = _laplace_obj_cache_lookup_obj(obj_cache, θt_free, method.cache.theta_tol)
        cached_obj !== nothing && return cached_obj
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj = _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                      inner=inner_opts, hessian=method.hessian,
                                      cache_opts=method.cache, multistart=multistart_opts,
                                      rng=rng, serialization=serialization, hmode=hmode)
        !isfinite(obj) && return infT
        if is_map
            lp = logprior(fe, θu)
            !isfinite(lp) && return infT
            obj += -lp
        end
        has_penalty && (obj += _penalty_value(θu, penalty))
        _laplace_obj_cache_set_obj!(obj_cache, θt_free, obj)
        return obj
    end

    function obj_grad(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        cached = _laplace_obj_cache_lookup(obj_cache, θt_free, method.cache.theta_tol)
        cached !== nothing && return cached
        T = eltype(θt_free)
        infT = convert(T, Inf)
        zgrad = ComponentArray(zeros(T, length(θt_free)), axs_free)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        obj, grad_full, _ = _laplace_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                        inner=inner_opts, hessian=method.hessian,
                                                        cache_opts=method.cache, multistart=multistart_opts,
                                                        rng=rng, serialization=serialization, hmode=hmode)
        !isfinite(obj) && return (infT, zgrad)
        grad_u = grad_full
        if is_map
            lp = logprior(fe, θu)
            !isfinite(lp) && return (infT, zgrad)
            obj += -lp
            grad_u = grad_u .- ForwardDiff.gradient(x -> logprior(fe, x), θu)
        end
        if has_penalty
            obj += _penalty_value(θu, penalty)
            grad_u = grad_u .+ ForwardDiff.gradient(x -> _penalty_value(x, penalty), θu)
        end
        grad_t_ca = apply_inv_jacobian_T(inv_transform, θt_full, grad_u)
        grad_free = similar(θt_free)
        for name in free_names
            setproperty!(grad_free, name, getproperty(grad_t_ca, name))
        end
        # NaN gradient → treat as non-finite objective so the line search backtracks.
        any(isnan, grad_free) && return (infT, zgrad)
        _laplace_obj_cache_set_obj_grad!(obj_cache, θt_free, obj, grad_free)
        return (obj, grad_free)
    end

    optf = OptimizationFunction(obj_only, method.adtype;
                                grad = (G, θt, p) -> (G .= obj_grad(θt, p)[2]))
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = lower_t[free_names]
    upper_t_free = upper_t[free_names]
    use_bounds = !method.ignore_model_bounds && !(all(isinf, lower_t_free) && all(isinf, upper_t_free))
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    if user_bounds
        lb = method.lb
        ub = method.ub
        lb isa ComponentArray && (lb = lb[free_names])
        ub isa ComponentArray && (ub = ub[free_names])
    else
        lb = lower_t_free
        ub = upper_t_free
    end
    use_bounds = use_bounds || user_bounds
    prob = use_bounds ? OptimizationProblem(optf, θ0_free_t; lb=lb, ub=ub) :
                        OptimizationProblem(optf, θ0_free_t)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), axs_full)
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = is_map ?
             LaplaceMAPResult(sol, sol.objective, niter, raw, NamedTuple(), ebe_cache.bstar_cache.b_star) :
             LaplaceResult(sol, sol.objective, niter, raw, NamedTuple(), ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end

function _fit_model(dm::DataModel, method::FOCEI, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants, constants_re=constants_re, penalty=penalty,
                  ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization,
                  rng=rng, theta_0_untransformed=theta_0_untransformed, store_data_model=store_data_model)
    return _fit_focei(dm, method, false, args, fit_kwargs;
                      constants=constants, constants_re=constants_re, penalty=penalty,
                      ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization,
                      rng=rng, theta_0_untransformed=theta_0_untransformed, store_data_model=store_data_model)
end

function _fit_model(dm::DataModel, method::FOCEIMAP, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants, constants_re=constants_re, penalty=penalty,
                  ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization,
                  rng=rng, theta_0_untransformed=theta_0_untransformed, store_data_model=store_data_model)
    return _fit_focei(dm, method, true, args, fit_kwargs;
                      constants=constants, constants_re=constants_re, penalty=penalty,
                      ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization,
                      rng=rng, theta_0_untransformed=theta_0_untransformed, store_data_model=store_data_model)
end

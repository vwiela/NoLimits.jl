# ── Shared test fixtures ─────────────────────────────────────────────────────
#
# The suite's biggest cost is rebuilding the same models and re-running the same
# fits in testset after testset: every distinct `@Model` recompiles the formula/
# DE/logf closures + the estimator specialised to its types, and every fit
# repeats that work. This module builds a small set of canonical model archetypes
# and ONE fit per (archetype, method), built LAZILY and memoised, so the whole
# suite shares them: a model is compiled once and each method is fit once, then
# reused by the estimation / plotting / UQ / residual / summary / RE-diagnostic
# tests instead of each rebuilding its own.
#
# Conventions:
#   * Tiny data + maxiters ≤ 3 ⇒ each fit is cheap.
#   * Accessors are `fx_<archetype>_<thing>()`; call them, never build your own.
#   * A test that genuinely needs a different structure (multi-group batching,
#     MVN, ODE, NPF, non-normal outcome, an error path) keeps a bespoke `@Model`
#     — share for the common case, stay faithful for the specific one (balanced).

using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Turing

const _FX = Dict{Symbol, Any}()
_fx(key::Symbol, build) = get!(build, _FX, key)
const _SER = NoLimits.EnsembleSerial()

# ── Datasets ─────────────────────────────────────────────────────────────────
fx_nore_df() = _fx(:nore_df, () -> DataFrame(
    ID = [1, 1, 2, 2, 3, 3, 4, 4],
    t  = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y  = [0.2, 0.25, 0.1, 0.18, 0.3, 0.34, 0.16, 0.21],
))

fx_re_df(; n_ids::Int=6, n_obs::Int=3) = _fx(:re_df, () -> begin
    ids = repeat(1:n_ids, inner=n_obs)
    t   = repeat(collect(0.0:(n_obs - 1)), n_ids)
    y   = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, t)]
    DataFrame(ID=ids, t=t, y=y)
end)

# ── No random effects: y ~ Normal(a + b*t, σ) ────────────────────────────────
fx_nore_model() = _fx(:nore_model, () -> @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        σ = RealNumber(0.3, scale=:log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end)

fx_nore_prior_model() = _fx(:nore_prior_model, () -> @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        b = RealNumber(0.1, prior=Normal(0.0, 1.0))
        σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end)

fx_nore_dm()       = _fx(:nore_dm,       () -> DataModel(fx_nore_model(),       fx_nore_df(); primary_id=:ID, time_col=:t))
fx_nore_prior_dm() = _fx(:nore_prior_dm, () -> DataModel(fx_nore_prior_model(), fx_nore_df(); primary_id=:ID, time_col=:t))

# ── One scalar random effect: y ~ Normal(a + η, σ), η ~ Normal(0, ω) ──────────
fx_re_model() = _fx(:re_model, () -> @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.3, scale=:log, lower=1e-8, upper=Inf)
        ω = RealNumber(0.4, scale=:log, lower=1e-8, upper=Inf)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column=:ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end)

fx_re_prior_model() = _fx(:re_prior_model, () -> @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        ω = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column=:ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end)

fx_re_dm()       = _fx(:re_dm,       () -> DataModel(fx_re_model(),       fx_re_df(); primary_id=:ID, time_col=:t))
fx_re_prior_dm() = _fx(:re_prior_dm, () -> DataModel(fx_re_prior_model(), fx_re_df(); primary_id=:ID, time_col=:t))

# ── Fits: one per (archetype, method), built once and reused everywhere ───────
# Fixed-effects-only methods on the no-RE archetype:
fx_mle()    = _fx(:mle,    () -> fit_model(fx_nore_dm(),       NoLimits.MLE(; optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_map()    = _fx(:map,    () -> fit_model(fx_nore_prior_dm(), NoLimits.MAP(; optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_pooled() = _fx(:pooled, () -> fit_model(fx_re_dm(),         NoLimits.Pooled(; optim_kwargs=(maxiters=3,)); serialization=_SER))

# Random-effects methods on the scalar-RE archetype:
fx_laplace() = _fx(:laplace, () -> fit_model(fx_re_dm(), NoLimits.Laplace(; optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_lmap()    = _fx(:lmap,    () -> fit_model(fx_re_prior_dm(), NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_focei()   = _fx(:focei,   () -> fit_model(fx_re_dm(), NoLimits.FOCEI(; multistart_n=1, multistart_k=1, optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_ghq()     = _fx(:ghq,     () -> fit_model(fx_re_dm(), NoLimits.GHQuadrature(; level=2, optim_kwargs=(maxiters=3,)); serialization=_SER))
fx_saem()    = _fx(:saem,    () -> fit_model(fx_re_dm(), NoLimits.SAEM(; maxiters=2, q_store_max=2); serialization=_SER))
fx_mcem()    = _fx(:mcem,    () -> fit_model(fx_re_dm(), NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false), maxiters=2); serialization=_SER))

# Bayesian fits (priors required). MCMC supports random effects; VI does not.
fx_mcmc()    = _fx(:mcmc,    () -> fit_model(fx_nore_prior_dm(), NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false)); rng=Random.Xoshiro(1)))
fx_mcmc_re() = _fx(:mcmc_re, () -> fit_model(fx_re_prior_dm(),   NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false)); rng=Random.Xoshiro(2)))
fx_vi()      = _fx(:vi,      () -> fit_model(fx_nore_prior_dm(), NoLimits.VI(; turing_kwargs=(max_iter=30, progress=false)); rng=Random.Xoshiro(3)))

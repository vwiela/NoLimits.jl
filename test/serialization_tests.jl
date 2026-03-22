using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using SciMLBase

# ── Shared fixtures ───────────────────────────────────────────────────────────

function _simple_model_no_re()
    @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
end

function _simple_model_with_re()
    @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin y ~ Normal(a + η, σ) end
    end
end

function _simple_model_with_priors()
    @Model begin
        @fixedEffects begin
            a = RealNumber(0.5; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; prior=LogNormal(0.0, 0.5))
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
end

_df_no_re() = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0])
_df_with_re() = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0,1.2,1.1])

# ── MLE ───────────────────────────────────────────────────────────────────────

@testset "Serialization MLE" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)
    @test get_converged(res) == get_converged(res2)
    @test get_raw(res2) === nothing
    @test get_data_model(res2) === dm
end

# ── MLE include_data ──────────────────────────────────────────────────────────

@testset "Serialization MLE include_data" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=model)   # no dm — reconstructed from saved df

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) !== nothing
end

# ── MLE no dm ─────────────────────────────────────────────────────────────────

@testset "Serialization MLE no dm" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res)               # include_data=false
    res2 = load_fit(path)             # no model, no dm

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) === nothing
end

# ── MAP ───────────────────────────────────────────────────────────────────────

@testset "Serialization MAP" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MAP())

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
end

# ── Laplace ───────────────────────────────────────────────────────────────────

@testset "Serialization Laplace" begin
    model = _simple_model_with_re()
    df    = _df_with_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.Laplace())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=model)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)

    re1 = get_random_effects(dm, res)
    re2 = get_random_effects(res2)
    @test re1.η == re2.η

    ll1 = get_loglikelihood(dm, res)
    ll2 = get_loglikelihood(res2)
    @test ll1 ≈ ll2
end

# ── MCMC ──────────────────────────────────────────────────────────────────────

@testset "Serialization MCMC" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MCMC(;
                turing_kwargs=(n_samples=50, n_adapt=25, progress=false)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_chain(res).value == get_chain(res2).value
    @test get_n_samples(res) == get_n_samples(res2)
    @test get_sampler(res2) isa NoLimits._SavedSamplerStub
    @test get_sampler(res2).kind == :nuts
end

# ── VI ────────────────────────────────────────────────────────────────────────

@testset "Serialization VI" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.VI(;
                turing_kwargs=(max_iter=100, family=:meanfield, progress=false)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test get_vi_state(res2) === nothing
    s1 = sample_posterior(res;  n_draws=20)
    s2 = sample_posterior(res2; n_draws=20)
    @test size(s1) == size(s2)
    @test get_vi_trace(res2) isa AbstractVector
end

# ── Multistart ────────────────────────────────────────────────────────────────

@testset "Serialization Multistart" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 2.0))
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
    df  = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0])
    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms  = NoLimits.Multistart(dists=(; a=Normal(1.0, 0.2)),
                              n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=5,)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test length(get_multistart_results(res)) == length(get_multistart_results(res2))
    @test get_multistart_best_index(res) == get_multistart_best_index(res2)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test eltype(get_multistart_errors(res2)) == String
end

# ── Format version check ──────────────────────────────────────────────────────

@testset "Serialization format version check" begin
    import JLD2
    path = tempname() * ".jld2"
    # Write a fake saved object with wrong version
    JLD2.jldsave(path; saved=(; format_version=999))
    @test_throws ErrorException load_fit(path)
end

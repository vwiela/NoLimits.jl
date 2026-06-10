using Test
using NoLimits
using Lux
# Import ONLY the SimpleChains symbols we use — a bare `using SimpleChains` would put its
# exports (e.g. `relu`, which Lux also exports) into Main, and because the batch runner shares
# one process across files, that makes `relu`/etc. ambiguous for every later test file that
# uses them unqualified. (Same rationale as the `using Turing: MH` note in fixtures.jl.)
using SimpleChains: SimpleChain, static, TurboDense, numparam
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using ForwardDiff
using FiniteDifferences

# NNParameters accepts a SimpleChains.SimpleChain as a drop-in alternative to a Lux.Chain.
# SimpleChains parameters are natively a flat vector and the forward pass `chain(x, θ)` is
# ForwardDiff-compatible, so the SimpleChain backend works with every ForwardDiff-based fit
# (it is NOT Enzyme-differentiable — that is the documented limitation, Lux covers Enzyme).

@testset "SimpleChains NNParameters construction" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    nn = NNParameters(chain; name=:nn, function_name=:NN1, seed=0)
    @test nn.name == :nn
    @test nn.function_name == :NN1
    @test nn.chain isa SimpleChain
    @test nn.value isa Vector{Float64}
    @test length(nn.value) == numparam(chain)
    @test length(nn.value) > 0
    @test all(isinf, nn.lower) && all(<(0), nn.lower)
    @test all(isinf, nn.upper) && all(>(0), nn.upper)

    # Deterministic init by seed.
    nn_a = NNParameters(chain; function_name=:NNa, seed=3)
    nn_b = NNParameters(chain; function_name=:NNb, seed=3)
    nn_c = NNParameters(chain; function_name=:NNc, seed=4)
    @test nn_a.value == nn_b.value
    @test nn_a.value != nn_c.value

    # Prior validation mirrors the Lux backend (length-based via _check_nn_prior).
    n = length(nn.value)
    nn_pv = NNParameters(chain; function_name=:NN2, seed=0, prior=fill(Normal(), n))
    @test nn_pv.prior isa AbstractVector{<:Distribution}
    @test_throws ErrorException NNParameters(chain; function_name=:NN3, prior=:not_a_prior)
    @test_throws ErrorException NNParameters(chain; function_name=:NN4, prior=fill(Normal(), n - 1))
    nn_mvn = NNParameters(chain; function_name=:NN5, seed=0, prior=MvNormal(zeros(n), I))
    @test nn_mvn.prior isa Distribution
    @test_throws ErrorException NNParameters(chain; function_name=:NN6, prior=MvNormal(zeros(n - 1), I))
end

@testset "SimpleChains model_fun plumbing + output shape" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    fe = @fixedEffects begin
        σ = RealNumber(0.4)
        ζ = NNParameters(chain; function_name=:NN1, seed=1, calculate_se=false)
    end
    mf = get_model_funs(fe)
    @test haskey(mf, :NN1)
    θ0 = get_θ0_untransformed(fe)
    p = collect(θ0.ζ)
    x = [0.3, -0.2]

    # The model function must faithfully call the SimpleChain on (x, params).
    direct = chain(x, p)
    out = mf.NN1(x, p)
    @test out isa AbstractVector            # indexable like the Lux Vector output
    @test length(out) == 1
    @test isapprox(out[1], direct[1]; rtol=1e-10)
    @test mf.NN1(x, p)[1] isa Real          # `NN1(...)[1]` is a scalar usable in formulas
end

@testset "SimpleChains ForwardDiff correctness" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 5), TurboDense(identity, 1))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name=:NN1, seed=2, calculate_se=false)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).ζ)
    x = [0.4, -0.1]

    # Gradient w.r.t. the NN parameters matches finite differences.
    g_fd = ForwardDiff.gradient(v -> mf.NN1(x, v)[1], p)
    g_num = FiniteDifferences.grad(central_fdm(5, 1), v -> mf.NN1(x, v)[1], p)[1]
    @test all(isfinite, g_fd)
    @test isapprox(g_fd, g_num; rtol=1e-5, atol=1e-7)

    # Gradient w.r.t. the input also works (Duals flow through both arguments).
    gx_fd = ForwardDiff.gradient(xx -> mf.NN1(xx, p)[1], x)
    gx_num = FiniteDifferences.grad(central_fdm(5, 1), xx -> mf.NN1(xx, p)[1], x)[1]
    @test isapprox(gx_fd, gx_num; rtol=1e-5, atol=1e-7)
end

@testset "SimpleChains end-to-end MLE + Laplace" begin
    df = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   Age=[0.3,0.3,-0.2,-0.2,0.1,0.1], BMI=[0.1,0.1,0.4,0.4,-0.3,-0.3],
                   y=[1.0,1.1,0.9,1.0,1.2,1.05])

    # Fixed-effects-only NN model -> MLE (ForwardDiff default).
    chain_mle = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    model_mle = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale=:log)
            ζ = NNParameters(chain_mle; function_name=:NN1, calculate_se=false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI])
        end
        @formulas begin
            μ = NN1([x.Age, x.BMI], ζ)[1]
            y ~ Normal(μ, σ)
        end
    end
    dm_mle = DataModel(model_mle, df; primary_id=:ID, time_col=:t)
    res_mle = fit_model(dm_mle, NoLimits.MLE(optim_kwargs=(; iterations=10)))
    @test isfinite(NoLimits.get_objective(res_mle))

    # NN + random effect -> Laplace (exercises the Empirical-Bayes ForwardDiff path).
    chain_lap = SimpleChain(static(1), TurboDense(tanh, 4), TurboDense(identity, 1))
    model_lap = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.5, scale=:log)
            ζ = NNParameters(chain_lap; function_name=:NN1, calculate_se=false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            μ = NN1([x.Age], ζ)[1] + η
            y ~ Normal(μ, σ)
        end
    end
    dm_lap = DataModel(model_lap, df; primary_id=:ID, time_col=:t)
    res_lap = fit_model(dm_lap, NoLimits.Laplace(optim_kwargs=(; iterations=10)))
    @test isfinite(NoLimits.get_objective(res_lap))

    # calculate_formulas_obs returns the expected observation distribution.
    θ = get_θ0_untransformed(model_lap.fixed.fixed)
    obs = calculate_formulas_obs(model_lap, θ, ComponentArray((η = 0.1,)),
                                 (x = (Age = 0.3,),), (t = 0.0,))
    @test obs.y isa Normal
end

@testset "SimpleChains and Lux NN backends coexist" begin
    # Both backends usable in one session; the Lux path is unchanged.
    sc = SimpleChain(static(2), TurboDense(tanh, 3), TurboDense(identity, 1))
    lx = Lux.Chain(Lux.Dense(2, 3, tanh), Lux.Dense(3, 1))
    fe = @fixedEffects begin
        ζ_sc = NNParameters(sc; function_name=:SC, seed=0, calculate_se=false)
        ζ_lx = NNParameters(lx; function_name=:LX, seed=0, calculate_se=false)
    end
    mf = get_model_funs(fe)
    θ0 = get_θ0_untransformed(fe)
    x = [0.2, 0.5]
    @test mf.SC(x, collect(θ0.ζ_sc))[1] isa Real
    @test mf.LX(x, collect(θ0.ζ_lx))[1] isa Real
end

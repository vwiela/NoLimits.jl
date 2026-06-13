using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using Random

function _softtree_scalar(x, tree, params)
    return sum(tree(x, params))
end

@testset "SoftTree AD" begin
    # Compare gradients across AD backends for inputs and parameters.
    # Random (asymmetric) params: all-zero init makes leaf probabilities uniform,
    # which would mask any leaf-ordering disagreement between the eval variants.
    tree = SoftTree(3, 2, 2)
    params = init_params(tree, Xoshiro(7))
    x = [0.1, -0.2, 0.3]

    f(xv) = _softtree_scalar(xv, tree, params)
    f_inplace(xv) = sum(tree(xv, params, Val(:inplace)))
    f_fast(xv) = sum(tree(xv, params, Val(:fast)))

    flat, recon = destructure_params(params)
    params2 = recon(flat)
    @test size(params2.node_weights) == size(params.node_weights)
    @test size(params2.leaf_values) == size(params.leaf_values)
    @test length(params2.node_biases) == length(params.node_biases)
    f_params(v) = sum(tree(x, recon(v)))

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), x)
    @test length(grad_fwd) == length(x)

    # The mutating-buffer Val(:inplace) variant must agree with the default under
    # ForwardDiff (it uses a plain promoted-eltype buffer, so Duals flow through).
    val_fwd_inplace, grad_fwd_inplace = value_and_gradient(f_inplace, AutoForwardDiff(), x)
    @test isapprox(val_fwd_inplace, val_fwd; rtol = 1e-6, atol = 1e-8)
    @test isapprox(grad_fwd_inplace, grad_fwd; rtol = 1e-6, atol = 1e-8)

    val_fwd_fast, grad_fwd_fast = value_and_gradient(f_fast, AutoForwardDiff(), x)
    @test isapprox(val_fwd_fast, val_fwd; rtol = 1e-6, atol = 1e-8)
    @test isapprox(grad_fwd_fast, grad_fwd; rtol = 1e-6, atol = 1e-8)

    val_fwd_p, grad_fwd_p = value_and_gradient(f_params, AutoForwardDiff(), flat)
    @test length(grad_fwd_p) == length(flat)
end

using LinearAlgebra
using Lux
import ForwardDiff

using ComponentArrays

export ForwardTransform
export TransformSpec
export InverseTransform
export log_forward
export log_inverse
export logit_forward
export logit_inverse
export cholesky_forward
export cholesky_inverse
export expm_forward
export expm_inverse
export stickbreak_forward
export stickbreak_inverse
export lograterows_forward
export lograterows_inverse
export apply_inv_jacobian_T

struct TransformSpec
    name::Symbol
    kind::Symbol
    size::Tuple{Int, Int}
    mask::Union{Nothing, Vector{Symbol}}
end

# `out_axes`/`n_out` (when provided) describe the output ComponentArray layout and
# enable a type-stable assembly path: the legacy `ComponentArray(NamedTuple{...})`
# construction is dynamic (runtime `names`) and routes through ComponentArrays'
# `make_idx`, whose IdDict lookups (`jl_eqtable_get`) have no Enzyme forward-mode
# rule. The 2-arg constructors keep the legacy dynamic path for ad-hoc transforms
# (e.g. restricted transforms in plotting/UQ).
struct ForwardTransform{A}
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
    out_axes::A
    n_out::Int
end
ForwardTransform(names, specs) = ForwardTransform(names, specs, nothing, -1)

struct InverseTransform{A}
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
    out_axes::A
    n_out::Int
end
InverseTransform(names, specs) = InverseTransform(names, specs, nothing, -1)

function (ft::ForwardTransform)(θ::ComponentArray)
    if ft.out_axes === nothing
        vals = _transform_vals(θ, ft.names, ft.specs)
        return _assemble_ca(vals, ft.names, ft.out_axes, ft.n_out, eltype(θ))
    end
    # Axes path: write each parameter's transformed values straight into the flat
    # output buffer. The legacy route collected per-parameter results in a boxed
    # `Vector{Any}` (the `map` closure's branches return heterogeneous types) and
    # then re-walked it — measurably hot, since the inverse runs inside the ODE RHS
    # and both run once per objective evaluation.
    flat = Vector{eltype(θ)}(undef, ft.n_out)
    k = 1
    @inbounds for i in eachindex(ft.names)
        k = _write_forward_spec!(flat, k, ft.specs[i], getproperty(θ, ft.names[i]))
    end
    k == ft.n_out + 1 ||
        error("Transform output length mismatch: expected $(ft.n_out), got $(k - 1).")
    return ComponentArray(flat, ft.out_axes)
end

function (it::InverseTransform)(θ::ComponentArray)
    if it.out_axes === nothing
        vals = _inverse_vals(θ, it.names, it.specs)
        return _assemble_ca(vals, it.names, it.out_axes, it.n_out, eltype(θ))
    end
    flat = Vector{eltype(θ)}(undef, it.n_out)
    k = 1
    @inbounds for i in eachindex(it.names)
        k = _write_inverse_spec!(flat, k, it.specs[i], getproperty(θ, it.names[i]))
    end
    k == it.n_out + 1 ||
        error("Transform output length mismatch: expected $(it.n_out), got $(k - 1).")
    return ComponentArray(flat, it.out_axes)
end

# Direct flat-buffer writers for the axes path. Branch-for-branch identical math to
# `_transform_vals` / `_inverse_vals` (which remain as the legacy no-axes route):
# scalar kinds write elementwise with no temporaries; matrix/simplex kinds reuse the
# shared helpers and copy their (small) results in the same element order
# (`for x in M` is column-major, matching the legacy `vec`-based writes).
function _write_forward_spec!(flat::Vector, k::Int, spec::TransformSpec, val)
    kind = spec.kind
    if kind === :log
        if val isa Number
            flat[k] = log(val)
            return k + 1
        end
        for x in val
            flat[k] = log(x)
            k += 1
        end
        return k
    elseif kind === :logit
        if val isa Number
            flat[k] = logit_forward(val)
            return k + 1
        end
        for x in val
            flat[k] = logit_forward(x)
            k += 1
        end
        return k
    elseif kind === :elementwise
        for j in eachindex(val)
            flat[k] = _scalar_forward(spec.mask[j], val[j])
            k += 1
        end
        return k
    elseif kind === :cholesky
        return _write_flat!(flat, cholesky_forward(val), k)
    elseif kind === :expm
        M = expm_forward(val)
        n = size(M, 1)
        for j in 1:n, i in 1:j
            flat[k] = M[i, j]
            k += 1
        end
        return k
    elseif kind === :stickbreak
        return _write_flat!(flat, stickbreak_forward(val), k)
    elseif kind === :stickbreakrows
        n = size(val, 1)
        for i in 1:n
            k = _write_flat!(flat, stickbreak_forward(view(val, i, :)), k)
        end
        return k
    elseif kind === :lograterows
        return _write_flat!(flat, lograterows_forward(val), k)
    else
        return _write_flat!(flat, val, k)
    end
end

function _write_inverse_spec!(flat::Vector, k::Int, spec::TransformSpec, val)
    kind = spec.kind
    if kind === :log
        if val isa Number
            flat[k] = exp(val)
            return k + 1
        end
        for x in val
            flat[k] = exp(x)
            k += 1
        end
        return k
    elseif kind === :logit
        if val isa Number
            flat[k] = logit_inverse(val)
            return k + 1
        end
        for x in val
            flat[k] = logit_inverse(x)
            k += 1
        end
        return k
    elseif kind === :elementwise
        for j in eachindex(val)
            flat[k] = _scalar_inverse(spec.mask[j], val[j])
            k += 1
        end
        return k
    elseif kind === :cholesky
        n1, n2 = spec.size
        return _write_flat!(flat, cholesky_inverse(reshape(val, n1, n2)), k)
    elseif kind === :expm
        n1, _ = spec.size
        return _write_flat!(flat, expm_inverse(_sym_from_upper(val, n1)), k)
    elseif kind === :stickbreak
        return _write_flat!(flat, stickbreak_inverse(val), k)
    elseif kind === :stickbreakrows
        n = spec.size[1]
        return _write_flat!(flat, _stickbreakrow_inverse(val, n), k)
    elseif kind === :lograterows
        n = spec.size[1]
        return _write_flat!(flat, lograterows_inverse(val, n), k)
    else
        return _write_flat!(flat, val, k)
    end
end

# Legacy dynamic assembly (runtime names; ForwardDiff-compatible, not Enzyme-forward).
function _assemble_ca(
        vals, names::Vector{Symbol}, ::Nothing, n_out::Int, ::Type{T}) where {T}
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end

# Type-stable assembly: write the heterogeneous per-parameter values into one flat
# vector (single up-front allocation, assigned-once slots) and wrap it with the
# precomputed axes.
function _assemble_ca(
        vals, names::Vector{Symbol}, out_axes::Tuple, n_out::Int, ::Type{T}) where {T}
    flat = Vector{T}(undef, n_out)
    k = 1
    for v in vals
        k = _write_flat!(flat, v, k)
    end
    k == n_out + 1 ||
        error("Transform output length mismatch: expected $(n_out), got $(k - 1).")
    return ComponentArray(flat, out_axes)
end

_write_flat!(flat::Vector, v::Number, k::Int) = (flat[k] = v; k + 1)

function _write_flat!(flat::Vector, v::AbstractArray, k::Int)
    for x in vec(v)
        flat[k] = x
        k += 1
    end
    return k
end

function log_forward(x::Real)
    return log(x)
end

function log_inverse(x::Real)
    return exp(x)
end

function logit_forward(x::Real)
    return clamp(log(x / (1 - x)), -20.0, 20.0)
end

function logit_inverse(x::Real)
    return Lux.sigmoid(clamp(x, -20.0, 20.0))
end

@inline function _logit_inv_jacobian(x::Real)
    abs(x) >= 20.0 && return zero(x)
    s = logit_inverse(x)
    return s * (one(s) - s)
end

# Safe log-scale inverse Jacobian: g * exp(t), but returns 0 when g == 0 and exp(t) = Inf.
# Without this guard, 0 * Inf = NaN in IEEE arithmetic, which corrupts gradients when
# a parameter is driven to an extreme transformed value by the optimizer.
@inline function _safe_log_inv_jac(g::Real, t::Real)
    e = exp(t)
    isinf(e) && iszero(g) && return zero(typeof(g * e))
    return g * e
end

"""
    stickbreak_forward(p) -> Vector

Map a k-probability vector `p` (summing to 1, all ≥ 0) to a k-1 vector of
unconstrained reals via the logistic stick-breaking transform:

    ν_i = p_i / (1 - Σ_{j<i} p_j),   t_i = logit(ν_i),  i = 1, ..., k-1

The last probability is determined and is not stored.
"""
function stickbreak_forward(p::AbstractVector{<:Real})
    k = length(p)
    k >= 2 || error("stickbreak_forward requires at least 2 elements.")
    T = promote_type(eltype(p), Float64)
    # `remaining` tracks 1 - Σ_{j<i} p_j, updated by subtracting each pᵢ in the loop.
    # A simple mutating loop suffices here: unlike the inverse, the forward transform
    # is not typically differentiated by Zygote, so it need not be non-mutating.
    t = Vector{T}(undef, k - 1)
    remaining = one(T)
    for i in 1:(k - 1)
        pi = T(p[i])
        νi = pi / remaining
        t[i] = logit_forward(νi)
        remaining -= pi
    end
    return t
end

"""
    stickbreak_inverse(t) -> Vector

Map a k-1 vector of unconstrained reals to a k-probability vector via the
inverse logistic stick-breaking transform. Recovers a valid probability vector
summing to 1 with all entries in [0, 1].

Single-allocation sequential fill. `remaining` carries the running product
`∏_{j<i}(1-σ_j)` — the same left-to-right multiplication order as the historical
`cumprod` formulation, so results are bit-identical. ForwardDiff/Enzyme-forward
compatible (plain index assignment into a locally allocated, promoted-eltype
vector); reverse-mode Zygote would need the old non-mutating `cumprod` form.
"""
function stickbreak_inverse(t::AbstractVector{<:Real})
    k1 = length(t)
    T = typeof(logit_inverse(one(eltype(t))))
    out = Vector{T}(undef, k1 + 1)
    remaining = one(T)
    @inbounds for i in 1:k1
        σi = logit_inverse(t[i])
        out[i] = σi * remaining
        remaining *= one(T) - σi
    end
    out[k1 + 1] = remaining
    return out
end

# Apply stickbreak_forward row-wise to an n×n matrix; returns n*(n-1) flat vector.
function _stickbreakrow_forward(P::AbstractMatrix{<:Real})
    n = size(P, 1)
    r1 = stickbreak_forward(view(P, 1, :))
    out = Vector{eltype(r1)}(undef, n * (n - 1))
    copyto!(out, 1, r1, 1, n - 1)
    for i in 2:n
        ri = stickbreak_forward(view(P, i, :))
        copyto!(out, (i - 1) * (n - 1) + 1, ri, 1, n - 1)
    end
    return out
end

# Apply stickbreak_inverse row-wise; reconstructs n×n row-stochastic matrix.
function _stickbreakrow_inverse(t::AbstractVector{<:Real}, n::Int)
    k1 = n - 1
    r1 = stickbreak_inverse(view(t, 1:k1))
    M = Matrix{eltype(r1)}(undef, n, n)
    for j in 1:n
        M[1, j] = r1[j]
    end
    for i in 2:n
        ri = stickbreak_inverse(view(t, ((i - 1) * k1 + 1):(i * k1)))
        for j in 1:n
            M[i, j] = ri[j]
        end
    end
    return M
end

"""
    lograterows_forward(Q) -> Vector

Map an `n×n` Q-matrix (off-diagonal entries ≥ 0, rows sum to 0) to an `n*(n-1)`
unconstrained vector by taking the element-wise logarithm of the off-diagonal entries,
traversed in row-major order (row 1: columns 1..n skipping diagonal, then row 2, etc.).
"""
function lograterows_forward(Q::AbstractMatrix{<:Real})
    n = size(Q, 1)
    T = promote_type(eltype(Q), Float64)
    out = Vector{T}(undef, n * (n - 1))
    idx = 1
    for i in 1:n
        for j in 1:n
            i == j && continue
            out[idx] = log(T(Q[i, j]))
            idx += 1
        end
    end
    return out
end

"""
    lograterows_inverse(t, n) -> Matrix

Map an `n*(n-1)` unconstrained vector (log off-diagonal rates) back to an `n×n` Q-matrix.
Off-diagonal entries are `exp(t[k])`; diagonal entries are `-rowsum`.
"""
function lograterows_inverse(t::AbstractVector{<:Real}, n::Int)
    T = promote_type(eltype(t), Float64)
    Q = zeros(T, n, n)
    idx = 1
    for i in 1:n
        # Accumulate the row sum inline (same j-ascending order as the previous
        # filtered-generator `sum`, so results are bit-identical) instead of
        # re-walking the row through a lazy generator.
        rowsum = zero(T)
        for j in 1:n
            i == j && continue
            qij = exp(T(t[idx]))
            Q[i, j] = qij
            rowsum += qij
            idx += 1
        end
        Q[i, i] = -rowsum
    end
    return Q
end

# Compute J^T * g for stick-breaking inverse: maps k-vector g to k-1-vector.
# Uses the O(k) backward recurrence:
#   Forward pass: r[i] = remaining probability before step i, p[i] = σ(t[i])*r[i]
#   Backward pass: g_t[j] = σ'(t[j]) * r[j] * (g[j] - S / r[j+1])
#                  S accumulates Σ_{i≥j+1} p[i] * g[i]
function _stickbreak_inv_jacobian_T(t::AbstractVector{<:Real}, g::AbstractVector{<:Real})
    k1 = length(t)
    k = k1 + 1
    length(g) == k || error("Gradient length $(length(g)) must equal k=$(k).")
    T = promote_type(eltype(t), eltype(g), Float64)
    # Forward pass: compute r and p on natural scale
    r = Vector{T}(undef, k)
    p = Vector{T}(undef, k)
    r[1] = one(T)
    for i in 1:k1
        σi = logit_inverse(T(t[i]))
        p[i] = σi * r[i]
        r[i + 1] = r[i] - p[i]
    end
    p[k] = r[k]
    # Backward pass
    g_t = Vector{T}(undef, k1)
    S = T(g[k]) * p[k]  # = g[k] * r[k]
    for j in k1:-1:1
        σj = logit_inverse(T(t[j]))
        σj_prime = σj * (one(T) - σj)
        rj1 = r[j + 1]
        g_t[j] = σj_prime * r[j] * (T(g[j]) - S / rj1)
        S += T(g[j]) * p[j]
    end
    return g_t
end

@inline function _scalar_forward(kind::Symbol, x::Real)
    kind === :log && return log(x)
    kind === :logit && return logit_forward(x)
    return x
end

@inline function _scalar_inverse(kind::Symbol, x::Real)
    kind === :log && return exp(x)
    kind === :logit && return logit_inverse(x)
    return x
end

@inline function _scalar_inv_jacobian(kind::Symbol, g::Real, x::Real)
    kind === :log && return _safe_log_inv_jac(g, x)
    kind === :logit && return g * _logit_inv_jacobian(x)
    return g
end

function cholesky_forward(A::AbstractMatrix{<:Real})
    Asym = Symmetric((A + A') / 2)
    L = cholesky(Asym).L
    T = Matrix(L)
    d = diag(T)
    return T - Diagonal(d) + Diagonal(log.(d))
end

function cholesky_inverse(T::AbstractMatrix{<:Real})
    L = Matrix(LowerTriangular(T))
    d = diag(L)
    Lexp = L - Diagonal(d) + Diagonal(exp.(d))
    return Lexp * Lexp'
end

function expm_forward(A::AbstractMatrix{<:Real})
    Asym = Symmetric((A + A') / 2)
    return Matrix(log(Asym))
end

function expm_inverse(T::AbstractMatrix{<:Real})
    # `exp(Symmetric(T))` is symmetric in value, but under ForwardDiff its derivative
    # (Fréchet) partials can come back slightly asymmetric. Since Ω(T) is symmetric for
    # every T, the true derivative is symmetric too, so re-wrapping the result in
    # `Symmetric` before densifying both fixes the partials and guarantees an exactly
    # Hermitian matrix — required by the (stricter) `cholesky` inside MvNormal/PDMats.
    return Matrix(Symmetric(exp(Symmetric(T))))
end

function _expm_frechet(A::AbstractMatrix{<:Real}, E::AbstractMatrix{<:Real})
    n = size(A, 1)
    size(A, 2) == n || error("A must be square.")
    size(E, 1) == n && size(E, 2) == n || error("E must match A size.")
    TT = promote_type(eltype(A), eltype(E))
    Z = zeros(TT, n, n)
    M = [Matrix{TT}(A) Matrix{TT}(E); Z Matrix{TT}(A)]
    EM = exp(M)
    return EM[1:n, 1:n], EM[1:n, (n + 1):(2n)]
end

# ForwardDiff-aware matrix exponential of a symmetric matrix (single-level Dual).
# The generic eigen-based `exp(Symmetric)` has NaN / asymmetric ForwardDiff derivatives at
# repeated eigenvalues (e.g. Ω = I, the typical optimization/UQ point — the derivative of an
# eigen-based matrix function divides by eigenvalue gaps), and the Padé `exp(::Matrix)` has no
# `Dual` method. So compute the value with the eigen-exp of the (Float64) symmetric value, and
# each partial direction with the AD-safe block-2×2 Padé Fréchet derivative (`_expm_frechet`,
# which has no eigengaps), then reassemble the `Dual`. This matches the reverse-mode Jacobian
# the transform already uses and yields finite, exactly-symmetric derivatives, so the
# reconstructed covariance is accepted by the (stricter) `cholesky` inside MvNormal/PDMats.
# Nested Duals fall back to the generic `<:Real` method above.
function expm_inverse(T::AbstractMatrix{ForwardDiff.Dual{
        Tg, V, N}}) where {Tg, V <: AbstractFloat, N}
    n = size(T, 1)
    Tv = ForwardDiff.value.(T)
    Sv = 0.5 .* (Tv .+ Tv')                          # symmetric value
    Mv = Matrix(Symmetric(exp(Symmetric(Sv))))       # value (eigen ok — no AD here)
    dMs = ntuple(N) do k
        Pk = ForwardDiff.partials.(T, k)
        Ek = 0.5 .* (Pk .+ Pk')                      # symmetric seed direction
        _, F = _expm_frechet(Sv, Ek)
        Matrix(Symmetric(F))                          # symmetric Fréchet derivative
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = ForwardDiff.Dual{Tg}(Mv[i, j], ntuple(k -> dMs[k][i, j], N)...)
    end
    return out
end

function _upper_tri_vec(T::AbstractMatrix{<:Real})
    n = size(T, 1)
    return [T[i, j] for j in 1:n for i in 1:j]
end

function _sym_from_upper(v::AbstractVector{<:Real}, n::Int)
    idx = 1
    return [begin
                if i <= j
                    val = v[idx]
                    idx += 1
                    val
                else
                    v[(j - 1) * j ÷ 2 + i]
                end
            end
            for i in 1:n, j in 1:n]
end

function _upper_tri_vec_grad(G::AbstractMatrix{<:Real})
    n = size(G, 1)
    out = Vector{eltype(G)}(undef, n * (n + 1) ÷ 2)
    idx = 1
    for j in 1:n
        for i in 1:j
            out[idx] = i == j ? G[i, j] : (G[i, j] + G[j, i])
            idx += 1
        end
    end
    return out
end

function apply_inv_jacobian_T(
        it::InverseTransform, θt::ComponentArray, grad_u::ComponentArray)
    names = it.names
    specs = it.specs
    # Sequential flat fill on the θt layout. The previous `map` collected the
    # heterogeneous per-parameter results into a boxed `Vector{Any}` and then wrote
    # them back through a `setproperty!` loop — both boxing-heavy on a per-gradient
    # path (FOCEI / identifiability). Per-spec math is unchanged (`_inv_jac_spec_val`).
    flat = Vector{eltype(θt)}(undef, length(θt))
    k = 1
    for i in eachindex(names)
        v = _inv_jac_spec_val(
            specs[i], getproperty(θt, names[i]), getproperty(grad_u, names[i]))
        k = _write_flat!(flat, v, k)
    end
    k == length(θt) + 1 ||
        error("apply_inv_jacobian_T length mismatch: expected $(length(θt)), got $(k - 1).")
    return ComponentArray(flat, getaxes(θt))
end

function _inv_jac_spec_val(spec::TransformSpec, θti, gu)
    begin
        if spec.kind == :log
            return _safe_log_inv_jac.(gu, θti)
        elseif spec.kind == :logit
            return gu .* _logit_inv_jacobian.(θti)
        elseif spec.kind == :elementwise
            return [_scalar_inv_jacobian(spec.mask[j], gu[j], θti[j])
                    for j in eachindex(θti)]
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            T = reshape(θti, n1, n2)
            L = Matrix(LowerTriangular(T))
            d = diag(L)
            Lexp = L - Diagonal(d) + Diagonal(exp.(d))
            G = gu
            Gsym = G + G'
            grad_Lexp = Gsym * Lexp
            grad_T = zeros(eltype(grad_Lexp), n1, n2)
            for j in 1:n2
                for i in j:n1
                    if i == j
                        grad_T[i, j] = grad_Lexp[i, j] * exp(d[i])
                    else
                        grad_T[i, j] = grad_Lexp[i, j]
                    end
                end
            end
            return vec(grad_T)
        elseif spec.kind == :expm
            n1, n2 = spec.size
            T = _sym_from_upper(θti, n1)
            S = Symmetric(T)
            G = gu
            Gsym = Symmetric((G + G') / 2)
            _, F = _expm_frechet(S, Gsym)
            return _upper_tri_vec_grad(F)
        elseif spec.kind == :stickbreak
            # θti: k-1 transformed; gu: k natural gradient → k-1 output
            return _stickbreak_inv_jacobian_T(θti, gu)
        elseif spec.kind == :stickbreakrows
            # θti: n*(n-1) transformed; gu: n×n natural gradient → n*(n-1) output
            n = spec.size[1]
            T_acc = promote_type(eltype(θti), eltype(gu), Float64)
            out = Vector{T_acc}(undef, n * (n - 1))
            for i in 1:n
                chunk_t = @view θti[((i - 1) * (n - 1) + 1):(i * (n - 1))]
                g_row = vec(gu[i, :])
                g_t_row = _stickbreak_inv_jacobian_T(chunk_t, g_row)
                out[((i - 1) * (n - 1) + 1):(i * (n - 1))] .= g_t_row
            end
            return out
        elseif spec.kind == :lograterows
            # θti: n*(n-1) log off-diagonal rates; gu: n×n natural gradient → n*(n-1) output
            # Q[i,j] = exp(t[k]) for off-diagonal (i,j), Q[i,i] = -sum_row
            # ∂Q[i,j]/∂t[k] = exp(t[k]), ∂Q[i,i]/∂t[k] = -exp(t[k])
            # So g_t[k] = exp(t[k]) * (g_u[i,j] - g_u[i,i])
            n = spec.size[1]
            T_acc = promote_type(eltype(θti), eltype(gu), Float64)
            out = Vector{T_acc}(undef, n * (n - 1))
            idx = 1
            for i in 1:n
                g_ii = T_acc(gu[i, i])
                for j in 1:n
                    i == j && continue
                    out[idx] = _safe_log_inv_jac(T_acc(gu[i, j]) - g_ii, T_acc(θti[idx]))
                    idx += 1
                end
            end
            return out
        else
            return gu
        end
    end
end

function _transform_vals(
        θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    return map(1:length(names)) do i
        name = names[i]
        spec = specs[i]
        val = θ[name]
        if spec.kind == :log
            return log.(val)
        elseif spec.kind == :logit
            return logit_forward.(val)
        elseif spec.kind == :elementwise
            return [_scalar_forward(spec.mask[j], val[j]) for j in eachindex(val)]
        elseif spec.kind == :cholesky
            T = cholesky_forward(val)
            return vec(T)
        elseif spec.kind == :expm
            T = expm_forward(val)
            return _upper_tri_vec(T)
        elseif spec.kind == :stickbreak
            return stickbreak_forward(val)
        elseif spec.kind == :stickbreakrows
            return _stickbreakrow_forward(val)
        elseif spec.kind == :lograterows
            return lograterows_forward(val)
        else
            return val
        end
    end
end

function _inverse_vals(
        θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    return map(1:length(names)) do i
        name = names[i]
        spec = specs[i]
        val = θ[name]
        if spec.kind == :log
            return exp.(val)
        elseif spec.kind == :logit
            return logit_inverse.(val)
        elseif spec.kind == :elementwise
            return [_scalar_inverse(spec.mask[j], val[j]) for j in eachindex(val)]
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            T = reshape(val, n1, n2)
            return cholesky_inverse(T)
        elseif spec.kind == :expm
            n1, n2 = spec.size
            T = _sym_from_upper(val, n1)
            return expm_inverse(T)
        elseif spec.kind == :stickbreak
            return stickbreak_inverse(val)
        elseif spec.kind == :stickbreakrows
            n = spec.size[1]
            return _stickbreakrow_inverse(val, n)
        elseif spec.kind == :lograterows
            n = spec.size[1]
            return lograterows_inverse(val, n)
        else
            return val
        end
    end
end

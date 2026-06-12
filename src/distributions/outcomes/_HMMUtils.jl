# Accepts tuples as well as vectors: the per-row HMM logpdf paths fuse their
# per-state terms into tuples (no intermediate vectors); index-order max scan
# and exp-sum are identical for both, so values are bit-identical.
@inline function _hmm_logsumexp(xs::Union{AbstractVector, Tuple})
    isempty(xs) && return -Inf
    m = xs[1]
    @inbounds for i in 2:length(xs)
        m = max(m, xs[i])
    end
    isfinite(m) || return m
    s = zero(m)
    # Ignore terms that are far below the max in value space. For Dual numbers,
    # these underflow-scale terms can carry non-finite sensitivities (e.g. -Inf)
    # that are numerically irrelevant to the value but can poison gradients.
    cutoff = -700.0
    @inbounds for x in xs
        δ = x - m
        δ > cutoff || continue
        s += exp(δ)
    end
    return m + log(s)
end

# Static-length view of a state-probability vector, sized by the (type-level)
# number of emission distributions: lets logpdf/posterior fuse their per-state
# work into tuple operations without allocating intermediate vectors.
@inline _hmm_probs_tuple(p::AbstractVector, ::NTuple{N, Any}) where {N} = ntuple(
    i -> @inbounds(p[i]), Val(N))

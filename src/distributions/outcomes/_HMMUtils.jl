@inline function _hmm_logsumexp(xs::AbstractVector)
    isempty(xs) && return -Inf
    m = xs[1]
    @inbounds for i in 2:length(xs)
        m = max(m, xs[i])
    end
    isfinite(m) || return m
    s = zero(m)
    @inbounds for x in xs
        s += exp(x - m)
    end
    return m + log(s)
end

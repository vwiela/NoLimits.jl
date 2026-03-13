# Internal helpers shared by HMM estimation and simulation paths.
# Not exported.

using Distributions

@inline _is_hmm_dist(::Any) = false
@inline _is_hmm_dist(::DiscreteTimeDiscreteStatesHMM) = true
@inline _is_hmm_dist(::ContinuousTimeDiscreteStatesHMM) = true
@inline _is_hmm_dist(::MVDiscreteTimeDiscreteStatesHMM) = true
@inline _is_hmm_dist(::MVContinuousTimeDiscreteStatesHMM) = true

function _hmm_onehot_prior(n_states::Int, state::Int)
    probs = zeros(Float64, n_states)
    probs[state] = 1.0
    return Categorical(probs)
end

@inline _hmm_probs_to_categorical(probs) = Categorical(probs)

function _sanitize_hmm_probs(probs_in)
    probs = Float64.(probs_in)
    @inbounds for i in eachindex(probs)
        probs[i] = max(probs[i], 0.0)
    end
    s = sum(probs)
    (isfinite(s) && s > 0.0) || error("Invalid HMM state probabilities for simulation.")
    probs ./= s
    return probs
end

@inline function _hmm_with_initial_state(dist::DiscreteTimeDiscreteStatesHMM, state::Int)
    return DiscreteTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_onehot_prior(dist.n_states, state),
    )
end

@inline function _hmm_with_initial_state(dist::ContinuousTimeDiscreteStatesHMM, state::Int)
    return ContinuousTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_onehot_prior(dist.n_states, state),
        dist.Δt,
    )
end

@inline function _hmm_with_initial_state(dist::MVDiscreteTimeDiscreteStatesHMM, state::Int)
    return MVDiscreteTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_onehot_prior(dist.n_states, state),
    )
end

@inline function _hmm_with_initial_state(dist::MVContinuousTimeDiscreteStatesHMM, state::Int)
    return MVContinuousTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_onehot_prior(dist.n_states, state),
        dist.Δt,
    )
end

@inline function _hmm_with_initial_probs(dist::DiscreteTimeDiscreteStatesHMM, probs)
    return DiscreteTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_probs_to_categorical(probs),
    )
end

@inline function _hmm_with_initial_probs(dist::ContinuousTimeDiscreteStatesHMM, probs)
    return ContinuousTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_probs_to_categorical(probs),
        dist.Δt,
    )
end

@inline function _hmm_with_initial_probs(dist::MVDiscreteTimeDiscreteStatesHMM, probs)
    return MVDiscreteTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_probs_to_categorical(probs),
    )
end

@inline function _hmm_with_initial_probs(dist::MVContinuousTimeDiscreteStatesHMM, probs)
    return MVContinuousTimeDiscreteStatesHMM(
        dist.transition_matrix,
        dist.emission_dists,
        _hmm_probs_to_categorical(probs),
        dist.Δt,
    )
end

@inline _hmm_with_prior(dist, prior_probs) =
    prior_probs === nothing ? dist : _hmm_with_initial_probs(dist, prior_probs)

@inline _hmm_predicted_probs(dist, prior_probs=nothing) =
    probabilities_hidden_states(_hmm_with_prior(dist, prior_probs))

@inline _hmm_posterior_probs(dist, y, prior_probs=nothing) =
    posterior_hidden_states(_hmm_with_prior(dist, prior_probs), y)

@inline function _sample_hmm_hidden_state(rng::AbstractRNG, dist)
    probs = _sanitize_hmm_probs(probabilities_hidden_states(dist))
    return rand(rng, Categorical(probs))
end

@inline function _sample_hmm_hidden_state(rng::AbstractRNG, dist, prev_state::Int)
    next_dist = _hmm_with_initial_state(dist, prev_state)
    return _sample_hmm_hidden_state(rng, next_dist)
end

@inline _hmm_emission_rand(rng::AbstractRNG, dist::DiscreteTimeDiscreteStatesHMM, state::Int) =
    rand(rng, dist.emission_dists[state])

@inline _hmm_emission_rand(rng::AbstractRNG, dist::ContinuousTimeDiscreteStatesHMM, state::Int) =
    rand(rng, dist.emission_dists[state])

@inline _hmm_emission_rand(rng::AbstractRNG, dist::MVDiscreteTimeDiscreteStatesHMM, state::Int) =
    _mv_emission_rand(rng, dist.emission_dists[state])

@inline _hmm_emission_rand(rng::AbstractRNG, dist::MVContinuousTimeDiscreteStatesHMM, state::Int) =
    _mv_emission_rand(rng, dist.emission_dists[state])

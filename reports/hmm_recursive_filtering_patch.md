# HMM Recursive Filtering Patch

## Root Cause

The HMM estimation path had been reduced to a rowwise mixture likelihood.

At each observation row, the code rebuilt a fresh HMM distribution from
`calculate_formulas_obs(...)` and immediately evaluated `logpdf(dist, y)`.
That rebuilt the row-specific `Q` / `P`, emissions, and `dt`, but it did not
carry forward the posterior hidden-state distribution from the previous row.

As a result:

- row-specific covariates still affected the current-row HMM object,
- but the sequence dependence of the hidden state was lost,
- missing rows did not propagate hidden-state information correctly in
  likelihood-based estimation after the earlier rowwise-alignment change,
- SAEM HMM emission statistics were also computed rowwise rather than from the
  full sequence.

The failure mode is easiest to see with an upper-triangular transition model
and deterministic emissions:

- simulation with hidden-state carry-forward can reach later states,
- rowwise estimation can incorrectly treat those later states as impossible or
  underweight them because it keeps reusing the first-row prior.

## Code Changes

### Shared HMM helpers

Added shared internal helpers in
`src/distributions/outcomes/_HMMSimulationUtils.jl` for:

- identifying HMM distributions,
- rebuilding an HMM with a supplied hidden-state prior,
- obtaining predicted hidden-state probabilities from an injected prior,
- reusing the same machinery from both simulation and estimation.

### Likelihood recursion restored

Updated `src/estimation/common.jl` to maintain per-observable hidden-state
posteriors across rows:

- observed HMM rows:
  - build the row distribution,
  - replace its prior with the previous row posterior,
  - evaluate `logpdf`,
  - update the stored prior to `posterior_hidden_states(...)`,
- missing HMM rows:
  - propagate the prior with `probabilities_hidden_states(...)`,
  - contribute zero likelihood,
  - carry the propagated prior into the next row.

This restores recursive filtering for:

- `DiscreteTimeDiscreteStatesHMM`
- `ContinuousTimeDiscreteStatesHMM`
- `MVDiscreteTimeDiscreteStatesHMM`
- `MVContinuousTimeDiscreteStatesHMM`

### SAEM sequence smoothing restored

Updated `src/estimation/saem.jl`:

- `_saem_hmm_smoothed_gamma` now performs a forward-backward pass for discrete
  HMMs instead of using rowwise posteriors,
- the forward pass uses the recursive prior injection,
- the backward pass uses the row transition matrix from the next observation
  row,
- missing observations are handled as unit emission likelihoods and still
  propagate state information.

### AD-stable HMM logpdf

The recursive filtering patch also exposed a separate automatic-differentiation
problem in HMM `logpdf` for `Laplace` / `LaplaceMAP`.

The issue was not the recursion itself. It came from `Lux.logsumexp(...)` in
the HMM distribution implementations. Under nested `ForwardDiff` use in
Laplace-style estimation, `Lux.logsumexp` reduced a vector of tagged dual
numbers against an internal `Dual{Nothing}` seed, which triggered a dual-tag
ordering error.

That path was replaced with a small HMM-local `_hmm_logsumexp(...)` helper
that keeps the reduction in the active element type and avoids mixed-tag
promotion. This restores `Laplace` / `LaplaceMAP` compatibility for HMM
outcomes.

## New Regression Tests

Added or updated tests so the patch fails if the code regresses back to rowwise
mixture behavior:

- `test/hmm_discrete_time_tests.jl`
  - recursive filtering with deterministic upper-triangular transitions
  - missing-row propagation
- `test/hmm_continuous_tests.jl`
  - recursive filtering with deterministic emissions
  - missing-row propagation
- `test/hmm_mv_discrete_tests.jl`
  - recursive filtering for multivariate discrete HMMs
  - missing-row propagation
- `test/hmm_mv_continuous_tests.jl`
  - recursive filtering for multivariate continuous-time HMMs
  - missing-row propagation
- `test/hmm_estimation_method_matrix_tests.jl`
  - random-effects smoke matrix for `Laplace`, `LaplaceMAP`, `MCMC`, `VI`,
    `MCEM`, and `SAEM` across all four HMM families
  - continuous-time smoke models use bounded log-rates
- `test/estimation_saem_tests.jl`
  - SAEM HMM gamma regression against exact path enumeration
- existing HMM integration files now also cover `VI`

## Verification Summary

### Recursive filtering tests

Verified with:

- `julia --project=. test/hmm_discrete_time_tests.jl`
- `julia --project=. test/hmm_continuous_tests.jl`
- `julia --project=. test/hmm_mv_discrete_tests.jl`
- `julia --project=. test/hmm_mv_continuous_tests.jl`
- targeted SAEM HMM recursion check against exact enumeration

### Estimation front-end smoke matrix

Verified on March 12, 2026 with:

- `julia --project=. test/hmm_estimation_method_matrix_tests.jl`
- `julia --project=. test/hmm_discrete_time_tests.jl`
- `julia --project=. test/hmm_continuous_tests.jl`
- `julia --project=. test/hmm_mv_discrete_tests.jl`
- `julia --project=. test/hmm_mv_continuous_tests.jl`

Successful fixed-effects HMM methods:

- scalar discrete HMM: `MLE`, `MAP`, `MCMC`, `VI`
- scalar continuous-time HMM: `MLE`, `MAP`, `MCMC`, `VI`
- multivariate discrete HMM: `MLE`, `MAP`, `MCMC`, `VI`
- multivariate continuous-time HMM: `MLE`, `MAP`, `MCMC`, `VI`

Successful random-effects HMM methods across all four HMM families:

- `Laplace`
- `LaplaceMAP`
- `MCMC`
- `VI`
- `MCEM`
- `SAEM`

For continuous-time smoke coverage, the verification models clamp log-rates to
`[-1, 1]` before exponentiation so the generator matrix stays in a cheap,
well-behaved regime during short smoke runs.

Deprecated / out of scope:

- `FOCEI`
- `FOCEIMAP`

These are no longer part of the active verification target. They remain
outside HMM smoke coverage here.

## Current Status

After this patch:

- simulation carries hidden states forward,
- main likelihood-based estimation carries hidden states forward,
- SAEM discrete HMM emission stats use sequence-aware posteriors again,
- `Laplace` / `LaplaceMAP` run again for HMM outcomes after the AD-stability
  fix in HMM `logpdf`,
- all active HMM estimation methods are covered by smoke tests,
- deprecated `FOCEI` / `FOCEIMAP` are excluded from the active HMM method
  matrix.

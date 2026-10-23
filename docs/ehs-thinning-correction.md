# Elastic hard-scattering candidate acceptance

`sample_ehs_event` generates a candidate from an exponential grammage with
reference mean free path `lambda_ref`. Its acceptance probability must be the
ratio of event rate to reference rate:

`acceptance = (1/lambda_event)/(1/lambda_ref) = lambda_ref/lambda_event`.

Both Julia overloads previously inverted that ratio. For a mean free path
increasing with energy, this overaccepted candidates instead of thinning them.
The correction sets `r=lambda_event/lambda_ref` and retains acceptance `1/r`,
matching official [PUMAS step_transport, lines 6647–6651](https://github.com/niess/pumas/blob/529976ef51bd420f37ec63b1d3f615784d35b626/src/pumas.c#L6647)
(tag v1.2, library 1.2.3). The existing step-local candidate approximation and
angular scattering model are unchanged.

`test/test_ehs_thinning.jl` constructs a controlled material with reference mean
free path 2 times a material scale and event mean free path 3 times that scale.
An event must then survive with probability 2/3. Forced uniform draws 0.6 and
0.8 distinguish correct acceptance from the inverted formula; a uniform CDF
grid verifies the probability without sampling noise. Tests cover forward and
backward steps, the integer overload, a single-material mixture, a genuine
mass-fraction mixture, candidate distance/energy, and endpoint exclusion.

```bash
JULIA_NUM_THREADS=2 julia --project=. test/test_ehs_thinning.jl
```

This correction changes stochastic scattering. Fixed-path tomography now rejects
scattering because it cannot retrace geometry, so its energy-loss-only estimator
is unaffected. Complete boundary-aware transport still requires common-input C
comparisons; this local regression alone does not validate its flux.

> Applied after successful production source audits on 2026-10-23. The notes below describe the historical deferred review. Integrated package results and the computed-to-patched source transition are recorded in `../final_production_completion/`.

# Deferred optional shallow-mixture correction

The patch is **not applied**. It deletes only the shallow branch's pure-porous
shortcut, allowing its existing three-component mass recipe to operate at zero
water. This preserves the differentiable model and positive-water limit:
porous = rock = (1-w)/2, water = w, with additive specific volumes.
It does not implement fixed-porosity saturation or alter the ordinary branch.

`prepare.py` generates the patch and extracts the replacement function from the
patched text. `validate.jl` loads the current package, defines that replacement
under a separate name in its module, and keeps the original method available as
a negative control. No loaded tracked method is overwritten.

Validation on Julia 1.11.7 passed **73/73** checks: declared independent mixture
formulas for density, CSDA/mixed stopping, straggling and hard cross section at
zero, both sides of the old threshold, and representative wet endpoints; the
original dry mismatch; unchanged positive/disabled branches; constant-stopping
homogeneous-slab energy and one-sided numerical derivatives. Synthetic table
values are deliberately distinct and are not accuracy data. This is deterministic
recipe consistency, **not validation of stochastic transport**.

Command (CPU20, one CPU, 2GiB address-space bound):

```bash
python3 dev/run_bounded.py --cpus 1 --memory-gib 2 -- taskset -c 20 \
  julia --startup-file=no --project=. \
  /mnt/big_data/agent-builds/shallow-mixture-fix-20261023/validate.jl
```

`validation.log` records the successful run. Expected absent-dE/dx warnings
reflect deliberate synthetic fixture setup; the package's unrelated Kaleido
initialization may also warn under the memory bound. `status.json` and
`frozen-before.json` verify all 47 tracked source/test/Project/Manifest files
checked were unchanged. `git apply --check shallow-mixture.patch` passed.

After production source audits finish: apply the reviewed patch, move these
regressions into package tests, and run the full suite. Any enabled-layer
stochastic calibration additionally requires stochastic validation; the current
production's zero porous thickness leaves this optional branch inactive.

## Integrated regression patch prepared after the candidate run

`shallow-mixture-tests.patch` adds `test/test_shallow_mixture.jl` and one include
in the existing `test/runtests.jl` structure. Its 77 expected assertions call
the actual production helper, not the candidate alias. The old implementation
should fail its declared dry/threshold expectations. It preserves the ordinary
branch's existing tiny-water shortcut and independently checks all three
optional-branch disable guards. Julia syntax parsing and `git apply --check`
passed. **This integrated patch has not been applied or executed**; the 73/73
candidate result above must not be reported as a package-suite result for it.

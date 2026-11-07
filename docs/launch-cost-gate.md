# Launch cost gate

`sarek/execute/test/test_launch_cost.ml` runs in `dune test` (CI included) and fails when the
launch path of `Execute.run_vectors` gets slower for a reason that has nothing to do with the
kernel. It exists because #522 added a whole-IR fold to every launch, took 42 to 53 % off a
consumer decode (281 to 497 launches per token), and was found only by that consumer.
`test_launch_check_cost.ml` pins the two checks that regressed then; this gate covers the launch
path as a whole for the arms it measures (below), whichever check the next regression hides in.

## What it measures

N = 2000 launches through `Execute.run_vectors` on a probe JIT backend that does nothing at the
device (no GPU, no driver, no clock-dependent code), after a warming launch (which pays code
generation). The launches rotate over 4 distinct IRs of the same size, so the lookup is not served by
the per-domain last-lookup cache of `Ir_memo` alone. The three figures (small kernel, big kernel, reference loop) are timed in turn inside each of 7 rounds,
so that a stretch of load reaches all of them, and the minimum over the rounds is kept for each,
which filters slow outliers (a preempted round) but can also pick an unusually fast one, so it can
understate a cost; both assertions compare figures taken the same way, which cancels the bias.
Two kernels per case: 1 statement and 20000 statements, with and without a warp shuffle (the
shuffle turns the launch rule on).

Two probe backends run the same code, because the path depends on the backend name:

- `LaunchCostProbe`: the immediate launch path.
- `HIP` (the real HIP backend is not linked into this executable): the path of a GPU. The launch
  is deferred by the launch window (`Window_gate.default_backends`, flushed inside the timed
  region) and the matrix-unit rewrite is tried (`Matrix_specialise.device_possible`). A fold
  placed in `Matrix_specialise.decide` after `device_possible` is only reachable this way: a
  probe on the first name alone passes it (measured, see below).

Not covered: a real driver, vector arguments and transfers (the only argument is one scalar),
stream launches, SoA arguments, epilogue fusion. A regression there needs its own test. (The plain
kernel on the `HIP` probe is horizontally fused by the launch window, up to 8 launches per backend
issue; the shuffle kernel is never fused.)

## Two assertions

1. **IR-size independence**: a launch of the 20000-statement kernel costs at most 25 times a
   launch of the 1-statement kernel. A launch may depend on the IR only through a cache lookup.
   Anything that folds the IR at each launch is hundreds to thousands of times slower on the big
   kernel.
2. **Fixed cost**: a launch costs at most 8 units, where a unit is a reference loop timed in the
   same process (8 hash-table lookups and one formatted string). This catches a constant added to
   every launch, which assertion 1 cannot see. The ratio makes it largely independent of the
   machine, not strictly: the units figure was measured on Apple M4 only.

The runner is verbose, so the `launch-cost` lines of a passing run are in the CI log (alcotest
otherwise keeps a passing test's output in a file). Read the first CI log after a change of
runner image and recalibrate if the units figure is above 3.

## Calibration

Apple M4 (10 cores), OCaml 5.4.0, dune build in the default (dev) profile, 2026-10-18, commit
`f12ac093` plus the test.

| Condition (final version of the test) | Runs | Big/small ratio | Small launch (units) |
|---|---|---|---|
| Idle machine, 4 cases | 5 | 1.06 to 1.12 | 0.7 to 1.3 |
| 24 busy loops on 10 cores, 3 runs in parallel, 4 cases | 30 runs (120 cases), all pass | 1.02 to 1.15 | at most 1.4 |
| `Execute` and `Matrix_specialise` reverted to the #522 code (`dc6f27d6`) | 2 | 830 to 1540 | 0.7 to 1.6 |
| Raw `kernel_uses_shuffle` fold in each launch | 2 | 830 to 1510 | 0.8 to 1.4 |
| Raw warp-op fold in `Matrix_specialise.decide` only | 2 | 550 to 590 on the `HIP` probe, 1.1 on the immediate probe (not detected) | 0.7 to 1.3 |
| Fixed cost of 6.6 us added to every launch (the size of the #522 downstream decode loss, 6.4 us) | 1 | 1.0 | 12.0 to 12.9 |

The size limit (25) sits more than 20 times above the worst healthy ratio and over 20 times under
the lowest regression. The fixed-cost limit (8) sits about 5 times above the worst healthy value
under load and below the 12 to 13 units a #522-sized constant costs. The earlier limit of 30 units
let that constant through. `test_gate_sees_a_whole_ir_fold` fails if a fold per launch ever stops
exceeding the size limit; no in-process self-test covers the fixed-cost limit, so a change to it
must be justified by a mutation run like the last row.

## When it fails

Read the `launch-cost` line. A big/small ratio far above 1 means something walks the IR at each
launch: record its verdict once per IR (a slot of `Ir_memo`). A small launch above 8 units means
a fixed cost was added to the path (allocation, a lock, a hash of the arguments); measure it before
accepting a limit change, and update the calibration table above.

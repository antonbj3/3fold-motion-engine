# Bounded RT body-pair experiment

The baseline is unchanged `contact_body_pairs_gpu.py`. Before candidate design,
`probes/body_pair_scale_probe.py` measures full rebuilds at10000 and100000 bodies.
Each synthetic two-layer fixture uses the original18-point box cloud, horizontal
spacing0.35, lower center0.0999 and vertical spacing0.1999. These are contacted
geometry fixtures, not integrated trajectories. The expected total is N pairs:
one ground pair per lower body and one vertical pair per upper body.

Preregistered baseline gates: exact sorted pairs against an independent CPU
spatial-tree enumeration using the actual conservative device bounds; complete
array bytes identical in two isolated workers with reversed fixture order;
nonzero expected ground/body pairs and exact expected counts; finite outputs
and positive synchronized timings for both sizes. Each timing includes three
warm rebuilds followed by twenty full rebuilds; setup is excluded. CPU oracle
and copies for verification are outside timing. No speed threshold applies to
this mechanism observer. The baseline's existing spatial/capacity guards remain
unchanged. Warp1.13.0 and NumPy2.5.3 are pinned.

At baseline preregistration the RT candidate was not yet designed or measured. A candidate comparison
must include acceleration-structure rebuild and canonical pair generation,
require exact baseline pairs and independent repeat bytes, and retain any
speed failure. Trace-only timing cannot establish a full-rebuild win. The
existing100-step engine profile already spends about0.997ms in solving and
0.847ms in sort/select, so a broad-phase result alone cannot establish a
submillisecond full step.

## Baseline measurement before candidate design

Modal L4, driver580.95.05: all4/4 baseline gates pass.

| Bodies | Ground pairs | Body pairs | Full rebuild ms, run1/run2 | Exact pair oracle |
|---|---|---|---|---|
| 10000 | 5041 | 4959 | 0.492190100/0.461520300 | pass |
| 100000 | 50176 | 49824 | 0.762048650/0.768368850 | pass |

Both runs retain identical complete input, conservative-bound and pair arrays.
The sphere-tree candidate counts44302/447556 are oracle enumeration costs,
not GPU hash-grid visit counts. No RT candidate has run at this checkpoint.

## Candidate design and gates

`contact_body_pairs_rt.py` keeps the baseline bounds and acceptance predicate.
Each custom primitive indexes its original double bounds expanded by the
validated maximum query-body extent0.35. Therefore every true overlap contains
the other body's rounded center inside the expanded box. A short vertical ray
through that center requests candidate primitives; the custom intersection
program applies the unchanged six inclusive double comparisons and `b>a`.
Ground pairs use the unchanged strict lower-z comparison. Float acceleration
bounds receive64eps times coordinate scale padding; this covers conversion and
ray-origin subtraction rounding under the existing coordinate bound10000.
Padding only adds traversal candidates, never accepted pairs.

The GPU creates acceleration bounds, OptiX rebuilds its BVH on each call, and
per-ray bounded slots collect hits. Single-any-hit geometry flags prevent
repeated primitive visits from emitting duplicates. Counts are scanned, keys
sorted and decoded using the baseline format. Per-ray or total overflow must
refuse and leave the public count zero. The Python wrapper owns native lifetime
and uses its current Warp stream; native calls finish before returning.

Preregistered candidate gates: exact pair/bound bytes versus unchanged baseline
on both large fixtures and directed boundary/empty controls; exact independent
repeat arrays; all spatial, layout, capacity, per-ray and closed-handle refusals;
positive finite full-rebuild times; full-rebuild improvement in both workers at
both sizes. The last gate may fail without invalidating the numerical results.
Each timed path uses3warm+20measured rebuilds with order reversed in worker2.
Setup and oracle copies remain separate. No dynamics or full-step speed claim.

The [Mochi paper](https://arxiv.org/html/2604.23520v1) motivates RT traversal for
collision search but targets spherical particles and includes a different proxy
construction. This bounded AABB experiment is not a reproduction of its method
or published speedups. The [OptiX API](https://raytracing-docs.nvidia.com/optix8/api/group__optix__types.html)
defines the single-any-hit flag used here. The official external OptiX9.0.0
headers are pinned tofff65c2a7c592f1ea5f1661ad7d2381cf965f9bd; no SDK is vendored.

## Reproduction

Set `OPTIX_INCLUDE` to the externally installed pinned headers. With CUDA12.9
and a host C++17 compiler available, run:

```sh
python scripts/build_body_pairs_rt.py artifacts/body_pairs_rt
export MOTION_RT_LIBRARY="$PWD/artifacts/body_pairs_rt/pairs.so"
export MOTION_RT_PTX="$PWD/artifacts/body_pairs_rt/pairs.ptx"
python probes/body_pair_rt_probe.py
```

The build targets compute89 for the L4 experiment. Compiling requires no GPU
context. The runtime is an internal device-pointer adapter called only through
its owning Python wrapper; arbitrary external pointers are outside this API
contract. Context-manager exit releases native storage and pipeline state;
close is idempotent. Public pairs and conservative bounds are the deterministic
outputs; internal traversal slot order is not a public output.

## Full rebuild result

OWN-GATE-FAIL4/5 onL4: all exactness and11refusal controls pass; speed fails.

| Bodies | Baseline ms, run1/run2 | RT ms, run1/run2 |
|---|---|---|
| 10000 | 0.471293350/0.460960200 | 0.865792150/0.870069000 |
| 100000 | 0.775272200/0.739300500 | 1.367434050/1.378639050 |

No solver adopts this candidate. The near-contact fixture is not an exact
acceptance-boundary proof: baseline padding shifts that boundary. Before closing
the candidate, a supplemental probe will test exactly2r and adjacent float32
values on all three axes, plus exactlyr and adjacent ground-height values,
where r is the actual padded radius at coordinate scale1. Expected pairs are
explicit, not inferred from RT. Two independent complete captures must match;
a separate memory-check run must preserve them and report zero errors.

The supplemental boundary probe passes3/3 onL4:12explicit cases in each
independent worker. Plain and memory-check full JSON and120arrays/3224bytes
are identical. The memory-check command uses target-processes all and error
exit77; it exits0 with one visible zero-error summary. Child stdout is captured,
so separate child summaries are not retained. This checks the small directed
cases, not a memory-instrumented100000-body workload. The original speed
failure remains4/5 and the candidate is not adopted.

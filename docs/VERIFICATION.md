# Explicit motion verification

## Baseline mechanism before design

| Measurement | Count |
|---|---|
| Current VERIFIED-FRESH declarations | 160 |
| Sources with a main-guard candidate | 129 |
| Libraries or test sources requiring an explicit runner | 31 |
| Explicit verification recipes | 0 |
| Existing make verify target | absent |

Source entry points do not prove that every required numerical gate passes.
The inventory executes no source. Unmapped declarations are identified by
RUNNING line and declaration hash rather than copying legacy fixture labels.

## Preregistered contract

Add an explicit recipe manifest and isolated runner beside the numerical
baselines. Each recipe binds declared source rows and their exact status notes,
an aggregate source-tree digest, complete expected report bytes and exact
named boolean gates. Zero exit alone is insufficient. CPU recipes hide CUDA
and set a per-child Haswell OpenBLAS profile; this does not change global
settings or certify arbitrary CPU dispatch.

A fresh temporary repository copy is used for each invocation. Expected output
reports are removed before a child runs, timeouts kill its process group, and
all requested output files must be recreated with the exact expected hashes.
No baseline receipt is overwritten. Source/note pins are checked before any
child. Full make verify refuses before execution while any fresh declaration
is unmapped; an explicit selected target may pass only its listed subset.

Own contract gates are strict status parsing, duplicate/source/note refusal,
manifest and path validation, exact typed nonempty gate maps, full-report hash
checks, stale-output refusal, process-exit refusal and independent repeat
receipts. Controls must include deliberately false and omitted gates. Full
coverage is a separate gate and remains a failure until every current fresh
row is explicitly covered. Numeric limits will not be widened for replay.

## Selected CPU slice

The current manifest has eight commands covering 10 of 161 fresh declarations.
Selected verification passes 8/8 in two independent local invocations; complete
verifier JSON is byte-identical. Full coverage remains OWN-GATE-FAIL 10/161,
with 151 unmapped declarations and zero children launched by the complete target.

```sh
make verify-declared PYTHON=python
make verify PYTHON=python
```

Use the pinned environment: Warp1.13.0, NumPy2.5.3 and SciPy1.18.1. The initial six
recipes exercise static normal coupling, hybrid refusal controls, the rational
rounding model, scalar and phase attribution, and this verifier's own contract.
The static normal recipe covers its library and its dedicated probe; contract
controls likewise cover the verifier library and probe. GPU engine declarations
are not counted merely because the refusal probe imports their classes.
Archived device comparison inputs are pinned but are not fresh GPU execution.

The source digest covers `src`, `scripts`, `probes`, `tests`, `examples` and
`Makefile`. The input digest covers reports other than declared output reports
and generated `verification_` receipts. The manifest records every selected
status-note hash and full expected JSON hash. Package binaries are not part of
these source pins; full output hashes reject numerical changes under the
explicit per-child CPU profile. A source or selected note change requires
reviewed pins and fresh execution, not automatic acceptance of new outputs.

The contract probe passes8/8 in two independent workers, including24deliberate
invalid-input/report/process controls plus positive, complete-coverage and
repeated-isolation cases. A zero-exit child writing a false gate is rejected;
its entire failed report is retained beside the baseline. Timeouts and nonzero
exits also remain failures even if a valid-looking report exists. Failed fresh
reports use a generated `verification_failed_` name and never overwrite their
reference. The initial snapshot omission is retained in
`reports/verification_initial_refusal.json`; the corrected example-source
regression is included in the isolation control.

The same six-command slice passes twice on ModalL4 with CPU children. Complete
selected-verifier JSON is byte-identical locally, across both cloud invocations,
and across hosts. Full-coverage refusal JSON also matches exactly. The cloud
command ends with the expected make exit2 for154unmapped rows; this is not a
successful full verification. See `reports/verification_six_host_parity.json`.
The first runtime package installation was stopped before numerical execution;
baking the identical pins into the image installed them in11.93s. The retained
failed setup receipt is not counted as a numerical run.


The extension adds the retained position-solve matrix audit and the complete
axis-filter capture audit. Both use their existing exact report hashes and
four named boolean gates, without numerical source changes. These are archived
input replays, not new device captures.

The NNLS engine was incorrectly marked fresh despite its own retained receipt
passing only 3/4 gates. K16 penetration is 1.748346388340 against the unchanged
0.5 limit, with minimum spacing 0.117582678795 below 0.164. Its row now reads
OWN-GATE-FAIL; neither source nor numerical receipt changed. This correction
reduces the fresh denominator from 162 to 161 independently of recipe coverage.
See `reports/verification_nnls_status_audit.json`. Historical six-command
receipts retain their original denominator in `reports/verification_six_*.json`.


The extended eight-command Modal attempt did not reach an observed numerical
child within the 600-second setup bound. The service reported zero running
containers; the owned app was stopped. The delay is not attributed uniquely to
upload or allocation. `reports/verification_extended_cloud_setup.json` retains
this failure. The extended slice therefore has two exact local runs only;
its cross-host verification remains open. The prior six-command cross-host
result does not certify the additional recipes.

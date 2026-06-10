# Replay

The `mneme replay` command re-executes a previously recorded GPU kernel
in isolation, using the artifacts produced during recording.
For full CLI details, see [CLI → mneme replay](cli.md#mneme-replay).

## Verification

After executing the kernel, Mneme compares the resulting device memory
state against the recorded **epilogue** snapshot.  The replay result
reports `"verified": true` when the comparison passes for every
recorded memory blob.

By default the comparison is **byte-exact**.  If the application
annotated device buffers with `mneme::annotate()` before the recorded
launch, Mneme uses the stored metadata (data type, threshold, norm)
instead.  This allows replay to accept numerically equivalent results
that differ at the bit level — essential when exploring alternative
optimization pipelines or launch configurations.

For a detailed explanation of the annotation API, metadata fields,
threshold semantics, and a complete example, see
**[Usage → Verification](verification.md)**.

## Reset mode

Replay restores the recorded prologue before each measured kernel run.
When the recording has a diff epilogue, Mneme can avoid copying the
entire prologue by restoring only the prologue byte ranges that differ
in the epilogue:

```bash
mneme replay ... --reset-mode diff "default<O3>"
```

Supported modes are:

| Mode | Behavior |
| ---- | -------- |
| `bytes` | Always use the original full prologue reset. |
| `diff` | Require diff reset and fail if the epilogue is not a valid diff snapshot. Diff reset restores ranges with one device scatter kernel per warm reset. |

When no reset mode is supplied, Mneme selects `diff` for diff epilogue
snapshots and `bytes` otherwise. Diff reset uses raw ranges by default;
tune optional coalescing with
`MNEME_REPLAY_DIFF_SCATTER_MAX_GAP_BYTES` and chunk size with
`MNEME_REPLAY_DIFF_SCATTER_TASK_BYTES`.

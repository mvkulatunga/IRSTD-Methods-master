# YOLO YAML — Layer & Module Args

Every layer is `[from, repeats, module, args]`:

- **from** — input source by layer index. `-1` = previous layer; an int (e.g. `17`) = that layer; a list (e.g. `[-1, 6]`) = multiple inputs (used by `Concat`).
- **repeats** — how many times the module is stacked; scaled by the `depth` multiplier.
- **module** — the layer type.
- **args** — passed positionally to the module constructor *after* the auto-inferred input channels.

## Scales

`n: [depth, width, max_channels] = [0.50, 0.25, 1024]`

- `depth` multiplies **repeats**.
- `width` multiplies every **c_out**, then capped at `max_channels`.
- Numbers in the YAML are full-size values; the scale shrinks them (e.g. `[1024]` → 256 channels at `n`).

## Modules

- **`Conv, [c_out, k, s]`** — Conv-BN-act. Channels, kernel, stride. `s=2` halves spatial size (downsample); `s=1` keeps it.
- **`C3k2, [c_out, c3k, e]`** — CSP feature block. `c3k` (`True`/`False`) picks the heavier vs. cheaper bottleneck; `e` = expansion ratio (hidden width fraction).
- **`A2C2f, [c_out, a2, area]`** — YOLOv12's R-ELAN aggregation block (residual ELAN) with optional Area Attention. `a2` (`True`/`False`) toggles area-attention; `area` = number of attention regions. `-1` = use default.
- **`CoordAtt, [c]`** — Coordinate Attention on `c` channels (built directly from this number).
- **`nn.Upsample, [size, scale, mode]`** — e.g. `[None, 2, "nearest"]` = 2× upsample, nearest-neighbor. No channel arg.
- **`Concat, [d]`** — join inputs along dim `d` (`1` = channels). Output channels = sum of inputs.
- **`Detect, [nc]`** — detection head array for `nc` classes; builds one head per feature layer listed in `from`.

## Notes

- Input channels are always inferred — never written.
- Flags map to named constructor params (`c3k`, `a2`, `area`, `shortcut`); `-1` means "use default."
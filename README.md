# SwarmUI-KreaGoneWild

A SwarmUI extension with a small toolbox of Krea 2 tricks. Everything is gated on the selected
model being detected as **Krea 2** (`krea-2` / `krea-2/lora`) and does nothing otherwise.

Three features, each living in its own nested sub-group under the **Krea Gone Wild** parameter
group. Each sub-group's **toggle is the feature's on/off switch** — there is no separate enable
param. Install buttons appear inside the sub-group of whichever required node pack is missing.
KGW Rebalance requires nova452's pack. True renormalize is separate: it needs a node that provides
class `Krea2Rebalance`, and that node has no install button. See [Rebalance-only extract](#rebalance-only-extract).

## 1. KGW Rebalance

Reweights the 12 Qwen3-VL taps packed into Krea 2 conditioning, on the positive and negative
prompts (and the refiner pass, if present). The group requires
[nova452/ComfyUI-ConditioningKrea2Rebalance](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance).

True renormalize needs a backend node that provides class `Krea2Rebalance`, with inputs
`conditioning`, `preset`, `per_layer_weights`, `multiplier`, and `renormalize`. The recommended
source is a rebalance-only extract of RedNodeAI's `rebalance.py` in `custom_nodes` (no core
patches). Installing the full
[RedNodeAI/ComfyUI-Krea2Moodboard](https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard) pack also
provides that class, but loading it monkeypatches core Krea 2 functions globally: it replaces
`SingleStreamDiT._forward`, `Krea2` `model_base.extra_conds` and `extra_conds_shapes`,
`Krea2TEModel.encode_token_weights`, and `Qwen3VL.preprocess_embed`. Those patches apply as soon
as the pack loads.

**KGW per_layer_weights** sets the ratio between taps. **KGW Renormalize** (on by default) then
holds the conditioning magnitude, and **KGW Multiplier** is a gain on top of that (1 holds the
magnitude). Pooled output is left as-is. The RMS is one scale per batch row, over every remaining
dimension (tokens and taps together). It is not per-token and not per-layer.

What gets emitted:

- **Renormalize on, and `Krea2Rebalance` is on the backend.** Class `Krea2Rebalance` with
  `preset` `custom`, your per-layer weights, your multiplier, and `renormalize` true. The node
  captures each batch row's RMS before the weights, applies the per-tap gains, rescales to that
  captured RMS, then multiplies by **KGW Multiplier**. A stored multiplier M is M times the
  original conditioning RMS.
- **Renormalize off.** nova452's `ConditioningKrea2Rebalance` with the multiplier passed through
  unchanged, the same way it ran before renormalize existed. The weights and the multiplier both
  amplify.
- **Renormalize on, and `Krea2Rebalance` is missing.** Generation still runs. The extension falls
  back to nova452's node and folds in the equal-energy estimate `k = sqrt(mean(weight²))` as
  `multiplier / k`. That equals a measured RMS renormalize when the taps have the same mean
  square. When they do not, `k` is only an approximation, because the real ratio depends on the
  prompt. The first time this fallback runs, a log warning points here, at
  [Rebalance-only extract](#rebalance-only-extract).

The rebalance group stays gated on nova452's node, so renormalize-off and the fallback both work
without `Krea2Rebalance`. nova452's install button does not block generation.

- Toggle the **KGW Rebalance** group on to enable it.
- **KGW Multiplier** — gain on the original RMS when renormalize is on and a `Krea2Rebalance`
  node is on the backend (default **1**). When renormalize is off, this is the original global
  multiplier and it stacks with the per-layer weights.
- **KGW Renormalize** — on by default.
- **KGW per_layer_weights** — comma-separated weights for the 12 conditioning layers.
  Default `1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0`.

### Multiplier default and existing presets

The registered default of **KGW Multiplier** is **1**. Swarm stores a preset value under
`kgwmultiplier` only when that preset sets it. A preset that explicitly sets 2 or 4 still sends
2 or 4; the registered default does not rewrite those stored numbers. The default 1 applies when
the param is absent. Old presets have no `kgwrenormalize` entry, so the default **on** applies.

With the default weights and equal tap energy, `k = sqrt(56.46 / 12) ≈ 2.1691`. nova452's RMS
gain is `multiplier × k`. With true renormalize on, the gain is the stored multiplier itself.

| Stored multiplier | Old RMS gain (renormalize off) | RMS gain with true renormalize on |
| --- | --- | --- |
| 4 | 4 × 2.1691 ≈ **8.68×** | **4×** the original RMS |
| 2 | 2 × 2.1691 ≈ **4.34×** | **2×** the original RMS |
| 1 (default) | 1 × 2.1691 ≈ **2.17×** | **1×** the original RMS |

The per-layer ratios are the same in both columns. Only the overall magnitude changes. The
fallback (renormalize on, `Krea2Rebalance` missing) aims at the right-hand column when the taps
have equal energy, and only approximates it when they do not.

- Clean renormalized look: renormalize on, multiplier **1**. That holds the measured RMS.
- Match the old equal-energy magnitude while leaving true renormalize on: multiplier **≈ 8.68**
  where the preset stored 4 (8.676× the original RMS), or **≈ 4.34** where it stored 2 (4.338×).
  That is a real amplification. Values above 1 can oversaturate.
- Match the old node exactly, including unequal tap energies: renormalize **off**, multiplier
  **4** or **2** as the preset already stores.

### Rebalance-only extract

True renormalize needs class `Krea2Rebalance` on the ComfyUI backend. SwarmUI detects that class
as feature `krea2_rednode_rebalance`. There is no install button for it.

1. In the ComfyUI install this SwarmUI uses, make a folder under `custom_nodes`. The name is
   yours. `ComfyUI-Krea2Rebalance-RedNode` is a fine one.
2. Copy `rebalance.py` from
   [RedNodeAI/ComfyUI-Krea2Moodboard](https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard) at
   commit [`ffa2cd4`](https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard/blob/ffa2cd4327f02a31105291b93a711d43e2b86c48/rebalance.py)
   into that folder. Copy the file as-is.
3. Add `__init__.py` next to it, exporting only that class:

```python
from .rebalance import Krea2Rebalance

NODE_CLASS_MAPPINGS = {"Krea2Rebalance": Krea2Rebalance}
```

Restart the ComfyUI backend so SwarmUI sees the node. This folder does not patch core Krea 2.
Do not install the full RedNodeAI pack unless you want those global patches.

`rebalance.py` is RedNodeAI's file and stays under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html).
The mechanic it reimplements is Apache-2.0, from
[nova452/ComfyUI-ConditioningKrea2Rebalance](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance)
and the RMS-renormalized variant in
[huwhitememes/comfyui-krea2-conditioning](https://github.com/huwhitememes/comfyui-krea2-conditioning).

## 2. KGW Ostris Edit

Runs [ComfyUI-Krea2-Ostris-Edit](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit) edit LoRAs
(ai-toolkit `krea2` arch, `model_kwargs.edit: true`) — "Kontext-style" reference-image editing.

It wires the edit natively into SwarmUI rather than replacing SwarmUI's conditioning:

1. Your prompt image(s) are scaled to ~1 MP (÷16 snapped, matching the training preprocessing) and
   VAE-encoded, then attached as **reference latents** onto the positive and negative conditioning via
   the core `ReferenceLatent` node.
2. The model is wrapped in `Krea2OstrisEditModelPatch` so the Krea 2 DiT actually consumes those
   reference latents (`index_timestep_zero`).

Because SwarmUI's own `SwarmClipTextEncodeAdvanced` still builds the conditioning, all the advanced
prompt syntax (`<break>`, weighting, `[from:to:when]`, regions, …) keeps working on edits. Only the
model patch comes from the ostris pack; the reference latents use core ComfyUI nodes — exactly the
"chain conditioning from … Set Reference Latent nodes" path the pack documents.

- Toggle the **KGW Ostris Edit** group on to enable it. Requires a Krea 2 **edit LoRA** loaded and at
  least one prompt image. With no prompt image it is a no-op (the model patch passes through unchanged).
- **KGW Edit Reference Megapixels** — target size the reference is scaled to before VAE-encoding
  (default 1.0, matching ai-toolkit).

> Note: the reference latents are attached to the main generation pass only; a separate refiner pass
> is not edit-patched.

## 3. KGW Krea2Edit

Runs [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) — instruction-based editing
with the `krea2_edit` identity LoRA. Different approach from the ostris pack: instead of reference
latents on the conditioning, the source is prepended to the sampling sequence as a block of clean
in-context tokens (RoPE frame=1), with the instruction encoded through Qwen3-VL alongside the image in
the exact layout the LoRA trained on.

Two halves to the recipe. Only the model patch comes from the pack:

1. **Grounding.** SwarmUI already routes Krea 2 through `SwarmClipTextEncodeAdvanced` with the prompt
   images wired and its stock `"krea2"` template, which appends the vision blocks *after* the
   instruction with a `Picture N: ` label. krea2_edit trained with them *before* the instruction and
   unlabelled. That node passes any non-magic `llama_template` string straight through to
   `clip.tokenize` and leaves the prompt text alone, so this extension just retemplates the existing
   positive and negative encoders in place. Result: the pack's grounded encode, with `<break>`,
   `[from:to:when]`, `[alter|nate]`, weighting and per-step scheduling all still working — none of
   which the pack's own `Krea2EditGroundedEncode` node supports.
2. **In-context source.** `Krea2EditModelPatch` wraps the model. It gets the raw prompt image(s) plus
   the VAE on the pixel path (immune to input/output resolution mismatches), the VAE-encoded source as
   the required `source_latent`, and the sampler's own latent as `target_latent` so the source encode
   happens *before* sampling instead of evicting the diffusion model mid-run.

Up to two prompt images feed the model patch: the first is the scene (`ref_boost_a`), the second the
subject (`ref_boost`) — the training-matched order for the multi-reference LoRAs. Grounding sees every
prompt image SwarmUI batched, one vision block each.

- Toggle the **KGW Krea2Edit** group on to enable it. Needs the `krea2_edit` LoRA and at least one
  prompt image; with no prompt image it is a no-op.
- **KGW Krea2Edit Ref Boost** — reference-fidelity dial on the *last* reference. 1.0 = off, above 1
  pulls harder toward the reference's appearance. Optimal value is model-specific.
- **KGW Krea2Edit Ref Boost A** *(advanced)* — same dial for the *first* reference (the scene). No
  effect with a single prompt image.
- **KGW Krea2Edit Fit Mode** *(advanced)* — `fit` (training-matched, default) or `crop (legacy)` for
  older weights.
- **KGW Krea2Edit Grounded Encode** — on by default. Turn it off to keep SwarmUI's stock `"krea2"`
  template (still grounded, just with the training-mismatched vision-block placement).
- **KGW Krea2Edit Grounding Pixels** — caps the longest side of the images fed to the vision tower,
  via a core `ImageScaleToMaxDimension` node. 0 = leave SwarmUI's own prompt-image sizing alone.
  The LoRA trained with 384–768px jitter, so 640–768 is in-distribution.
- **KGW Krea2Edit System Prompt** *(advanced)* — replaces the system turn of the grounding template.
  Empty = krea2_edit's training default.

> Note: the pack's own guidance — Turbo at 8 steps / CFG 1 for most edits, Raw at CFG 3 / 20 steps
> for removals, generate at ≤2 MP.

## Usage

1. Install the required ComfyUI node pack using the button in the matching sub-group, then restart
   the ComfyUI backend when prompted. For true renormalize, add the
   [rebalance-only extract](#rebalance-only-extract) instead of installing the full RedNodeAI pack.
2. Select a Krea 2 model.
3. Toggle on the sub-group(s) you want (and, for either editing mode, add a prompt image + the
   matching edit LoRA).

The features are independent toggles. The two editing modes are alternative approaches to the same
job — enable one, not both.

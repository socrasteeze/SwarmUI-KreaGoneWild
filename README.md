# SwarmUI-KreaGoneWild

A SwarmUI extension with a small toolbox of Krea 2 tricks. Everything is gated on the selected
model being detected as **Krea 2** (`krea-2` / `krea-2/lora`) and does nothing otherwise.

Three features, each backed by a third-party ComfyUI node pack and each living in its own nested
sub-group under the **Krea Gone Wild** parameter group. Each sub-group's **toggle is the feature's
on/off switch** — there is no separate enable param. Install buttons appear inside the sub-group of
whichever node pack is missing.

## 1. KGW Rebalance

Applies [ComfyUI-ConditioningKrea2Rebalance](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance)
to both the positive and negative Krea 2 conditioning (and the refiner pass, if present).

Krea 2 packs 12 Qwen3-VL layer taps into one conditioning tensor. **KGW per_layer_weights**
changes the ratio between those taps. **KGW Renormalize** (on by default) then rescales so the
conditioning RMS matches the input RMS, and **KGW Multiplier** is a gain on top of that
(1 holds the magnitude). Pooled output is left as-is.

The RMS in the reference nodes (huwhitememes `ConditioningKrea2Rebalance`, RedNodeAI
`Krea2Rebalance`) is one scale per batch row, taken over every remaining dimension — all
tokens and all taps together. It is not per-token and not per-layer. Both leave masks and
pooled output untouched. This extension follows that definition.

The installed nova452 node has no `renormalize` input (upstream `Rebalance-Pack` does not
either; the open PR that would add it is unmerged). Core ComfyUI cannot compute the ratio:
`ConditioningMultiply` is a constant, and it also scales pooled output, which the reference
math does not. huwhitememes uses the same class name as nova452, so depending on it would
collide if both packs are installed. RedNodeAI's `Krea2Rebalance` does not collide, but it
is a different pack and is not required here.

So Renormalize stays on the already-required nova452 node. The scale is the equal-energy
estimate `k = sqrt(mean(weight²))`, sent as `multiplier / k`. That equals a measured RMS
renormalize when the taps have the same mean square. When they do not, `k` is only an
approximation — the real ratio depends on the prompt. A log warning says so the first time
a renormalized rebalance runs. Turn **KGW Renormalize** off to run the original node unchanged.

- Toggle the **KGW Rebalance** group on to enable it.
- **KGW Multiplier** — post-renorm gain when renormalize is on (default **1**). When renormalize
  is off, this is the original global multiplier and it stacks with the per-layer weights.
- **KGW Renormalize** — on by default.
- **KGW per_layer_weights** — comma-separated weights for the 12 conditioning layers.
  Default `1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0`.

### Multiplier default and existing presets

The registered default of **KGW Multiplier** changed from 4 to 1. Swarm stores a preset value
under `kgwmultiplier` only when that preset sets it. A preset that explicitly sets 2 or 4 still
sends 2 or 4; changing the registered default does not rewrite those stored numbers. The new
default applies when the param is absent (a fresh UI, or a preset that never stored it because
4 used to be the default — that case now resolves to 1). Old presets have no `kgwrenormalize`
entry, so the new default **on** applies.

With the default weights, equal tap energy, and renormalize on, `k = sqrt(56.46 / 12) ≈ 2.169`.
The old node's RMS gain was `multiplier × k`.

| Stored multiplier | Old RMS gain (renormalize off) | RMS gain with renormalize on |
| --- | --- | --- |
| 4 | 4 × 2.169 ≈ **8.68×** | **4×** |
| 2 | 2 × 2.169 ≈ **4.34×** | **2×** |
| 1 (new default) | 1 × 2.169 ≈ **2.17×** | **1×** |

The per-layer ratios are the same in both columns. Only the overall magnitude changes.

- Clean renormalized look: renormalize on, multiplier **1**.
- Approximate the old magnitude while leaving renormalize on: multiplier **≈ 8.68** where the
  preset was 4, or **≈ 4.34** where it was 2. This restores the equal-energy RMS gain. Real
  prompts can sit off that number because tap energies are not equal.
- Match the old node exactly, including unequal tap energies: renormalize **off**, multiplier
  **4** or **2** as the preset already stores.

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

1. Install the required ComfyUI node(s) using the button in the matching sub-group, then restart the
   ComfyUI backend when prompted.
2. Select a Krea 2 model.
3. Toggle on the sub-group(s) you want (and, for either editing mode, add a prompt image + the
   matching edit LoRA).

The features are independent toggles. The two editing modes are alternative approaches to the same
job — enable one, not both.

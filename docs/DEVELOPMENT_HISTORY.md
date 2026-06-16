# Development History (VLCFusion)

> Not part of the user-facing README. This records how the fusion design evolved and what
> each experimental variant was, so the released code can stay lean. Everything removed
> during the release cleanup is **recoverable** from the git tag `pre-release-archive`
> (commit `89f3e73`): `git checkout pre-release-archive -- "<path>"`.

## Final method

**VLCFusion** is the design previously prototyped as **CrossCBAM_DiT_V11**:

```
concat(F1, F2) -> MultiHeadFusionBottleneck (1x1/3x3/5x5 input-adaptive branches)
              -> VLC block -> VLC block -> fused features
```
Each VLC block: `GroupNorm -> FiLM(VLM conditions) -> CBAM -> residual`, then a
FiLM-conditioned FFN. The number of stacked VLC blocks is configurable (paper default **2**;
see the block-count ablation). Matches paper §3.2 / Fig. 4.

- ATR: `ATR Experiment/vlc_fusion_utils.py` (classes `VLCFusion`, `VLCFusionTransformLayer`,
  `VLCFusionTransformQueries`), selected via `ensemble_method="VLCFusion"` in `MultimodalDetr`.
- Waymo: `Waymo Experiment/configs/cbam_ensemble_vlm/vlc_fusion.py` (registered detector `VLCFusion`).

## Variant ledger

| Prototype name (code) | Idea | Conditioning | Status |
|---|---|---|---|
| CrossCBAM_DiT_V11 | multi-scale fusion head + 2 VLC blocks | VLM (FiLM) | **→ VLCFusion (final)** |
| CBAM (`multimodal_detr_utils`) | concat + CBAM | none | **baseline kept** (paper "RGB-X") |
| FusionSSD | concat + conv | none | **baseline kept** |
| FusionSSD_SelfAttention | + self-attention | none | **baseline kept** |
| LearnableAlign | cross-attention align | none | **baseline kept** |
| CBAM_FiLM, FusionSSD_FiLM | conditioned baselines | VLM | archived (not in paper) |
| VLC, VLCA, VLCA_Cross | early conditioned-fusion ideas | VLM | archived |
| VLCAM (`mh_vlcam_utils`) | multi-head VL cross-attn modulation | VLM | archived |
| CGB (`cgb_utils`) | conditioned gating block | VLM | archived |
| CrossCBAM_AdaLN | AdaLN conditioning | VLM | archived |
| CrossCBAM_DiT v1–v10 | DiT-block iterations toward V11 | VLM | archived |
| *_metaclip, continuous conditions | MetaCLIP / continuous condition source | MetaCLIP | archived (not in paper) |
| Waymo: cbam_ensemble_vlm(_film), multi_cbam_ensemble_vlm, cross_cbam_dit_vlm_film | misc Waymo fusion trials | mixed | archived |

## Condition sources

Kept (paper): **GPT-4o** (main), **Moondream2**, **SmolVLM** (small-VLM ablation, §4.6) —
`conditions/{,moondream2_conditions/,smolvlm_conditions/}`.
Archived: MetaCLIP and "continuous" conditions (not reported).

## Eval snapshots (historical, ATR seen/unseen overall mAP)

Architecture-iteration progression that led to VLCFusion (from the old `ATR Experiment/notes.txt`,
7 VLM conditions, 2 VLC blocks):

| Prototype | Seen mAP | Unseen mAP |
|---|---|---|
| Base detector (IR) | 0.366 | 0.102 |
| Base detector (Visible) | 0.522 | 0.225 |
| CrossCBAM_DiT_V4 | 0.557 | 0.134 |
| CrossCBAM_DiT_V5 | 0.569 | 0.127 |
| CrossCBAM_DiT_V8 | 0.593 | 0.135 |
| **CrossCBAM_DiT_V11 (= VLCFusion)** | **0.602** | 0.130 |

(The number-of-VLC-blocks ablation — 2/4/6/8 blocks of VLCFusion — is a separate sweep.)

## Recovery

Any removed file is in the `pre-release-archive` tag, e.g.:
```
git checkout pre-release-archive -- "ATR Experiment/cross_cbam_dit_v8_utils.py"
```

# Copyright 2026 Anthropic PBC / Antigravity Multimodal Extension
# SPDX-License-Identifier: Apache-2.0
"""Standalone Adaptive Multi-Scale Spatial Overlay Exporter for cross_attn_adaptive_v3 MLLMs.

Unlike jlens/vis_multimodal.py (which overlays a fixed 12x12 uniform grid, correct
only for TokenPacker's fixed-size projector), this module supports projectors that
do multi-scale, importance-masked adaptive patch selection (e.g. mm_projector_type
'cross_attn_adaptive_v3'), where each visual token can cover a different-sized
rectangular region of the image. The overlay is built directly from the projector's
own per-image selection masks (llava.model.multimodal_projector.builder.tmp_masks),
not from a synthetic grid formula.
"""

import os
import io
import html
import json
import base64
import gzip
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from typing import Optional, Any

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.vis import SliceData, _ranks_of, build_page


def build_patch_boxes(masks_by_scale: dict, image_px_size: int) -> list:
    """Maps a projector's per-scale selection masks to per-token pixel-fraction boxes.

    masks_by_scale: {patch_size_units: (H_p, W_p) bool/float array, batch dim already
    stripped}, e.g. {1: (24,24), 2: (12,12), 4: (6,6)}. patch_size_units is expressed
    in units of the finest scale's grid cell (raw_grid_size = the largest mask's side).

    Returns patch_boxes[k] for k = 0..N-1, ordered to match the projector's actual
    output token order: scale ascending (finest first), then row-major (row, col)
    within each scale's mask. Each box has normalized [0,1] x0/y0/x1/y1 fractions of
    the image, plus the source scale/row/col for display.
    """
    if not masks_by_scale:
        raise ValueError("masks_by_scale is empty")

    raw_grid_size = max(m.shape[-1] for m in masks_by_scale.values())
    unit_px = image_px_size // raw_grid_size
    if unit_px * raw_grid_size != image_px_size:
        raise ValueError(
            f"image_px_size={image_px_size} not evenly divisible by raw_grid_size={raw_grid_size}"
        )

    boxes = []
    for patch_size_units in sorted(masks_by_scale.keys()):
        mask = np.asarray(masks_by_scale[patch_size_units]).astype(bool)
        rows, cols = np.nonzero(mask)  # row-major traversal
        for r, c in zip(rows.tolist(), cols.tolist()):
            x0 = c * patch_size_units * unit_px
            y0 = r * patch_size_units * unit_px
            side = patch_size_units * unit_px
            boxes.append({
                "x0": x0 / image_px_size,
                "y0": y0 / image_px_size,
                "x1": (x0 + side) / image_px_size,
                "y1": (y0 + side) / image_px_size,
                "scale": patch_size_units,
                "row": r,
                "col": c,
            })
    return boxes


@torch.no_grad()
def export_multimodal_slice_html_adaptive(
    lens_model: Any,
    fitted_lens_path: str,
    image_path: str,
    prompt_text: str,
    output_html_path: str = "out/visualizations/multimodal_adaptive_slice.html",
    top_n: int = 10,
    allow_token_count_mismatch: bool = False,
):
    """Exports a standalone HTML visualization with a variable-size adaptive-scale
    spatial image overlay under the heatmap, built from the projector's own real
    per-image patch selection masks (not a uniform grid).
    """
    device = lens_model.input_device
    tokenizer = lens_model.tokenizer

    if not os.path.exists(fitted_lens_path):
        print(f"Error: Lens checkpoint {fitted_lens_path} not found.")
        return None

    lens = JacobianLens.from_pretrained(os.path.dirname(fitted_lens_path), filename=os.path.basename(fitted_lens_path))
    print(f"Loaded Jacobian Lens from {fitted_lens_path}")

    # 1. Process Image into Preprocessed Tensor
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    raw_image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([raw_image], lens_model.image_processor, lens_model.model.config)[0].unsqueeze(0).to(device, dtype=torch.bfloat16)
    image_px_size = image_tensor.shape[-1]

    # Un-normalize image_tensor [3, H, W] -> HxW PIL Image
    mean = getattr(lens_model.image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
    std = getattr(lens_model.image_processor, 'image_std', [0.26862954, 0.26130258, 0.27577711])

    img_np = image_tensor.detach().squeeze(0).cpu().float().numpy()
    for c in range(3):
        img_np[c] = img_np[c] * std[c] + mean[c]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    processed_pil = Image.fromarray(img_np.transpose(1, 2, 0))

    # Base64 encode the preprocessed image
    buffered = io.BytesIO()
    processed_pil.save(buffered, format="JPEG")
    image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

    # 2. Compute Fused Multimodal Input Embeddings, capturing the projector's own
    # per-image selection masks immediately afterward (tmp_masks is a module-level
    # global only valid right after this forward pass).
    full_prompt = prompt_text
    input_ids = tokenizer_image_token(full_prompt, lens_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    inputs_embeds = lens_model.get_multimodal_inputs_embeds(input_ids, image_tensor)
    seq_len = inputs_embeds.shape[1]

    import llava.model.multimodal_projector.builder as builder
    raw_masks = getattr(builder, "tmp_masks", None)
    if not raw_masks:
        raise RuntimeError(
            "builder.tmp_masks is empty/missing after get_multimodal_inputs_embeds(); "
            "this checkpoint's projector did not run PatchTokenizer_v2.forward() — "
            "is this a cross_attn_adaptive_v3 checkpoint? For TokenPacker-style "
            "checkpoints, use jlens.vis_multimodal.export_multimodal_slice_html_with_grid instead."
        )
    masks_by_scale = {}
    for patch_size_units, v in raw_masks.items():
        if v.shape[0] != 1:
            raise NotImplementedError(
                f"batch size {v.shape[0]} != 1 for scale {patch_size_units}; "
                "this exporter only supports single-image export (B=1)"
            )
        masks_by_scale[int(patch_size_units)] = v[0].detach().to("cpu").numpy()

    layers = sorted(set(list(lens.source_layers) + [lens_model.n_layers - 1]))
    n_layers = len(layers)

    # 3. Record Activations & Compute Layer Logits
    with ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(inputs_embeds)
        activations = {layer: recorder.activations[layer].detach() for layer in layers}

    top_ids = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    top_ranks = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    tracked_ids_set = set()

    for l_idx, layer in enumerate(layers):
        res = activations[layer][0].float()
        if layer in lens.jacobians:
            res = lens.transport(res, layer)
        logits = lens_model.unembed(res).float()

        top_k = torch.topk(logits, k=top_n, dim=-1)
        top_ids[:, l_idx, :] = top_k.indices.cpu().numpy()
        top_ranks[:, l_idx, :] = np.arange(top_n)[None, :]

        for tid in top_k.indices.flatten().tolist():
            tracked_ids_set.add(tid)

    tracked_token_ids = sorted(tracked_ids_set)
    rank_tensor = np.zeros((seq_len, n_layers, len(tracked_token_ids)), dtype=np.int32)

    for l_idx, layer in enumerate(layers):
        res = activations[layer][0].float()
        if layer in lens.jacobians:
            res = lens.transport(res, layer)
        logits = lens_model.unembed(res).float()
        target_tensors = torch.tensor(tracked_token_ids, device=logits.device)
        ranks = _ranks_of(logits, target_tensors)
        rank_tensor[:, l_idx, :] = ranks.cpu().numpy()

    # 4. Build Dynamic Token Sequence & <image> Alignment
    context_token_ids = input_ids[0].tolist()
    if IMAGE_TOKEN_INDEX in context_token_ids:
        img_pos = context_token_ids.index(IMAGE_TOKEN_INDEX)
    else:
        img_pos = 1

    n_img_tokens = seq_len - (len(context_token_ids) - 1)
    img_strs = [f"<img_{i}>" for i in range(max(1, n_img_tokens))]
    img_ids = [IMAGE_TOKEN_INDEX] * len(img_strs)

    prefix_ids = context_token_ids[:img_pos]
    suffix_ids = context_token_ids[img_pos + 1:]

    prefix_strs = [tokenizer.decode([t], clean_up_tokenization_spaces=False) if t >= 0 else "<image>" for t in prefix_ids]
    suffix_strs = [tokenizer.decode([t], clean_up_tokenization_spaces=False) if t >= 0 else "<image>" for t in suffix_ids]

    full_ctx_strs = prefix_strs + img_strs + suffix_strs
    full_ctx_ids = prefix_ids + img_ids + suffix_ids

    patch_boxes = build_patch_boxes(masks_by_scale, image_px_size)
    if len(patch_boxes) != n_img_tokens:
        msg = (
            f"token count mismatch: n_img_tokens={n_img_tokens} (from sequence length) != "
            f"len(patch_boxes)={len(patch_boxes)} (from tmp_masks). Per-scale counts: "
            f"{ {k: int(v.sum()) for k, v in masks_by_scale.items()} }"
        )
        if not allow_token_count_mismatch:
            raise RuntimeError(msg + " Pass allow_token_count_mismatch=True to export anyway.")
        print("WARNING: " + msg)

    vocab_fragment = {}
    for tid in tracked_token_ids:
        if tid < 0 or tid == IMAGE_TOKEN_INDEX:
            vocab_fragment[tid] = "<image>"
        else:
            try:
                vocab_fragment[tid] = tokenizer.decode([tid])
            except Exception:
                vocab_fragment[tid] = f"token_{tid}"

    slice_data = SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=full_ctx_ids[:seq_len],
        context_token_strs=full_ctx_strs[:seq_len],
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked_token_ids,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab_fragment,
        vocab_size=getattr(tokenizer, 'vocab_size', 32000)
    )

    # Render HTML page using slice_vis_multimodal_adaptive.html (standalone new template)
    out_dir = os.path.dirname(os.path.abspath(output_html_path))
    os.makedirs(out_dir, exist_ok=True)

    from importlib.resources import files
    from jlens.vis import _slice_meta, _slice_bin, _template

    meta = _slice_meta(slice_data, prompt_text, f"Adaptive Multi-Scale Lens: {prompt_text[:30]}", f"Multimodal adaptive-patch spatial readout for prompt: '{prompt_text}'", None, None)
    meta["image_b64"] = image_b64
    meta["patch_boxes"] = patch_boxes
    meta["img_start"] = img_pos
    meta["n_img_tokens"] = n_img_tokens

    ranks = slice_data.rank_tensor.astype("<i4")
    file_map = {"slice.bin": _slice_bin(slice_data)} | {
        f"ranks/{tid}.bin": gzip.compress(ranks[:, :, i].tobytes(), compresslevel=6)
        for i, tid in enumerate(slice_data.tracked_token_ids)
    }

    bootstrap = {
        "mode": "embed",
        "meta": meta,
        "files": {
            name: base64.b64encode(body).decode() for name, body in file_map.items()
        },
    }

    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False).replace("</", "<\\/")

    try:
        template_str = (files("jlens") / "data" / "slice_vis_multimodal_adaptive.html").read_text(encoding="utf-8")
    except Exception:
        import jlens
        template_str = (Path(jlens.__file__).parent / "data" / "slice_vis_multimodal_adaptive.html").read_text(encoding="utf-8")

    d3_tag = _template("embed").split("<style>")[0]

    page_html = (
        template_str
        .replace("__TITLE__", html.escape(f"Adaptive Multi-Scale Lens: {prompt_text[:30]}"))
        .replace("__WHAT__", html.escape(f"Multimodal adaptive-patch spatial readout for prompt: '{prompt_text}'"))
        .replace("__D3__", d3_tag)
        .replace("__BOOTSTRAP__", bootstrap_json)
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(page_html)

    payload_bytes = sum(len(b) for b in file_map.values())
    print(f"🎉 Exported Adaptive Multi-Scale Spatial HTML to: {output_html_path}")
    print(f"Payload size: {payload_bytes / 1024:.1f} KB. Download to your machine and open in any browser!")
    return page_html

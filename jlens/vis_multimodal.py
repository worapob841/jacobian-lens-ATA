# Copyright 2026 Anthropic PBC / Antigravity Multimodal Extension
# SPDX-License-Identifier: Apache-2.0
"""Standalone Multimodal Spatial Grid Exporter for TokenPacker MLLMs.

Renders an interactive HTML page displaying the preprocessed 336x336 image
with a 12x12 SVG spatial grid overlay under the Jacobian Lens heatmap.
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


@torch.no_grad()
def export_multimodal_slice_html_with_grid(
    lens_model: Any,
    fitted_lens_path: str,
    image_path: str,
    prompt_text: str,
    output_html_path: str = "out/visualizations/multimodal_grid_slice.html",
    mode: str = "embed",
    top_n: int = 10
):
    """Exports a standalone HTML visualization with a 12x12 spatial image grid overlay under the heatmap.
    Uses the exact preprocessed 336x336 image_tensor for 100% spatial alignment.
    """
    device = lens_model.input_device
    tokenizer = lens_model.tokenizer

    if not os.path.exists(fitted_lens_path):
        print(f"Error: Lens checkpoint {fitted_lens_path} not found.")
        return None

    lens = JacobianLens.from_pretrained(os.path.dirname(fitted_lens_path), filename=os.path.basename(fitted_lens_path))
    print(f"Loaded Jacobian Lens from {fitted_lens_path}")

    # 1. Process Image into Preprocessed 336x336 Tensor
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    raw_image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([raw_image], lens_model.image_processor, lens_model.model.config)[0].unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    # Un-normalize image_tensor [3, 336, 336] -> 336x336 PIL Image
    mean = getattr(lens_model.image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
    std = getattr(lens_model.image_processor, 'image_std', [0.26862954, 0.26130258, 0.27577711])

    img_np = image_tensor.detach().squeeze(0).cpu().float().numpy()
    for c in range(3):
        img_np[c] = img_np[c] * std[c] + mean[c]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    processed_pil = Image.fromarray(img_np.transpose(1, 2, 0))

    # Base64 encode the preprocessed 336x336 image
    buffered = io.BytesIO()
    processed_pil.save(buffered, format="JPEG")
    image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

    # 2. Compute Fused Multimodal Input Embeddings
    # full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    full_prompt = prompt_text
    input_ids = tokenizer_image_token(full_prompt, lens_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    inputs_embeds = lens_model.get_multimodal_inputs_embeds(input_ids, image_tensor)
    seq_len = inputs_embeds.shape[1]
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

    # Render HTML page using slice_vis_multimodal.html (standalone new template)
    out_dir = os.path.dirname(os.path.abspath(output_html_path))
    os.makedirs(out_dir, exist_ok=True)

    from importlib.resources import files
    from jlens.vis import _slice_meta, _slice_bin, _template

    meta = _slice_meta(slice_data, prompt_text, f"TokenPacker Spatial Grid Lens: {prompt_text[:30]}", f"Multimodal 12x12 Spatial Grid Readout for prompt: '{prompt_text}'", None, None)
    meta["image_b64"] = image_b64
    meta["grid_rows"] = 12
    meta["grid_cols"] = 12
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
        template_str = (files("jlens") / "data" / "slice_vis_multimodal.html").read_text(encoding="utf-8")
    except Exception:
        import jlens
        template_str = (Path(jlens.__file__).parent / "data" / "slice_vis_multimodal.html").read_text(encoding="utf-8")

    d3_tag = _template("embed").split("<style>")[0]

    page_html = (
        template_str
        .replace("__TITLE__", html.escape(f"TokenPacker Spatial Grid Lens: {prompt_text[:30]}"))
        .replace("__WHAT__", html.escape(f"Multimodal 12x12 Spatial Grid Readout for prompt: '{prompt_text}'"))
        .replace("__D3__", d3_tag)
        .replace("__BOOTSTRAP__", bootstrap_json)
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(page_html)

    payload_bytes = sum(len(b) for b in file_map.values())
    print(f"🎉 Exported 336x336 Spatial Grid HTML to: {output_html_path}")
    print(f"Payload size: {payload_bytes / 1024:.1f} KB. Download to your machine and open in any browser!")
    return page_html




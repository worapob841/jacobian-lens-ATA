# from ata_multimodal_jlens import MODEL_PATH
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ==========================================
# TokenPacker Repository Path Setup
# ==========================================
TOKENPACKER_REPO = "/g/home/orachat.c/project/MLLM/TokenPacker"
if TOKENPACKER_REPO not in sys.path and os.path.exists(TOKENPACKER_REPO):
    sys.path.insert(0, TOKENPACKER_REPO)
    print(f"Added {TOKENPACKER_REPO} to sys.path")

import jlens
from jlens.fitting import valid_position_mask, SKIP_FIRST_N_POSITIONS
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.vis import compute_slice, build_page, notebook_iframe, SliceData, _ranks_of

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("jlens and TokenPacker environment initialized successfully.")

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


# ==========================================
# MultimodalTokenPackerLensModel Adapter
# ==========================================
class MultimodalTokenPackerLensModel:
    """LensModel protocol implementation for TokenPacker MLLMs.
    Supports cross_attn_adaptive_v* and HD patch architectures.
    """

    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self._text_module = model.model  # LlavaLlamaModel
        self.layers = self._text_module.layers
        self._final_norm = self._text_module.norm
        self._lm_head = model.lm_head

        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def input_device(self) -> torch.device:
        return self._lm_head.weight.device

    def encode(self, text: str, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", max_length=max_length
        ).input_ids.to(self.input_device)

    def get_multimodal_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        h_block: Optional[Any] = None,
        w_block: Optional[Any] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Generates fused image-text input embeddings via TokenPacker's projector."""
        h_b = h_block.tolist() if torch.is_tensor(h_block) else h_block
        w_b = w_block.tolist() if torch.is_tensor(w_block) else w_block

        # if hasattr(self.model, "prepare_adaptive_inputs_labels_for_multimodal"):
        _, _, _, inputs_embeds, _ = (
            self.model.prepare_adaptive_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=images,
                mode=mode,
                h_block=h_b,
                w_block=w_b,
            )
        )
        # else:
        # _, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
        #     input_ids=input_ids,
        #     attention_mask=None,
        #     past_key_values=None,
        #     labels=None,
        #     images=images,
        #     mode=mode,
        #     h_block=h_b,
        #     w_block=w_b
        # )
        return inputs_embeds

    def forward(self, input_ids_or_embeds: torch.Tensor) -> Any:
        if input_ids_or_embeds.dtype == torch.int64:
            return self._text_module(input_ids=input_ids_or_embeds, use_cache=False)
        else:
            return self._text_module(inputs_embeds=input_ids_or_embeds, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        return self._lm_head(
            self._final_norm(residual.to(target_dtype).to(target_device))
        )


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
            boxes.append(
                {
                    "x0": x0 / image_px_size,
                    "y0": y0 / image_px_size,
                    "x1": (x0 + side) / image_px_size,
                    "y1": (y0 + side) / image_px_size,
                    "scale": patch_size_units,
                    "row": r,
                    "col": c,
                }
            )
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

    lens = JacobianLens.from_pretrained(
        os.path.dirname(fitted_lens_path), filename=os.path.basename(fitted_lens_path)
    )
    print(f"Loaded Jacobian Lens from {fitted_lens_path}")

    # 1. Process Image into Preprocessed Tensor
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    raw_image = Image.open(image_path).convert("RGB")
    image_tensor = (
        process_images(
            [raw_image], lens_model.image_processor, lens_model.model.config
        )[0]
        .unsqueeze(0)
        .to(device, dtype=torch.bfloat16)
    )
    image_px_size = image_tensor.shape[-1]

    # Un-normalize image_tensor [3, H, W] -> HxW PIL Image
    mean = getattr(
        lens_model.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]
    )
    std = getattr(
        lens_model.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711]
    )

    img_np = image_tensor.detach().squeeze(0).cpu().float().numpy()
    for c in range(3):
        img_np[c] = img_np[c] * std[c] + mean[c]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    processed_pil = Image.fromarray(img_np.transpose(1, 2, 0))

    # Base64 encode the preprocessed image
    buffered = io.BytesIO()
    processed_pil.save(buffered, format="JPEG")
    image_b64 = (
        "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    )

    # 2. Compute Fused Multimodal Input Embeddings, capturing the projector's own
    # per-image selection masks immediately afterward (tmp_masks is a module-level
    # global only valid right after this forward pass).
    full_prompt = prompt_text
    input_ids = (
        tokenizer_image_token(
            full_prompt, lens_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )

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
    suffix_ids = context_token_ids[img_pos + 1 :]

    prefix_strs = [
        (
            tokenizer.decode([t], clean_up_tokenization_spaces=False)
            if t >= 0
            else "<image>"
        )
        for t in prefix_ids
    ]
    suffix_strs = [
        (
            tokenizer.decode([t], clean_up_tokenization_spaces=False)
            if t >= 0
            else "<image>"
        )
        for t in suffix_ids
    ]

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
            raise RuntimeError(
                msg + " Pass allow_token_count_mismatch=True to export anyway."
            )
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
        vocab_size=getattr(tokenizer, "vocab_size", 32000),
    )

    # Render HTML page using slice_vis_multimodal_adaptive.html (standalone new template)
    out_dir = os.path.dirname(os.path.abspath(output_html_path))
    os.makedirs(out_dir, exist_ok=True)

    from importlib.resources import files
    from jlens.vis import _slice_meta, _slice_bin, _template

    meta = _slice_meta(
        slice_data,
        prompt_text,
        f"Adaptive Multi-Scale Lens: {prompt_text[:30]}",
        f"Multimodal adaptive-patch spatial readout for prompt: '{prompt_text}'",
        None,
        None,
    )
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
        template_str = (
            files("jlens") / "data" / "slice_vis_multimodal_adaptive.html"
        ).read_text(encoding="utf-8")
    except Exception:
        import jlens

        template_str = (
            Path(jlens.__file__).parent / "data" / "slice_vis_multimodal_adaptive.html"
        ).read_text(encoding="utf-8")

    d3_tag = _template("embed").split("<style>")[0]

    page_html = (
        template_str.replace(
            "__TITLE__", html.escape(f"Adaptive Multi-Scale Lens: {prompt_text[:30]}")
        )
        .replace(
            "__WHAT__",
            html.escape(
                f"Multimodal adaptive-patch spatial readout for prompt: '{prompt_text}'"
            ),
        )
        .replace("__D3__", d3_tag)
        .replace("__BOOTSTRAP__", bootstrap_json)
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(page_html)

    payload_bytes = sum(len(b) for b in file_map.values())
    print(f"🎉 Exported Adaptive Multi-Scale Spatial HTML to: {output_html_path}")
    print(
        f"Payload size: {payload_bytes / 1024:.1f} KB. Download to your machine and open in any browser!"
    )
    return page_html


def prep_prompt_vqav2(question_id: int, model_id: str) -> str:
    """Prepares a prompt for VQAv2-style questions with an image."""
    VQAV2_QUESTION_PATH = "/g/home/orachat.c/project/MLLM/TokenPacker/playground/data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl"
    VQAV2_MODEL_ANS = f"/g/home/orachat.c/project/MLLM/TokenPacker/playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/{model_id}/merge.jsonl"
    # open question file where jsonl look like this
    # {"question_id": 262144000, "image": "COCO_test2015_000000262144.jpg", "text": "Is the ball flying towards the batter?\nAnswer the question using a single word or phrase.", "category": "default"}
    # {"question_id": 262144001, "image": "COCO_test2015_000000262144.jpg", "text": "What sport is this?\nAnswer the question using a single word or phrase.", "category": "default"}
    # and return image path, question, model answer
    image_root = (
        "/g/home/orachat.c/project/MLLM/TokenPacker/playground/data/eval/vqav2/test2015"
    )
    with open(VQAV2_QUESTION_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["question_id"] == question_id:
                question = item["text"]
                image_path = f"{image_root}/{item['image']}"
                break

    # model answer where answer is like this
    # {"question_id": 262144005, "prompt": "What credit card company is on the banner in the background?\nAnswer the question using a single word or phrase.", "text": "Mastercard", "answer_id": "To8CJUTumuQgR8kRHL9Si2", "model_id": "llava-adaptive-hd-it-thres6040-multilev-dubconv-MGM-Finetune-en-h100-01012026", "metadata": {}}
    # {"question_id": 262144003, "prompt": "Is the pitcher wearing a hat?\nAnswer the question using a single word or phrase.", "text": "Yes", "answer_id": "CPTyG27YwC5ZbFkaBUcroD", "model_id": "llava-adaptive-hd-it-thres6040-multilev-dubconv-MGM-Finetune-en-h100-01012026", "metadata": {}}
    with open(VQAV2_MODEL_ANS, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["question_id"] == question_id:
                model_answer = item["text"]
                break
    return (    
        image_path,
        f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT: {model_answer}",
    )


if __name__ == "__main__":
    MODEL_PATH = "/g/home/orachat.c/project/MLLM/TokenPacker/checkpoints/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026"
    MODEL_NAME = "llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026"
    FITTED_LENS_PATH = "/g/home/orachat.c/project/MLLM/jacobian-lens-ATA/out/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026/llava-cross_attn_adaptive-it-spatialscorrer-thres4060-bound-3060-alpha-001-multilev-dubconv-llava_v1_5_mix665k-en-h100-08092026_multimodal_vqav2_lens.pt"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    QID = 17515006

    print(f"Using device: {DEVICE}")
    try:
        from transformers import AutoTokenizer, AutoConfig
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

        if os.path.exists(MODEL_PATH):
            print(f"Loading TokenPacker tokenizer & model from {MODEL_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, model_max_length=2048, padding_side="right", use_fast=True
            )

            config = AutoConfig.from_pretrained(MODEL_PATH)
            config._attn_implementation = "sdpa"
            config.attn_implementation = "sdpa"

            model = LlavaLlamaForCausalLM.from_pretrained(
                MODEL_PATH,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=DEVICE,
            )

            # Force SDPA FlashAttention on both model and sub-module config
            model.config._attn_implementation = "sdpa"
            model.config.attn_implementation = "sdpa"
            if hasattr(model, "model") and hasattr(model.model, "config"):
                model.model.config._attn_implementation = "sdpa"
                model.model.config.attn_implementation = "sdpa"

            # Initialize Vision Tower & Image Processor
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=DEVICE, dtype=torch.bfloat16)
            image_processor = vision_tower.image_processor

            model.eval()
            attn_type = (
                type(model.model.layers[0].self_attn).__name__
                if hasattr(model, "model") and hasattr(model.model, "layers")
                else "Unknown"
            )
            print(
                f"Model loaded successfully: {type(model).__name__} | Attention Module: {attn_type}"
            )
        else:
            print(
                f"Note: MODEL_PATH '{MODEL_PATH}' not found locally. Update path when running on GPU server."
            )
            tokenizer, model, image_processor = None, None, None
    except Exception as err:
        print(f"Model loading error: {err}")
        tokenizer, model, image_processor = None, None, None

    if model is not None:
        lens_model = MultimodalTokenPackerLensModel(model, tokenizer, image_processor)
        print(
            f"MultimodalTokenPackerLensModel initialized: n_layers={lens_model.n_layers}, d_model={lens_model.d_model}"
        )
    else:
        lens_model = None

    image_path, prompt = prep_prompt_vqav2(QID, MODEL_NAME)
    print(image_path, prompt)
    tmp = export_multimodal_slice_html_adaptive(
        lens_model,
        fitted_lens_path=FITTED_LENS_PATH,
        image_path=image_path,
        prompt_text=prompt,
        output_html_path=f"out/{MODEL_NAME}/visualizations/multimodal_slice_vqav2_{QID}.html",
        # mode = "embed",
        top_n=10,
    )

    # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is the person holding?\nAnswer the question using a single word or phrase. ASSISTANT: Wii remote"
    # tmp = export_multimodal_slice_html_adaptive(
    #     lens_model,
    #     fitted_lens_path='/mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/jacobian-lens-ATA/out/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026_multimodal_vqav2_lens.pt',
    #     image_path="/mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker/playground/data/eval/vqav2/test2015/COCO_test2015_000000017515.jpg",
    #     prompt_text=prompt,
    #     output_html_path = "out/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026/visualizations/multimodal_slice_grid.html",
    #     # mode = "embed",
    #     top_n = 10
    # )
    # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWho is the fax to?\nAnswer the question using a single word or phrase. ASSISTANT:Mike"
    # tmp = export_multimodal_slice_html_adaptive(
    #     lens_model,
    #     fitted_lens_path='/mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/jacobian-lens-ATA/out/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026_multimodal_vqav2_lens.pt',
    #     image_path="/mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker//playground/data/eval/docvqa/images/hfkm0020_1.png",
    #     prompt_text=prompt,
    #     output_html_path = "out/llava-cross_attn_adaptive-it-randthres0205-0408-eval3040-multilev-dubconv-llava_v1_5_mix665k-en-h100-05152026/visualizations/multimodal_slice_grid-docvqa-hfkm0020_1.html",
    #     # mode = "embed",
    #     top_n = 10
    # )
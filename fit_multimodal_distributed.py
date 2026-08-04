"""
fit_multimodal_distributed.py
Distributed Data-Parallel Jacobian Lens fitting for TokenPacker MLLMs across multiple GPUs using torchrun and jlens.JacobianLens.merge().

Usage:
    torchrun --nproc_per_node=4 fit_multimodal_distributed.py \
        --model_path /mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker/checkpoints/TokenPacker-7b-144token \
        --question_file /mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image_folder /mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker/playground/data/eval/vqav2/test2015 \
        --n_samples 200 \
        --dim_batch 16 \
        --max_seq_len 256 \
        --output_lens_path out/tokenpacker_multimodal_vqav2_lens.pt
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add TokenPacker repo to sys.path
TOKENPACKER_REPO = os.environ.get("TOKENPACKER_REPO", "/mnt/pvc-shared-pvc-data-volume-ea328235/MLLM/TokenPacker")
if TOKENPACKER_REPO not in sys.path and os.path.exists(TOKENPACKER_REPO):
    sys.path.insert(0, TOKENPACKER_REPO)

import jlens
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens


class MultimodalTokenPackerLensModel:
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
        return self.tokenizer(text, return_tensors="pt", max_length=max_length).input_ids.to(self.input_device)

    def get_multimodal_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        h_block: Optional[Any] = None,
        w_block: Optional[Any] = None,
        mode: Optional[str] = None
    ) -> torch.Tensor:
        h_b = h_block.tolist() if torch.is_tensor(h_block) else h_block
        w_b = w_block.tolist() if torch.is_tensor(w_block) else w_block

        # if hasattr(self.model, "prepare_adaptive_inputs_labels_for_multimodal"):
        #     _, _, _, inputs_embeds, _ = self.model.prepare_adaptive_inputs_labels_for_multimodal(
        #         input_ids=input_ids,
        #         attention_mask=None,
        #         past_key_values=None,
        #         labels=None,
        #         images=images,
        #         mode=mode,
        #         h_block=h_b,
        #         w_block=w_b
        #     )
        # else:
        _, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=images,
            mode=mode,
            h_block=h_b,
            w_block=w_b
        )
        return inputs_embeds

    def forward(self, input_ids_or_embeds: torch.Tensor) -> Any:
        if input_ids_or_embeds.dtype == torch.int64:
            return self._text_module(input_ids=input_ids_or_embeds, use_cache=False)
        else:
            return self._text_module(inputs_embeds=input_ids_or_embeds, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        return self._lm_head(self._final_norm(residual.to(target_dtype).to(target_device)))


class VQAArgs:
    def __init__(self, conv_mode="vicuna_v1"):
        self.conv_mode = conv_mode


def main():
    parser = argparse.ArgumentParser(description="Distributed Multimodal Jacobian Lens Fitting")
    parser.add_argument("--model_path", type=str, required=True, help="Path to TokenPacker checkpoint")
    parser.add_argument("--question_file", type=str, required=True, help="Path to VQAv2 questions .jsonl")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to VQAv2 images directory")
    parser.add_argument("--n_samples", type=int, default=200, help="Total samples to fit across all GPUs")
    parser.add_argument("--dim_batch", type=int, default=16, help="Output dimensions computed per backward pass")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--skip_first", type=int, default=1, help="Leading positions to exclude")
    parser.add_argument("--output_lens_path", type=str, default="out/tokenpacker_multimodal_vqav2_lens.pt", help="Final saved lens path")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1", help="Conversation template mode")
    args = parser.parse_args()

    # Determine Distributed Rank and Device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cpu"

    print(f"[Rank {rank}/{world_size}] Initialized on device {device_str}")

    # Load Model & Tokenizer on local GPU
    from transformers import AutoTokenizer, AutoConfig
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.eval.model_vqa_loader import CustomDataset
    import llava.eval.model_vqa_loader as vqa_loader

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=2048, padding_side="right", use_fast=True)
    
    config = AutoConfig.from_pretrained(args.model_path)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_str
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device_str, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.eval()
    lens_model = MultimodalTokenPackerLensModel(model, tokenizer, image_processor)

    # Slice Dataset for this Rank
    vqa_loader.args = VQAArgs(conv_mode=args.conv_mode)
    with open(os.path.expanduser(args.question_file), "r") as f:
        all_questions = [json.loads(line) for line in f][:args.n_samples]

    samples_per_rank = math.ceil(len(all_questions) / world_size)
    rank_start = rank * samples_per_rank
    rank_end = min(len(all_questions), (rank + 1) * samples_per_rank)
    rank_questions = all_questions[rank_start:rank_end]

    print(f"[Rank {rank}] Processing {len(rank_questions)} samples (indices {rank_start} to {rank_end-1})")

    dataset = CustomDataset(rank_questions, args.image_folder, tokenizer, image_processor, model.config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Perform Fitting for this Rank
    n_layers, d_model = lens_model.n_layers, lens_model.d_model
    source_layers = list(range(n_layers - 1))
    target_layer = n_layers - 1
    mode = getattr(lens_model.model.config, 'image_aspect_ratio', None)

    jacobian_sum = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}
    n_done = 0

    pbar = tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0 and local_rank != 0))
    for sample_idx, batch in enumerate(pbar):
        input_ids, image_tensor, h_block, w_block = batch
        input_ids = input_ids.to(device_str)

        if image_tensor.ndim == 5:
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor.to(device_str, dtype=torch.bfloat16)

        with torch.no_grad():
            inputs_embeds = lens_model.get_multimodal_inputs_embeds(
                input_ids=input_ids,
                images=image_tensor,
                h_block=h_block,
                w_block=w_block,
                mode=mode
            )

        seq_len = inputs_embeds.shape[1]
        if seq_len > args.max_seq_len:
            inputs_embeds = inputs_embeds[:, :args.max_seq_len, :]
            seq_len = args.max_seq_len

        try:
            position_mask = valid_position_mask(seq_len, skip_first=args.skip_first)
        except ValueError as exc:
            continue

        n_passes = math.ceil(d_model / args.dim_batch)
        per_sample_J = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}

        with ActivationRecorder(lens_model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)) as recorder, torch.enable_grad():
            replicated_embeds = inputs_embeds.expand(args.dim_batch, -1, -1)
            lens_model.forward(replicated_embeds)

            target_act = recorder.activations[target_layer]
            source_acts = [recorder.activations[l] for l in source_layers]
            valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_act.device)
            batch_indices = torch.arange(args.dim_batch, device=target_act.device)
            cotangent = torch.zeros_like(target_act)

            for pass_idx, dim_start in enumerate(range(0, d_model, args.dim_batch)):
                n_dims = min(args.dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[batch_indices[:n_dims, None], valid_positions[None, :], dim_start + batch_indices[:n_dims, None]] = 1.0

                grads = torch.autograd.grad(
                    outputs=target_act,
                    inputs=source_acts,
                    grad_outputs=cotangent,
                    retain_graph=(pass_idx < n_passes - 1)
                )

                for layer, grad in zip(source_layers, grads, strict=True):
                    positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                    rows = grad[:n_dims, positions_on_device, :].float().mean(dim=1).cpu()
                    per_sample_J[layer][dim_start : dim_start + n_dims, :] = rows
                del grads

        for layer in source_layers:
            jacobian_sum[layer] += per_sample_J[layer]

        n_done += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save per-rank lens checkpoint
    out_dir = os.path.dirname(os.path.abspath(args.output_lens_path))
    os.makedirs(out_dir, exist_ok=True)
    rank_lens_path = os.path.join(out_dir, f"tokenpacker_rank_{rank}.pt")

    jacobian_mean = {l: jacobian_sum[l] / max(1, n_done) for l in source_layers}
    rank_lens = JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
    rank_lens.save(rank_lens_path)
    print(f"[Rank {rank}] Saved rank lens ({n_done} samples) to {rank_lens_path}")

    # Rank 0 merges all rank lenses
    if rank == 0:
        print("[Rank 0] Waiting for all rank checkpoints and merging...")
        rank_lenses = []
        for r in range(world_size):
            r_path = os.path.join(out_dir, f"tokenpacker_rank_{r}.pt")
            if os.path.exists(r_path):
                r_lens = JacobianLens.from_pretrained(out_dir, filename=os.path.basename(r_path))
                rank_lenses.append(r_lens)

        if len(rank_lenses) > 0:
            final_lens = JacobianLens.merge(rank_lenses)
            final_lens.save(args.output_lens_path)
            print(f"🎉 Successfully merged {len(rank_lenses)} rank lenses into: {args.output_lens_path}")


if __name__ == "__main__":
    main()

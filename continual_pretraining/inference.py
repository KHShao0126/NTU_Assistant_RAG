"""Simple chatbot REPL for a fine-tuned causal LM (supports PEFT adapters / QLoRA).

Usage examples:
  # load a full model directory
  python inference.py --model_name_or_path outputs/conti-llama

  # load a base model plus a PEFT adapter directory
  python inference.py --model_name_or_path /path/to/base-llama --adapter_dir outputs/conti-llama-qlora --use_peft

Options include generation hyperparams (max_new_tokens, temperature, top_p, top_k).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, List

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model_and_tokenizer(model_name_or_path: str, adapter_dir: Optional[str] = None, use_peft: bool = False, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer tokenizer from adapter/output dir when available so users don't need
    # to pass a separate tokenizer path. Fall back to the base model path.
    tokenizer = None
    if adapter_dir is not None:
        try:
            logger.info("Trying to load tokenizer from adapter dir: %s", adapter_dir)
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=True)
            logger.info("Loaded tokenizer from adapter dir: %s", adapter_dir)
        except Exception:
            logger.info("No tokenizer found in adapter dir (%s), falling back to model_name_or_path", adapter_dir)

    if tokenizer is None:
        logger.info("Loading tokenizer from %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    # Ensure tokenizer has pad token for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model from %s (device=%s)", model_name_or_path, device)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto" if device.startswith("cuda") else None, torch_dtype=torch.float16 if device.startswith("cuda") else None, trust_remote_code=True)

    if use_peft:
        if not _HAS_PEFT:
            raise RuntimeError("PEFT package is required to load adapters (install with `pip install peft`).")
        if adapter_dir is None:
            raise ValueError("--adapter_dir must be provided when --use_peft is set")
        logger.info("Loading PEFT adapter from %s", adapter_dir)
        try:
            model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto" if device.startswith("cuda") else None)
        except RuntimeError as e:
            # Common failure: embedding size mismatch between adapter checkpoint and base model.
            msg = str(e)
            logger.warning("Failed to load adapter on first attempt: %s", msg)
            # Try to inspect adapter checkpoint to find embedding size and resize base model.
            try:
                import os
                from safetensors.torch import load_file as load_safetensors
                adapter_file = None
                # prefer safetensors file named adapter_model.safetensors
                candidate = os.path.join(adapter_dir, "adapter_model.safetensors")
                if os.path.exists(candidate):
                    adapter_file = candidate
                    state = load_safetensors(adapter_file)
                else:
                    # fallback: find any .safetensors in dir
                    for fname in os.listdir(adapter_dir):
                        if fname.endswith(".safetensors"):
                            adapter_file = os.path.join(adapter_dir, fname)
                            state = load_safetensors(adapter_file)
                            break
                if adapter_file is None:
                    # fallback to torch load of .bin
                    for fname in os.listdir(adapter_dir):
                        if fname.endswith(".bin") or fname.endswith(".pt"):
                            adapter_file = os.path.join(adapter_dir, fname)
                            import torch as _torch
                            state = _torch.load(adapter_file, map_location="cpu")
                            break

                if adapter_file is None:
                    logger.error("Could not find adapter checkpoint in %s to inspect embedding size", adapter_dir)
                    raise

                # find an embed_tokens key
                embed_key = None
                for k in state.keys():
                    if "embed_tokens" in k and state[k].ndim == 2:
                        embed_key = k
                        break
                if embed_key is None:
                    logger.error("No embed_tokens key found in adapter checkpoint; keys: %s", list(state.keys())[:20])
                    raise

                checkpoint_vocab_size = state[embed_key].shape[0]
                current_vocab_size = model.get_input_embeddings().weight.shape[0]
                logger.info("Adapter vocab size: %d ; base model vocab size: %d", checkpoint_vocab_size, current_vocab_size)
                if checkpoint_vocab_size != current_vocab_size:
                    logger.info("Resizing base model embeddings to %d to match adapter checkpoint", checkpoint_vocab_size)
                    model.resize_token_embeddings(checkpoint_vocab_size)
                # retry loading adapter
                model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto" if device.startswith("cuda") else None)
            except Exception as e2:
                logger.exception("Failed to auto-resize embeddings and load adapter: %s", e2)
                raise

    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 50, do_sample: bool = True) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) if "attention_mask" in inputs else None

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        generated = model.generate(**gen_kwargs)

    # decode only the newly generated tokens
    generated = generated[0]
    generated_text = tokenizer.decode(generated[input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


def chat_repl(model, tokenizer, history: Optional[List[str]] = None, **gen_kwargs):
    history = history or []
    print("Chatbot ready. Type a message and press Enter. Ctrl-C or 'exit' to quit.")
    try:
        while True:
            user = input("User: ")
            if not user:
                continue
            if user.strip().lower() in ("exit", "quit"):
                print("Exiting.")
                break

            # naive history: join previous dialog turns (keep short)
            prompt = "\n".join(history[-6:] + ["User: " + user, "Assistant:"])
            reply = generate_reply(model, tokenizer, prompt, **gen_kwargs)
            print("Assistant:", reply)
            # append to history
            history.append("User: " + user)
            history.append("Assistant: " + reply)
    except KeyboardInterrupt:
        print("\nExiting.")


def main():
    parser = argparse.ArgumentParser(description="Run a simple chatbot with a fine-tuned causal LM")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to base model or HF repo id")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to PEFT adapter directory (if using PEFT)")
    parser.add_argument("--use_peft", action="store_true", help="Load PEFT adapter from --adapter_dir")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on (e.g., cuda:0 or cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling instead of greedy decoding")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, adapter_dir=args.adapter_dir, use_peft=args.use_peft, device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
    )

    chat_repl(model, tokenizer, **gen_kwargs)


if __name__ == "__main__":
    main()

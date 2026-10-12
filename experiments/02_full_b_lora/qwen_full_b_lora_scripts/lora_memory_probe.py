"""Measure one longest-input LoRA forward/backward before real training.

The probe applies no optimizer update, and restores Torch RNG and training flags.
CUDA capacity is measured on the user's machine rather than estimated from VRAM.
"""

import gc
import time


def probe_batch(model, examples, collator, loss_fn, use_bf16, context):
    import torch

    if not examples:
        raise ValueError("The memory probe needs at least one prepared example")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("The LoRA memory probe requires a model already on CUDA")

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    training_flags = [(module, module.training) for module in model.modules()]
    batch = None
    loss = None
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Checking the longest {len(examples)} prepared examples in one CUDA batch "
          f"with {str(dtype).removeprefix('torch.')} and checkpointing...", flush=True)

    try:
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        batch = {key: value.to(device) for key, value in collator(examples).items()}
        actual_examples, padded_tokens = batch["input_ids"].shape
        if actual_examples != len(examples):
            raise ValueError("The memory probe collator changed the example count")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = loss_fn(model, batch)
        if not bool(torch.isfinite(loss.detach()).all()):
            raise ValueError("The longest-batch memory probe produced non-finite loss")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        report = {
            "probe_kind": "longest_inputs_forward_backward_without_optimizer_update",
            "examples": actual_examples,
            "padded_tokens_per_example": padded_tokens,
            "padded_tokens_in_batch": actual_examples * padded_tokens,
            "autocast_dtype": str(dtype).removeprefix("torch."),
            "gradient_checkpointing": "nonreentrant",
            "answer_loss": float(loss.detach().float().item()),
            "trainable_parameters_with_grad": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ),
            "allocated_before_gib": allocated_before / (1024 ** 3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            "elapsed_seconds": elapsed,
            "run_context": context,
        }
        print(f"Longest-batch probe passed: {actual_examples} x {padded_tokens} tokens; "
              f"peak allocated {report['peak_allocated_gib']:.2f} GiB, "
              f"reserved {report['peak_reserved_gib']:.2f} GiB; "
              f"forward/backward {elapsed:.2f} seconds.", flush=True)
        return report
    except torch.cuda.OutOfMemoryError as exc:
        # Discard the failed forward's traceback frames before CUDA cache cleanup;
        # they can otherwise retain intermediate activation tensors.
        exc.__traceback__ = None
        raise RuntimeError(
            f"Real batch {len(examples)} did not fit GPU memory during the longest-input "
            "forward/backward probe. No optimizer update was applied. Choose a smaller "
            "real batch and a fresh output directory; this launcher does not reduce "
            "the batch or add accumulation automatically."
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        batch = None
        loss = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        for module, training in training_flags:
            module.training = training

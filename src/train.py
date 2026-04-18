"""Train the Hebrew ASR model.

Example:
    uv run src/train.py \
        --train-dataset dataset/train.jsonl \
        --eval-dataset dataset/val.jsonl \
        --output-dir outputs/asr
"""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import load_file

from checkpoint import resume_step, save_checkpoint, save_epoch_checkpoint
from evaluate import evaluate
from config import parse_args
from data import make_dataloaders
from model import ASRModel
from optimizer import build_optimizer, build_scheduler


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    train_loader, eval_loader = make_dataloaders(args)

    model = ASRModel()

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)
        if accelerator.is_main_process:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            batch.pop("texts", None)
            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(step=opt_step, loss=f"{train_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0 and writer:
                        writer.add_scalar("train/loss", train_loss, opt_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], opt_step)

                    if opt_step % args.save_steps == 0:
                        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, train_loss, args.save_total_limit)

                if opt_step % args.eval_steps == 0:
                    evaluate(model, eval_loader, accelerator, opt_step, writer, train_loader.dataset)

        if args.save_epochs and accelerator.is_main_process:
            save_epoch_checkpoint(accelerator.unwrap_model(model), output_dir, epoch + 1, opt_step, epoch_loss_sum / max(epoch_steps, 1))

    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, epoch_loss_sum / max(epoch_steps, 1), args.save_total_limit)
        if writer:
            writer.close()


if __name__ == "__main__":
    main()

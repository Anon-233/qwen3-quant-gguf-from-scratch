from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from miniqwen.evaluation.compare_logits import compare_logits
from miniqwen.generation import generate_tokens
from miniqwen.runtime.loader import load_gguf_runtime
from miniqwen.tokenizer_adapter import TokenizerAdapter


def load_prompts(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cleanup_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_gguf_model(
    model_path: str,
    tokenizer: TokenizerAdapter,
    prompts: list[dict],
    device: str,
    compute_dtype: str,
) -> tuple[list[dict], list[torch.Tensor]]:
    model = load_gguf_runtime(model_path, device=device, compute_dtype=compute_dtype)
    rows = []
    logits_list = []
    started = perf_counter()
    with torch.no_grad():
        for item in prompts:
            input_ids = tokenizer.encode(item["prompt"])
            logits = model.forward(input_ids)[:, -1, :].detach().cpu()
            logits_list.append(logits)
            gen_started = perf_counter()
            output_ids = generate_tokens(
                model,
                input_ids,
                max_new_tokens=int(item.get("max_new_tokens", 16)),
                temperature=0,
                eos_token_id=model.config.eos_token_id,
                use_cache=True,
            )
            gen_seconds = perf_counter() - gen_started
            new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "new_tokens": new_tokens,
                    "seconds": gen_seconds,
                    "tokens_per_s": new_tokens / gen_seconds if gen_seconds > 0 else 0.0,
                    "generation": tokenizer.decode(output_ids[0].detach().cpu().tolist()),
                    "next_token": int(torch.argmax(logits, dim=-1).item()),
                }
            )
    total_seconds = perf_counter() - started
    del model
    gc.collect()
    _cleanup_device(device)
    for row in rows:
        row["model_total_seconds"] = total_seconds
    return rows, logits_list


def evaluate_transformers_model(
    model_path: str,
    tokenizer: TokenizerAdapter,
    prompts: list[dict],
    device: str,
    compute_dtype: str,
) -> tuple[list[dict], list[torch.Tensor]]:
    from transformers import AutoModelForCausalLM

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    torch_dtype = (
        torch.float16
        if compute_dtype in {"auto", "float16"} and resolved_device.startswith("cuda")
        else torch.float32
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        .to(resolved_device)
        .eval()
    )
    rows = []
    logits_list = []
    started = perf_counter()
    with torch.no_grad():
        for item in prompts:
            input_ids = tokenizer.encode(item["prompt"]).to(resolved_device)
            logits = model(input_ids).logits[:, -1, :].detach().cpu()
            logits_list.append(logits)
            gen_started = perf_counter()
            output_ids = model.generate(
                input_ids,
                max_new_tokens=int(item.get("max_new_tokens", 16)),
                do_sample=False,
                pad_token_id=tokenizer.tokenizer.pad_token_id,
                eos_token_id=tokenizer.tokenizer.eos_token_id,
            )
            if resolved_device.startswith("cuda"):
                torch.cuda.synchronize()
            gen_seconds = perf_counter() - gen_started
            new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "new_tokens": new_tokens,
                    "seconds": gen_seconds,
                    "tokens_per_s": new_tokens / gen_seconds if gen_seconds > 0 else 0.0,
                    "generation": tokenizer.decode(output_ids[0].detach().cpu().tolist()),
                    "next_token": int(torch.argmax(logits, dim=-1).item()),
                }
            )
    total_seconds = perf_counter() - started
    del model
    gc.collect()
    _cleanup_device(resolved_device)
    for row in rows:
        row["model_total_seconds"] = total_seconds
    return rows, logits_list


def compare_series(
    reference_rows: list[dict],
    reference_logits: list[torch.Tensor],
    target_rows: list[dict],
    target_logits: list[torch.Tensor],
    top_k: int,
) -> list[dict]:
    out = []
    for ref_row, ref_logits, row, logits in zip(
        reference_rows, reference_logits, target_rows, target_logits, strict=True
    ):
        m = compare_logits(ref_logits, logits, k=top_k)
        m["id"] = ref_row["id"]
        m["category"] = ref_row["category"]
        m["next_token_match"] = ref_row["next_token"] == row["next_token"]
        out.append(m)
    return out


def write_markdown(
    output: str | Path,
    baseline_path: str,
    quantized_paths: list[str],
    baseline_rows: list[dict],
    quant_rows: dict[str, list[dict]],
    metrics: dict[str, list[dict]],
    ablations: dict[str, list[dict]] | None,
    device: str,
    compute_dtype: str,
) -> None:
    lines = [
        "# Qwen3-0.6B Quantization Evaluation Suite",
        "",
        "Baseline: `" + baseline_path + "`",
        "",
        f"Device: `{device}`",
        "",
        f"Compute dtype: `{compute_dtype}`",
        "",
        "Quantized models:",
        "",
    ]
    for path in quantized_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Aggregate Metrics vs f16 GGUF",
            "",
            "| model | avg cosine | avg MSE | avg top-k overlap | next-token match | avg tok/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for path in quantized_paths:
        ms = metrics[path]
        avg_cos = sum(m["cosine"] for m in ms) / len(ms)
        avg_mse = sum(m["mse"] for m in ms) / len(ms)
        avg_topk = sum(m["topk_overlap"] for m in ms) / len(ms)
        match = sum(1 for m in ms if m["next_token_match"]) / len(ms)
        avg_tps = sum(r["tokens_per_s"] for r in quant_rows[path]) / len(quant_rows[path])
        model_name = Path(path).name
        lines.append(
            f"| `{model_name}` | {avg_cos:.6f} | {avg_mse:.6f} | "
            f"{avg_topk:.3f} | {match:.2%} | {avg_tps:.2f} |"
        )
    base_tps = sum(r["tokens_per_s"] for r in baseline_rows) / len(baseline_rows)
    lines.extend(
        [
            "",
            "Baseline average generation speed in this Python teaching runtime: "
            f"`{base_tps:.2f}` tok/s.",
        ]
    )
    if ablations:
        lines.extend(
            [
                "",
                "## Ablation Metrics",
                "",
                "| comparison | avg cosine | avg MSE | avg top-k overlap | next-token match |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, ms in ablations.items():
            avg_cos = sum(m["cosine"] for m in ms) / len(ms)
            avg_mse = sum(m["mse"] for m in ms) / len(ms)
            avg_topk = sum(m["topk_overlap"] for m in ms) / len(ms)
            match = sum(1 for m in ms if m["next_token_match"]) / len(ms)
            lines.append(
                f"| {name} | {avg_cos:.6f} | {avg_mse:.6f} | {avg_topk:.3f} | {match:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Per-prompt Metrics",
            "",
            "| id | category | model | cosine | MSE | top-k overlap | next-token match | tok/s |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for path in quantized_paths:
        for row, metric in zip(quant_rows[path], metrics[path], strict=True):
            template = (
                "| {id} | {category} | `{model}` | {cos:.6f} | {mse:.6f} | "
                "{topk:.3f} | {match} | {tps:.2f} |"
            )
            lines.append(
                template.format(
                    id=row["id"],
                    category=row["category"],
                    model=Path(path).name,
                    cos=metric["cosine"],
                    mse=metric["mse"],
                    topk=metric["topk_overlap"],
                    match="yes" if metric["next_token_match"] else "no",
                    tps=row["tokens_per_s"],
                )
            )
    lines.extend(["", "## Greedy Generations", ""])
    for base in baseline_rows:
        lines.append(f"### {base['id']} / {base['category']}")
        lines.append("")
        lines.append(f"Prompt: `{base['prompt']}`")
        lines.append("")
        lines.append(f"- f16: {base['generation']}")
        for path in quantized_paths:
            row = next(r for r in quant_rows[path] if r["id"] == base["id"])
            lines.append(f"- {Path(path).name}: {row['generation']}")
        lines.append("")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/eval/qwen3_quant_prompts.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--quantized", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--transformers_reference", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--compute_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    parser.add_argument("--json_output", default="reports/qwen3_quant_suite.json")
    parser.add_argument("--md_output", default="reports/qwen3_quant_suite.md")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    tokenizer = TokenizerAdapter(args.tokenizer)
    baseline_rows, baseline_logits = evaluate_gguf_model(
        args.baseline, tokenizer, prompts, args.device, args.compute_dtype
    )

    quant_rows: dict[str, list[dict]] = {}
    metrics: dict[str, list[dict]] = {}
    quant_logits: dict[str, list[torch.Tensor]] = {}
    for path in args.quantized:
        rows, logits = evaluate_gguf_model(
            path, tokenizer, prompts, args.device, args.compute_dtype
        )
        quant_rows[path] = rows
        quant_logits[path] = logits
        metrics[path] = compare_series(baseline_rows, baseline_logits, rows, logits, args.top_k)

    transformers_rows = None
    transformers_logits = None
    ablations: dict[str, list[dict]] = {}
    if args.transformers_reference:
        transformers_rows, transformers_logits = evaluate_transformers_model(
            args.transformers_reference,
            tokenizer,
            prompts,
            args.device,
            args.compute_dtype,
        )
        ablations["f16 GGUF vs Transformers"] = compare_series(
            transformers_rows, transformers_logits, baseline_rows, baseline_logits, args.top_k
        )
        for path in args.quantized:
            ablations[f"{Path(path).name} vs Transformers"] = compare_series(
                transformers_rows,
                transformers_logits,
                quant_rows[path],
                quant_logits[path],
                args.top_k,
            )

    output_obj = {
        "baseline": args.baseline,
        "quantized": args.quantized,
        "transformers_reference": args.transformers_reference,
        "prompts": prompts,
        "device": args.device,
        "compute_dtype": args.compute_dtype,
        "baseline_rows": baseline_rows,
        "quant_rows": quant_rows,
        "metrics": metrics,
        "transformers_rows": transformers_rows,
        "ablations": ablations,
    }
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(
        json.dumps(output_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(
        args.md_output,
        args.baseline,
        args.quantized,
        baseline_rows,
        quant_rows,
        metrics,
        ablations,
        args.device,
        args.compute_dtype,
    )
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()

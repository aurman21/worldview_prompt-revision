"""
Batch image generation via Replicate API — SDXL + FLUX 2 Dev.

Generates images for both original and revised prompts from a CSV,
using two open-source image models without built-in revision layers.

Output structure:
  images/ablation/sdxl/original/{prompt_id}_o.png
  images/ablation/sdxl/revised/{prompt_id}_r.png
  images/ablation/flux2dev/original/{prompt_id}_o.png
  images/ablation/flux2dev/revised/{prompt_id}_r.png

Setup:
  pip install replicate requests
  export REPLICATE_API_TOKEN=r8_your_token_here

Usage:
  python generate_ablation_images.py                  # both models
  python generate_ablation_images.py --model sdxl     # SDXL only
  python generate_ablation_images.py --model flux2dev # Flux only
  python generate_ablation_images.py --dry-run        # cost estimate only
"""

import argparse
import csv
import os
import time
import requests
import replicate
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "sdxl": {
        "replicate_id": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "folder": "sdxl",
        "input_fn": lambda prompt, seed: {
            "prompt": prompt, "width": 1024, "height": 1024,
            "num_inference_steps": 25, "guidance_scale": 7.5,
            "seed": seed, "disable_safety_checker": True,
        },
        "cost_per_image": 0.008,
    },
    "flux2dev": {
        "replicate_id": "black-forest-labs/flux-2-dev",
        "folder": "flux2dev",
        "input_fn": lambda prompt, seed: {
            "prompt": prompt, "width": 1024, "height": 1024,
            "seed": seed, "num_inference_steps": 28, "guidance_scale": 3.5,
            "output_format": "png", "output_quality": 90,
            "disable_safety_checker": True,
        },
        "cost_per_image": 0.012,
    },
}

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR       = "data"
CSV_PATH       = os.path.join(DATA_DIR, "prompts_sdrun.csv")
OUTPUT_DIR     = "images/ablation"
MAX_WORKERS    = 8
SEED           = 21   # fixed for reproducibility; -1 for random
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5


# =============================================================================
# HELPERS
# =============================================================================

def setup_dirs(model_keys):
    for key in model_keys:
        folder = MODELS[key]["folder"]
        Path(f"{OUTPUT_DIR}/{folder}/original").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/{folder}/revised").mkdir(parents=True, exist_ok=True)


def load_prompts(csv_path):
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["prompt_id"].strip()
            original = row["original_prompt"].strip()
            revised = row["revised_prompt"].strip()
            prompts.append({
                "prompt_id": pid,
                "original": original,
                "revised": revised if revised and revised != "NA" else None,
            })
    return prompts


def generate_and_save(model_id, inputs, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000:
        return "SKIP"
    if os.path.exists(output_path):
        os.remove(output_path)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            output = replicate.run(model_id, input=inputs, use_file_output=False)
            if isinstance(output, list):
                url = str(output[0])
            elif isinstance(output, str):
                url = output
            else:
                url = str(next(iter(output)))

            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            if len(resp.content) > 100_000:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return "OK"
            else:
                raise ValueError(f"Image too small ({len(resp.content)} bytes)")
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return f"FAIL ({e})"
    return "FAIL (unknown)"


def build_tasks(prompts, model_keys, seed):
    tasks = []
    for key in model_keys:
        cfg = MODELS[key]
        folder = cfg["folder"]
        for p in prompts:
            tasks.append({
                "model_key": key, "model_id": cfg["replicate_id"],
                "inputs": cfg["input_fn"](p["original"], seed),
                "path": f"{OUTPUT_DIR}/{folder}/original/{p['prompt_id']}_o.png",
                "label": f"{key}/original/{p['prompt_id']}",
            })
            if p["revised"]:
                tasks.append({
                    "model_key": key, "model_id": cfg["replicate_id"],
                    "inputs": cfg["input_fn"](p["revised"], seed),
                    "path": f"{OUTPUT_DIR}/{folder}/revised/{p['prompt_id']}_r.png",
                    "label": f"{key}/revised/{p['prompt_id']}",
                })
    return tasks


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch image generation via Replicate")
    parser.add_argument("--model", choices=["sdxl", "flux2dev", "both"], default="both")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    model_keys = ["sdxl", "flux2dev"] if args.model == "both" else [args.model]
    seed = args.seed if args.seed >= 0 else None

    if not os.environ.get("REPLICATE_API_TOKEN") and not args.dry_run:
        print("ERROR: Set REPLICATE_API_TOKEN environment variable.")
        return

    prompts = load_prompts(args.csv)
    print(f"Loaded {len(prompts)} rows from {args.csv}")

    tasks = build_tasks(prompts, model_keys, seed)

    # Cost estimate
    cost_by_model = {}
    count_by_model = {}
    for t in tasks:
        mk = t["model_key"]
        cost_by_model[mk] = cost_by_model.get(mk, 0) + MODELS[mk]["cost_per_image"]
        count_by_model[mk] = count_by_model.get(mk, 0) + 1

    print(f"\n{'Model':<12} {'Images':>8} {'Est. cost':>10}")
    print("-" * 32)
    for mk in model_keys:
        print(f"{mk:<12} {count_by_model.get(mk, 0):>8} {cost_by_model.get(mk, 0):>9.2f}$")
    print(f"{'TOTAL':<12} {len(tasks):>8} {sum(cost_by_model.values()):>9.2f}$")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    setup_dirs(model_keys)
    print(f"\nStarting generation with {args.workers} workers...\n")

    done = ok = failed = skipped = 0
    total = len(tasks)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_and_save, t["model_id"], t["inputs"], t["path"]): t
            for t in tasks
        }
        for future in as_completed(futures):
            t = futures[future]
            status = future.result()
            done += 1
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skipped += 1
            else:
                failed += 1
                print(f"  FAIL [{done}/{total}] {t['label']}: {status}")
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = (ok + skipped) / elapsed * 3600 if elapsed > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.0f}s | "
                      f"{ok} ok, {skipped} skip, {failed} fail | ~{rate:.0f} img/hr")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | Generated: {ok} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()

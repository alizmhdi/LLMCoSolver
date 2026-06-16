#!/usr/bin/env python3
"""Build HuggingFace eval dataset + instances.pkl from CS SFT JSON (no instance field)."""
import argparse
import json
import os
import re


_JOB_RE = re.compile(r"Job (\d+), throughput: (\d+), gpus: (\d+)")


def parse_instance_from_input(input_text: str, num_gpus: float):
    throughputs = []
    gpu_counts = []
    for match in _JOB_RE.finditer(input_text):
        throughputs.append(int(match.group(2)))
        gpu_counts.append(int(match.group(3)))
    if not throughputs:
        raise ValueError(f"Could not parse jobs from input: {input_text[:120]!r}...")
    return [throughputs, gpu_counts, float(num_gpus)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SFT eval JSON path")
    parser.add_argument(
        "--output-json",
        default="data/cs/eval/test.json",
        help="Output JSON with parsed instance fields (for eval.py)",
    )
    parser.add_argument(
        "--instances-pkl",
        default="data/cs/instances.pkl",
        help="Pickle of raw [throughputs, gpu_counts, num_gpus] per row",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    instances = []
    enriched = []
    for row in records:
        row = dict(row)
        row["instance"] = parse_instance_from_input(row["input"], float(row["num_gpus"]))
        instances.append(row["instance"])
        enriched.append(row)

    json_dir = os.path.dirname(args.output_json)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f)

    pkl_dir = os.path.dirname(args.instances_pkl)
    if pkl_dir:
        os.makedirs(pkl_dir, exist_ok=True)
    import pickle

    with open(args.instances_pkl, "wb") as f:
        pickle.dump(instances, f, pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(enriched)} examples -> {args.output_json}")
    print(f"Saved instances.pkl -> {args.instances_pkl}")


if __name__ == "__main__":
    main()

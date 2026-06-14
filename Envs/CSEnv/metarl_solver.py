"""MetaRL inference helpers for LLMCO on cluster scheduling (single-instance solve)."""
from __future__ import annotations

import os
import re
import sys

import numpy as np

_LLMCO_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
)
if _LLMCO_ROOT not in sys.path:
    sys.path.insert(0, _LLMCO_ROOT)

from Envs.CSEnv.CSEnv import _format_job_descriptions  # noqa: E402

ALPACA_PROMPT = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance. 
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

    ### Instruction:{instruction}

    ### Input:{input}

    ### Response:"""


def parse_solution_cs(response: str) -> list[int] | None:
    """Parse scheduled job indices from an LLM completion."""
    pred_match = re.search(r"Schedule:\s*\[([^\]]+)\]", response)
    if not pred_match:
        return None

    schedule_str = pred_match.group(1)
    try:
        return [int(x.strip()) for x in schedule_str.split(",") if x.strip() != ""]
    except ValueError:
        return None


def build_cs_prompt_fields(jobs, num_gpus, top_k_density: int = 3) -> dict[str, str]:
    """Build instruction/input fields for one cluster scheduling instance."""
    jobs = np.asarray(jobs, dtype=np.float64)
    num_jobs = len(jobs)

    instruction = (
        f"Solve the GPU cluster scheduling problem with {num_jobs} jobs. "
        f"Each job j has a throughput c_j when scheduled and requires g_j GPUs. "
        f"The cluster has N = {int(num_gpus)} GPUs in total. "
        "Select a subset of jobs to maximize total throughput subject to "
        "the GPU capacity constraint sum(g_j) <= N. "
        "Each job is either fully scheduled or not scheduled.\n\n"
        "The input lists each job's throughput and GPU requirement, "
        "plus the top throughput/GPU density jobs as a hint. "
        "Provide the solution in the following format:\n"
        "1. Schedule: The list of scheduled job indices.\n"
        "2. Objective: The total throughput of scheduled jobs."
    )

    return {
        "instruction": instruction,
        "input": _format_job_descriptions(jobs, top_k_density=top_k_density),
    }


def build_prompt(jobs, num_gpus, top_k_density: int = 3) -> str:
    """Build the Alpaca user prompt for one cluster scheduling instance."""
    fields = build_cs_prompt_fields(jobs, num_gpus, top_k_density=top_k_density)
    return ALPACA_PROMPT.format(
        instruction=fields["instruction"],
        input=fields["input"],
    )


def _validate_cs_schedule(
    schedule: list[int] | None,
    throughputs,
    gpu_counts,
    num_gpus: float,
) -> list[int] | None:
    """Return a feasible schedule or None."""
    if not schedule:
        return None

    throughputs = np.asarray(throughputs, dtype=np.float64)
    gpu_counts = np.asarray(gpu_counts, dtype=np.float64)
    num_jobs = len(throughputs)

    if not all(0 <= j < num_jobs for j in schedule):
        return None
    if len(schedule) != len(set(schedule)):
        return None

    total_gpus = sum(float(gpu_counts[j]) for j in schedule)
    if total_gpus > float(num_gpus) + 1e-9:
        return None

    return [int(j) for j in schedule]


def evaluate_completion(response: str, jobs, num_gpus) -> tuple[float | None, np.ndarray | None]:
    """Parse an LLM completion and return (total_throughput, schedule)."""
    jobs = np.asarray(jobs, dtype=np.float64)
    throughputs = jobs[:, 0]
    gpu_counts = jobs[:, 1]

    schedule = _validate_cs_schedule(
        parse_solution_cs(response),
        throughputs,
        gpu_counts,
        num_gpus,
    )
    if schedule is None:
        return None, None

    total_throughput = float(sum(float(throughputs[j]) for j in schedule))
    return total_throughput, np.array(schedule, dtype=int)


def run_llmco_cs(
    jobs,
    num_gpus,
    call_llm,
    num_samples: int = 8,
    top_k_density: int = 3,
    verbose: bool = False,
):
    """Best-of-n LLMCO inference for one cluster scheduling instance.

    Parameters
    ----------
    jobs : array-like, shape (M, 2)
        Columns are [throughput, gpu_count].
    num_gpus : float
        Cluster GPU capacity.
    call_llm : Callable[[list[str]], list[str]]
        Function mapping a list of prompts to LLM response strings.

    Returns
    -------
    best_throughput : float or None
    best_schedule : np.ndarray or None
    """
    prompt = build_prompt(jobs, num_gpus, top_k_density=top_k_density)
    responses = call_llm([prompt] * num_samples)

    best_throughput = None
    best_schedule = None
    n_valid = 0

    for response in responses:
        throughput, schedule = evaluate_completion(response, jobs, num_gpus)
        if schedule is None:
            if verbose and best_schedule is None and n_valid == 0:
                print(f"[LLMCO-CS] sample invalid completion: {response[:200]!r}")
            continue
        n_valid += 1
        if best_throughput is None or throughput > best_throughput:
            best_throughput = throughput
            best_schedule = schedule

    if verbose:
        print(
            f"[LLMCO-CS] {n_valid}/{num_samples} valid schedules; "
            f"best throughput={best_throughput}"
        )

    if best_schedule is None:
        return None, None
    return best_throughput, best_schedule

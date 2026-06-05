"""MetaRL inference helpers for LLMCO on TSP (single-instance solve)."""
from __future__ import annotations  # PEP 585/604 hints on Python 3.8/3.9

import os
import sys

import numpy as np

_LLMCO_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
)
if _LLMCO_ROOT not in sys.path:
    sys.path.insert(0, _LLMCO_ROOT)

from Envs.TSPEnv.TSPEnv import (  # noqa: E402
    build_tsp_prompt_fields,
    calculate_total_distance,
    compute_euclidean_distance_matrix,
    parse_tsp_route,
)

ALPACA_PROMPT = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

    ### Instruction:{instruction}

    ### Input:{input}

    ### Response:"""


def build_prompt(coords, k_nn: int = 2) -> str:
    """Build the Alpaca user prompt for a coordinate matrix."""
    tsp_json = build_tsp_prompt_fields(np.asarray(coords, dtype=np.float32), k_nn=k_nn)
    return ALPACA_PROMPT.format(
        instruction=tsp_json["instruction"],
        input=tsp_json["input"],
    )


def evaluate_completion(response: str, coords) -> tuple[float | None, np.ndarray | None]:
    """Parse an LLM completion and return (tour_length, tour)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    tour = parse_tsp_route(response)
    if tour is None or len(set(tour)) != n:
        return None, None
    dist_matrix = compute_euclidean_distance_matrix(coords)
    length = float(calculate_total_distance(tour, dist_matrix))
    return length, np.array(tour, dtype=int)


def run_llmco_tsp(
    coords,
    call_llm,
    num_samples: int = 8,
    k_nn: int = 2,
    verbose: bool = False,
):
    """Best-of-n LLMCO inference for one TSP instance.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
    call_llm : Callable[[list[str]], list[str]]
        Function mapping a list of prompts to LLM response strings
        (e.g. ``opro.prompt_utils.call_vllm_server_func``).

    Returns
    -------
    best_length : float or None
    best_tour : np.ndarray or None
    """
    prompt = build_prompt(coords, k_nn=k_nn)
    responses = call_llm([prompt] * num_samples)

    best_length = None
    best_tour = None
    n_valid = 0

    for response in responses:
        length, tour = evaluate_completion(response, coords)
        if tour is None:
            continue
        n_valid += 1
        if best_length is None or length < best_length:
            best_length = length
            best_tour = tour

    if verbose:
        print(
            f"[LLMCO-TSP] {n_valid}/{num_samples} valid tours; "
            f"best tour_length={best_length}"
        )

    if best_tour is None:
        return None, None
    return best_length, best_tour

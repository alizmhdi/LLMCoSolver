"""TEEnv: Traffic Engineering environment for LLMCO training and inference.

This module is self-contained (no MetaRL imports) and provides:
  - Alpaca-style prompt building from TE instances (traffic matrices + topology).
  - Conversion of LP optimal solutions (continuous path flows) to discrete
    path-index routings suitable for LLM training targets.
  - Lightweight routing evaluation for scoring LLM outputs.

The output format uses the same ``<routing> i,j,k,... </routing>`` convention
as the OPRO TE solver so that both solvers can share the same parsing logic.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Topology helpers (standalone, no MetaRL import required)
# ---------------------------------------------------------------------------

def _path_to_edges(path):
    """Yield ``(u, v)`` pairs for consecutive nodes in *path*."""
    it = iter(path)
    prev = next(it)
    for node in it:
        yield (prev, node)
        prev = node


def build_te_structures(graph, node_names, paths_dict):
    """Build commodity / path lists from MetaRL topology kwargs.

    Returns
    -------
    commodities : list of (commod_id, s_idx, t_idx, path_edge_lists)
        ``path_edge_lists[i]`` is a list of ``(u, v)`` tuples for local path *i*.
    edge_cap : dict {(u, v): float}
    od_pair_labels : list of str
        Human-readable ``"OD_{id}: ({s}→{t})"`` labels.
    od_pair_paths_text : list of str
        Human-readable path descriptions.
    """
    commodities: list = []
    od_pair_labels: list = []
    od_pair_paths_text: list = []
    commod_id = 0

    for s_idx, s in enumerate(node_names):
        for t_idx, t in enumerate(node_names):
            if s == t:
                continue
            paths = paths_dict.get((s, t), [])
            if not paths:
                continue

            path_edge_lists = []
            path_strs = []
            for k, path in enumerate(paths):
                edges = list(_path_to_edges(path))
                path_edge_lists.append(edges)
                path_strs.append(f"P{k}=[{','.join(str(n) for n in path)}]")

            commodities.append((commod_id, s_idx, t_idx, path_edge_lists))
            od_pair_labels.append(f"OD_{commod_id}: ({s}→{t})")
            od_pair_paths_text.append(
                f"  OD_{commod_id} ({s}→{t}): " + ", ".join(path_strs)
            )
            commod_id += 1

    edge_cap = {
        (u, v): float(data.get("capacity", 0.0))
        for u, v, data in graph.edges(data=True)
    }
    return commodities, edge_cap, od_pair_labels, od_pair_paths_text


# ---------------------------------------------------------------------------
# Routing evaluation (mirrors OPRO's logic, kept standalone here)
# ---------------------------------------------------------------------------

def evaluate_routing(routing_indices, tm, commodities, edge_cap, objective):
    """Return objective value for a discrete path-index routing."""
    if objective == "min_max_link_util":
        return _eval_mlu(routing_indices, tm, commodities, edge_cap)
    return _eval_total_flow(routing_indices, tm, commodities, edge_cap)


def _eval_mlu(routing_indices, tm, commodities, edge_cap):
    edge_flow: dict = defaultdict(float)
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        for u, v in path_edge_lists[local_idx]:
            edge_flow[(u, v)] += demand
    mlu = 0.0
    for (u, v), cap in edge_cap.items():
        if cap > 0.0:
            mlu = max(mlu, edge_flow.get((u, v), 0.0) / cap)
    return mlu


def _eval_total_flow(routing_indices, tm, commodities, edge_cap):
    remaining_cap = dict(edge_cap)
    total_flow = 0.0
    sorted_commods = sorted(commodities, key=lambda c: -float(tm[c[1], c[2]]))
    for commod_id, s_idx, t_idx, path_edge_lists in sorted_commods:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        edges = path_edge_lists[local_idx]
        if not edges:
            continue
        bottleneck = min(remaining_cap.get((u, v), 0.0) for u, v in edges)
        allocated = min(demand, max(0.0, bottleneck))
        if allocated > 0.0:
            total_flow += allocated
            for u, v in edges:
                remaining_cap[(u, v)] = remaining_cap.get((u, v), 0.0) - allocated
    return total_flow


def build_edge_allocation(routing_indices, tm, commodities, node_names):
    """Return ``{(s_name, t_name): {(u,v): flow}}`` for the given routing."""
    allocation: dict = defaultdict(lambda: defaultdict(float))
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        s_name = node_names[s_idx]
        t_name = node_names[t_idx]
        for u, v in path_edge_lists[local_idx]:
            allocation[(s_name, t_name)][(u, v)] += demand
    return {k: dict(v) for k, v in allocation.items()}


# ---------------------------------------------------------------------------
# LP → discrete conversion
# ---------------------------------------------------------------------------

def lp_solution_to_path_indices(lp_path_flows, commodities_lp):
    """Discretize LP path flows to one integer path index per OD pair.

    For each commodity, picks the local path with the highest flow.

    Parameters
    ----------
    lp_path_flows : list[float]
        Flow on each *global* path index (order matches
        ``PathOptimalSolver.path_vars`` keys 0..N-1).
    commodities_lp : list
        ``(commod_id, s_idx, t_idx, [global_path_indices])`` as stored by
        ``PathOptimalSolver.commodities``.

    Returns
    -------
    list[int]
        Local 0-based path index per commodity (in commod_id order).
    """
    routing: list = []
    for _commod_id, _s, _t, global_path_indices in commodities_lp:
        if not global_path_indices:
            routing.append(0)
            continue
        flows = [lp_path_flows[g] for g in global_path_indices]
        routing.append(int(np.argmax(flows)))
    return routing


# ---------------------------------------------------------------------------
# Prompt / output formatting
# ---------------------------------------------------------------------------

def _edge_summary(graph, top_n: int = 20) -> str:
    edges = sorted(
        [(u, v, data.get("capacity", 0.0)) for u, v, data in graph.edges(data=True)],
        key=lambda x: -x[2],
    )[:top_n]
    return "\n".join(f"  ({u},{v}): cap={c:.0f}" for u, v, c in edges)


def build_instruction(objective: str) -> str:
    """Return the Alpaca 'instruction' field for a TE problem."""
    if objective == "total_flow":
        obj_desc = "maximize Total Flow (total traffic successfully routed)"
        direction = "higher"
        obj_label = "Total Flow"
    else:
        obj_desc = "minimize Maximum Link Utilisation (MLU)"
        direction = "lower"
        obj_label = "MLU"
    return (
        f"You are solving a Traffic Engineering routing problem. "
        f"Given a network topology, traffic demands, and a set of available paths "
        f"for each OD pair, assign exactly one path to each OD pair to {obj_desc}. "
        f"A {direction} {obj_label} value is better. "
        f"Output a comma-separated list of path indices (0-indexed, one per OD pair "
        f"in the listed order) wrapped in <routing> ... </routing> tags, "
        f"then state the {obj_label} value."
    )


def build_input_str(
    graph,
    node_names: list,
    tm,
    commodities: list,
    od_pair_labels: list,
    od_pair_paths_text: list,
    max_paths_show: int = 60,
) -> str:
    """Return the Alpaca 'input' field string for one TE instance."""
    n_nodes = len(node_names)
    n_edges = graph.number_of_edges()

    topo_str = (
        f"Nodes: {n_nodes} ({', '.join(str(n) for n in node_names)})\n"
        f"Directed edges ({n_edges} total), top by capacity:\n"
        + _edge_summary(graph)
    )

    # Non-zero demands only to keep the prompt short
    tm_lines = []
    for commod_id, s_idx, t_idx, _ in commodities:
        d = float(tm[s_idx, t_idx])
        if d > 0.0:
            tm_lines.append(f"  {od_pair_labels[commod_id]}: demand={d:.1f}")
    tm_str = "\n".join(tm_lines) if tm_lines else "  (all demands are zero)"

    if len(od_pair_paths_text) <= max_paths_show:
        paths_str = "\n".join(od_pair_paths_text)
    else:
        paths_str = (
            "\n".join(od_pair_paths_text[:max_paths_show])
            + f"\n  ... ({len(od_pair_paths_text) - max_paths_show} more OD pairs omitted)"
        )

    num_od = len(commodities)
    max_paths_per_od = max(
        (len(pel) for _, _, _, pel in commodities), default=1
    )

    return (
        f"=== Network ===\n{topo_str}\n\n"
        f"=== Traffic Demands ===\n{tm_str}\n\n"
        f"=== Available Paths ===\n"
        f"Each OD pair has up to {max_paths_per_od} paths (0-indexed):\n{paths_str}\n\n"
        f"Number of OD pairs: {num_od}"
    )


def routing_to_output_str(
    routing_indices: list,
    obj_value: Optional[float],
    objective: str,
) -> str:
    """Format routing as the Alpaca 'output' field string."""
    routing_str = ",".join(str(x) for x in routing_indices)
    if obj_value is not None:
        obj_label = "Total Flow" if objective == "total_flow" else "MLU"
        return f"<routing> {routing_str} </routing>\n{obj_label}: {obj_value:.4f}"
    return f"<routing> {routing_str} </routing>"


def extract_routing(input_string: str, num_od_pairs: int) -> Optional[List[int]]:
    """Parse LLM output to extract a routing (list of path indices).

    Mirrors ``opro.optimization.optimize_te.extract_routing`` so both solvers
    can share the same output convention.
    """
    if not input_string:
        return None
    match = re.search(r"<routing>(.*?)</routing>", input_string, re.DOTALL | re.IGNORECASE)
    raw = match.group(1) if match else input_string
    tokens = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        tok = tok.strip().strip(".,;[](){}")
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            tokens.append(int(tok))
    if len(tokens) != num_od_pairs:
        return None
    return tokens


# ---------------------------------------------------------------------------
# TEEnv class
# ---------------------------------------------------------------------------

class TEEnv:
    """Traffic Engineering environment for LLMCO prompt generation.

    Wraps topology structures and provides helpers to build Alpaca-style
    prompts and convert LP optimal solutions into training labels.

    Parameters
    ----------
    graph : networkx.DiGraph
        Topology graph with ``capacity`` edge attributes.
    node_names : list
        Ordered node IDs.
    paths_dict : dict
        ``{(s, t): [[node, ...], ...]}`` pre-computed k-shortest paths.
    objective : str
        ``'total_flow'`` or ``'min_max_link_util'``.
    """

    def __init__(self, graph, node_names, paths_dict, objective: str = "total_flow"):
        self.graph = graph
        self.node_names = list(node_names)
        self.paths_dict = paths_dict
        self.objective = objective

        (
            self.commodities,
            self.edge_cap,
            self.od_pair_labels,
            self.od_pair_paths_text,
        ) = build_te_structures(graph, node_names, paths_dict)

        self.num_od = len(self.commodities)
        self.instruction = build_instruction(objective)

    # ------------------------------------------------------------------
    # Prompt / dataset helpers
    # ------------------------------------------------------------------

    def build_input_str(self, tm) -> str:
        """Return the Alpaca 'input' field for this traffic matrix."""
        return build_input_str(
            self.graph,
            self.node_names,
            tm,
            self.commodities,
            self.od_pair_labels,
            self.od_pair_paths_text,
        )

    def lp_to_routing(self, lp_path_flows, commodities_lp) -> List[int]:
        """Discretize LP flows (PathOptimalSolver output) to path indices."""
        return lp_solution_to_path_indices(lp_path_flows, commodities_lp)

    def routing_to_output_str(
        self,
        routing_indices: list,
        obj_value: Optional[float] = None,
    ) -> str:
        """Return the Alpaca 'output' field for the given routing."""
        return routing_to_output_str(routing_indices, obj_value, self.objective)

    def make_dataset_entry(
        self,
        tm,
        routing_indices: list,
        obj_value: Optional[float] = None,
    ) -> dict:
        """Return one Alpaca-style dataset entry ``{instruction, input, output}``."""
        return {
            "instruction": self.instruction,
            "input": self.build_input_str(tm),
            "output": self.routing_to_output_str(routing_indices, obj_value),
        }

    # ------------------------------------------------------------------
    # Evaluation / allocation
    # ------------------------------------------------------------------

    def evaluate(self, routing_indices: list, tm) -> float:
        """Compute the objective value for a discrete routing."""
        return evaluate_routing(
            routing_indices, tm, self.commodities, self.edge_cap, self.objective
        )

    def build_edge_allocation(self, routing_indices: list, tm) -> dict:
        """Return ``{(s_name, t_name): {(u,v): flow}}`` for the given routing."""
        return build_edge_allocation(
            routing_indices, tm, self.commodities, self.node_names
        )

    def extract_routing(self, llm_output: str) -> Optional[List[int]]:
        """Parse LLM output text to a list of path indices."""
        return extract_routing(llm_output, self.num_od)

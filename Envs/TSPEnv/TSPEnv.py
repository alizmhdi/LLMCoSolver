from __future__ import annotations

import json
import random
import pickle
import numpy as np
import torch
from scipy.spatial import cKDTree, distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : np.ndarray
        Coordinates of the first point.
    point2 : np.ndarray
        Coordinates of the second point.

    Returns
    -------
    float
        Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def compute_euclidean_distance_matrix(locations: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for a set of points.

    Parameters
    ----------
    locations : np.ndarray
        Array of shape (N, 2) representing the coordinates of N nodes.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, N) where entry (i, j) is the distance between
        node i and node j.
    """
    # You could also do:
    # dist_matrix = distance.cdist(locations, locations, 'euclidean')
    # np.fill_diagonal(dist_matrix, 0)
    # return dist_matrix

    num_nodes = locations.shape[0]
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return dist_matrix


def calculate_total_distance(tour: list[int], dist_matrix: np.ndarray) -> float:
    """
    Calculate the total distance of a TSP tour using a given distance matrix.

    Parameters
    ----------
    tour : list[int]
        The tour path as a list of node indices.
    dist_matrix : np.ndarray
        The full pairwise distance matrix.

    Returns
    -------
    float
        The total distance of the tour, including the return to the starting city.
    """
    total_dist = 0.0
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        total_dist += dist_matrix[from_node][to_node]

    # Add the distance from the last node back to the starting node
    total_dist += dist_matrix[tour[-1]][tour[0]]
    return total_dist


def lkh(problem: torch.Tensor) -> tuple[list[int], float]:
    """
    Solve the TSP using the LKH (via elkai) solver.

    Parameters
    ----------
    problem : torch.Tensor
        A tensor of shape (N, 2) containing node coordinates.

    Returns
    -------
    tuple[list[int], float]
        A tuple where the first element is the best tour (list of node indices),
        and the second is the total distance of that tour.
    """
    import elkai  # optional; only required for LKH ground-truth labelling

    if isinstance(problem, torch.Tensor):
        locations = problem.detach().cpu().numpy()
    else:
        locations = np.array(problem)

    dist_matrix = compute_euclidean_distance_matrix(locations)
    cities = elkai.DistanceMatrix(dist_matrix)
    tour = cities.solve_tsp(runs=10)
    cost = calculate_total_distance(tour, dist_matrix)
    return tour, cost


def generate_weights(start: int, end: int) -> list[int]:
    """
    Generate weights for weighted random sampling (1 to n).

    Parameters
    ----------
    start : int
        Start of the range (inclusive).
    end : int
        End of the range (inclusive).

    Returns
    -------
    list[int]
        List of weights corresponding to the range [start, end].
    """
    return [i for i in range(start, end + 1)]


def weighted_random_choice(start: int, end: int) -> int:
    """
    Choose a random number from start to end (inclusive) with weighted probability.

    Parameters
    ----------
    start : int
        Start of the range (inclusive).
    end : int
        End of the range (inclusive).

    Returns
    -------
    int
        A randomly chosen integer in [start, end].
    """
    numbers = list(range(start, end + 1))
    weights = generate_weights(start, end)
    return random.choices(numbers, weights=weights, k=1)[0]


def calculate_top_k_nearest_nodes(nodes: np.ndarray, k: int = 2) -> list[list[tuple[int, float]]]:
    """
    For each node, calculate its top k nearest neighbors using a k-d tree.

    Parameters
    ----------
    nodes : np.ndarray
        Coordinates of the nodes. Shape: (N, 2).
    k : int, optional
        Number of nearest neighbors to find for each node. Default is 2.

    Returns
    -------
    list[list[tuple[int, float]]]
        A list of length N, where each element is a list of k tuples (neighbor_index, distance).
    """
    kdtree = cKDTree(nodes)
    top_k_nearest_nodes = []
    for node in nodes:
        distances, indices = kdtree.query(node, k + 1)  # k+1 to include the node itself
        # Exclude the node itself (first index)
        distances, indices = distances[1:], indices[1:]
        neighbors = [(idx, dist) for idx, dist in zip(indices, distances)]
        top_k_nearest_nodes.append(neighbors)
    return top_k_nearest_nodes


def parse_tsp_route(response: str):
    """Parse ``Route: [0, 1, 2, ...]`` from an LLM completion (TSP training format)."""
    import re

    match = re.search(r"Route:\s*\[([^\]]+)\]", response or "")
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None


def build_tsp_prompt_fields(instance, k_nn=2):
    """Build instruction and input text for a TSP instance (no LKH / elkai).

    Used by MetaRL inference and by :func:`tag_prompt_and_transform_to_json`.
    """
    if isinstance(instance, torch.Tensor):
        nodes = instance.detach().cpu().numpy()
        coord_at = lambda i: instance[i].tolist()
    else:
        nodes = np.asarray(instance)
        coord_at = lambda i: nodes[i].tolist()

    p_size = nodes.shape[0]
    instruction = (
        f"Solve the Traveling Salesman Problem (TSP) for the given list of {p_size} cities. "
        "Each city is represented as a node with coordinates (x, y). "
        "Identify the shortest route that visits every city exactly once and returns to the starting city. "
        f"The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. "
        "Provide the solution in the following format:\n\n"
        "1. Route: List the nodes in the order they are visited.\n"
        "2. Objective: The objective value (total travel distance)."
    )

    nns = calculate_top_k_nearest_nodes(nodes, k_nn)
    nodes_description = []
    for i in range(p_size):
        neighbor_str = [f"{n[0]}: {n[1]:.1f}" for n in nns[i]]
        node_desc = (
            f"Node {i}, coordinates: {coord_at(i)}, "
            f"neighbors (node_index: distance): [{', '.join(neighbor_str)}]"
        ).replace("\'", "")
        nodes_description.append(node_desc)

    input_text = ";".join(nodes_description) + "."

    return {
        "num_nodes": str(p_size),
        "instruction": instruction,
        "input": input_text,
    }


def tag_prompt_and_transform_to_json(instance, k_nn=2):
    """

    Combines tagging and JSON transformation for a TSP instance.

    Parameters:
        instance (torch.Tensor): Input tensor of node coordinates.
        k_nn (int): Number of nearest neighbors to include in the description.

    Returns:
        dict: JSON-ready dictionary containing the TSP instance description with all numerical results as text.
    """
    tsp_json = build_tsp_prompt_fields(instance, k_nn=k_nn)

    # LKH optimal tour for training labels (requires elkai)
    tour, cost = lkh(instance)
    tsp_json["output"] = (
        "Route: " + str([node for node in tour]) + ", Objective: " + f"{cost:.3f}"
    )

    return tsp_json


class TSPEnv:
    """
    A TSP Environment class to generate TSP instances (node coordinates)
    using different distributions and save them in JSON format.
    """

    def __init__(
            self,
            n_node_range: list[int],
            distributions: list[str],
            seed: int | None = None,
            n_c: int = 3,
            std_cluster: float = 0.07
    ) -> None:
        """
        Parameters
        ----------
        n_node_range : list[int]
            [min_nodes, max_nodes] range for random TSP instance size.
        distributions : list[str]
            A list of distribution names to sample from. E.g. ['uniform', 'gaussian_mixture_2_5', 'clustered', 'mixed'].
        seed : int or None
            Random seed for reproducibility.
        n_c : int
            Number of cluster centers for clustered and mixed distributions.
        std_cluster : float
            Standard deviation for normal distribution of city clusters.
        """
        self.n_node_range = n_node_range
        self.distributions = distributions
        self.seed = seed
        self.n_c = n_c
        self.std_cluster = std_cluster
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate_clustered_nodes(self, n_nodes: int, mixed: bool = False, max_xy: float = 1.0) -> np.ndarray:
        """
        Generate node coordinates using clustered or mixed distribution.
        
        Parameters
        ----------
        n_nodes : int
            Total number of nodes to generate.
        mixed : bool
            If True, use mixed distribution (uniform + clustered). If False, use pure clustered.
        max_xy : float
            Maximum coordinate value for the space.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_nodes, 2) with node coordinates.
        """
        uniform_frac = 0.5 if mixed else 0.0
        n_uniform = int(n_nodes * uniform_frac)
        n_clustered = n_nodes - n_uniform
        
        # Generate uniform nodes for mixed distribution
        uniform_locs = np.random.uniform(0, max_xy, size=(n_uniform, 2)) if n_uniform > 0 else np.empty((0, 2))
        
        # Generate cluster centers
        assert self.n_c < n_nodes, f"Number of clusters ({self.n_c}) must be less than number of nodes ({n_nodes})"
        centers = np.random.uniform(0.2, max_xy - 0.2, size=(self.n_c, 2))
        
        # Generate clustered nodes around centers
        n_clustered_samples = 0
        all_clustered_locs = []
        while n_clustered_samples < n_clustered:
            # Sample random centers for each point
            center_locs = centers[np.random.randint(len(centers), size=2 * (n_clustered - n_clustered_samples))]
            # Generate points around centers using normal distribution
            cluster_locs = np.random.normal(center_locs, self.std_cluster)
            # Keep only points within bounds
            cluster_locs = cluster_locs[(cluster_locs >= 0).all(axis=1) & (cluster_locs < max_xy).all(axis=1)]
            all_clustered_locs.append(cluster_locs)
            n_clustered_samples += len(cluster_locs)
        
        # Combine all clustered locations and trim to exact number needed
        cluster_locs = np.concatenate(all_clustered_locs)[:n_clustered] if all_clustered_locs else np.empty((0, 2))
        
        # Combine uniform and clustered locations
        if n_uniform > 0 and n_clustered > 0:
            xys = np.vstack((uniform_locs, cluster_locs))
        elif n_uniform > 0:
            xys = uniform_locs
        else:
            xys = cluster_locs
            
        return xys

    def generate_tensor_instances(self, n_instance: int) -> list[torch.Tensor]:
        """
        Generate a list of random TSP instances (as tensors).

        Parameters
        ----------
        n_instance : int
            Number of instances to generate.

        Returns
        -------
        list[torch.Tensor]
            A list where each element is a torch.Tensor of shape (N, 2),
            where N is randomly chosen in [n_node_range[0], n_node_range[1]].
        """
        instances = []
        for _ in range(n_instance):
            size_i = weighted_random_choice(self.n_node_range[0], self.n_node_range[1])
            distribution_i = random.choice(self.distributions)

            # Generate instance based on distribution type
            if distribution_i == 'uniform':
                instance = np.random.uniform(0, 1, [size_i, 2])
            elif distribution_i == 'clustered':
                instance = self.generate_clustered_nodes(size_i, mixed=False, max_xy=1.0)
            elif distribution_i == 'mixed':
                instance = self.generate_clustered_nodes(size_i, mixed=True, max_xy=1.0)
            elif distribution_i == 'gaussian_mixture_2_5':
                modes, cdist_ = 2, 5
                instance = self.generate_gaussian_mixture_tsp(1, size_i, modes, cdist_)[0]
            elif distribution_i == 'gaussian_mixture_3_10':
                modes, cdist_ = 3, 10
                instance = self.generate_gaussian_mixture_tsp(1, size_i, modes, cdist_)[0]
            else:
                raise NotImplementedError(f"Distribution '{distribution_i}' is not defined.")

            # Scale by 1000 and convert to int
            instance = instance * 1000
            instances.append(torch.tensor(instance).int())
        return instances

    def generate_gaussian_mixture_tsp(
            self,
            dataset_size: int,
            graph_size: int,
            num_modes: int = 0,
            cdist_: int = 0
    ) -> np.ndarray:
        """
        Generate TSP instances with Gaussian mixture distribution.

        Adapted from:
        AAAI-2022 "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum".

        Parameters
        ----------
        dataset_size : int
            Number of TSP instances to generate.
        graph_size : int
            Number of nodes per TSP instance.
        num_modes : int
            Number of Gaussian modes to generate the data from. If 0, uniform distribution.
        cdist_ : int
            Range for random centers in the Gaussian mixture.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (dataset_size, graph_size, 2).
        """

        def gaussian_mixture(graph_size=100, modes=0, cdist_val=1):
            """
            Create one TSP instance using a Gaussian mixture model.
            """
            nums = np.random.multinomial(graph_size, np.ones(modes) / modes)
            xy_list = []
            for num in nums:
                center = np.random.uniform(0, cdist_val, size=(1, 2))
                points = np.random.multivariate_normal(
                    mean=center.squeeze(),
                    cov=np.eye(2, 2),
                    size=(num,)
                )
                xy_list.extend(points)

            xy_arr = np.array(xy_list)
            xy_arr = MinMaxScaler().fit_transform(xy_arr)
            return xy_arr

        if num_modes == 0:
            # (0, 0) means uniform
            return np.random.uniform(0, 1, [dataset_size, graph_size, 2])
        else:
            result = []
            for _ in range(dataset_size):
                result.append(
                    gaussian_mixture(
                        graph_size=graph_size,
                        modes=num_modes,
                        cdist_val=cdist_
                    )
                )
            return np.array(result)

    def save_dataset(self, dataset, filename, disable_print=False):
        filedir = os.path.split(filename)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(check_extension(filename), 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        if not disable_print:
            print(">> Save dataset to {}".format(filename))

    def read_and_transform_pkl(self, pkl_file: str, output_file: str, rl_data: bool = False) -> None:
        """
        Read TSP instances from a pickle file and transform them to textual format.

        Parameters
        ----------
        pkl_file : str
            Path to the pickle file containing TSP instances.
        output_file : str
            Path where the resulting JSON file should be saved.
        rl_data : bool
            Whether to include the original instance data in the output.
        """
        # Read the pickle file
        with open(check_extension(pkl_file), 'rb') as f:
            instances = pickle.load(f)
        
        # Transform instances to text format
        tsp_data = []
        for instance in tqdm(instances, desc="Transforming TSP instances"):
            if instance[0][0] < 1:
                instance = torch.tensor(instance*1000).int()
            json_data = tag_prompt_and_transform_to_json(instance)
            if rl_data:
                json_data["instance"] = instance.tolist()
            tsp_data.append(json_data)

        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(tsp_data, f, indent=4)
        print(f">> Saved transformed data to {output_file}")

    def instances_to_json(self, instances: list[torch.Tensor], rl_data: bool = False) -> list[dict]:
        tsp_data = []
        for instance in tqdm(instances, desc="Generating TSP instances"):
            json_data = tag_prompt_and_transform_to_json(instance)
            if rl_data:
                json_data["instance"] = instance.tolist()
            tsp_data.append(json_data)
        return tsp_data

    def generate_instances_and_save(
        self,
        n_instance: int,
        file_name: str,
        save_pkl: bool,
        rl_data: bool = False,
        write_file: bool = True,
    ) -> list[dict]:
        """
        Generate TSP instances and save them to a JSON file.

        Parameters
        ----------
        n_instance : int
            Number of TSP instances to generate.
        file_name : str
            Path where the resulting JSON file should be saved.
        save_pkl : bool
            Whether to save the dataset as a pickle file.
        rl_data : bool
            Whether to generate data for Reinforcement Learning (RL) tasks.
        write_file : bool
            Whether to write the JSON output to ``file_name``.
        """
        instances = self.generate_tensor_instances(n_instance)
        if save_pkl:
            self.save_dataset(instances, "./ttt1000.pkl")
        tsp_data = self.instances_to_json(instances, rl_data=rl_data)
        avg_objective = np.mean([float(data['output'].split('Objective: ')[1]) for data in tsp_data])
        print(f"Average objective value: {avg_objective:.2f}")

        if write_file:
            with open(file_name, 'w') as f:
                json.dump(tsp_data, f, indent=4)
        return tsp_data

    def generate_per_distribution_and_save(
        self,
        n_instance_per_distribution: int,
        file_name: str,
        save_pkl: bool = False,
        rl_data: bool = False,
        append_to: str | None = None,
        replace_num_nodes: int | None = None,
    ) -> list[dict]:
        """Generate a fixed number of instances for each sampler distribution."""
        tsp_data: list[dict] = []
        for dist_idx, distribution in enumerate(self.distributions):
            env = TSPEnv(
                n_node_range=self.n_node_range,
                distributions=[distribution],
                seed=None if self.seed is None else self.seed + dist_idx,
                n_c=self.n_c,
                std_cluster=self.std_cluster,
            )
            print(f"Sampler: {distribution}")
            instances = env.generate_tensor_instances(n_instance_per_distribution)
            if save_pkl and dist_idx == 0:
                env.save_dataset(instances, "./instances.pkl")
            tsp_data.extend(env.instances_to_json(instances, rl_data=rl_data))

        if append_to is not None:
            self._append_json_records(
                tsp_data,
                append_to,
                replace_num_nodes=replace_num_nodes,
            )
            print(f">> Appended {len(tsp_data)} records to {append_to}")
        else:
            with open(file_name, 'w') as f:
                json.dump(tsp_data, f, indent=4)
            print(f">> Saved {len(tsp_data)} records to {file_name}")
        return tsp_data

    @staticmethod
    def _append_json_records(
        records: list[dict],
        target_file: str,
        replace_num_nodes: int | None = None,
    ) -> None:
        filedir = os.path.split(target_file)[0]
        if filedir and not os.path.isdir(filedir):
            os.makedirs(filedir)

        existing: list[dict] = []
        if os.path.exists(target_file):
            with open(target_file) as f:
                existing = json.load(f)

        if replace_num_nodes is not None:
            keep = [
                row for row in existing
                if str(row.get("num_nodes")) != str(replace_num_nodes)
            ]
            removed = len(existing) - len(keep)
            existing = keep
            print(f">> Removed {removed} existing records with num_nodes={replace_num_nodes}")

        merged = existing + records
        with open(target_file, 'w') as f:
            json.dump(merged, f, indent=4)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate TSP SFT/RL datasets with TSPEnv samplers.")
    parser.add_argument('--min_nodes', type=int, default=10, help='Minimum number of cities')
    parser.add_argument('--max_nodes', type=int, default=10, help='Maximum number of cities')
    parser.add_argument('--n_instances', type=int, help='Total number of TSP instances to generate')
    parser.add_argument(
        '--n_instances_per_distribution',
        type=int,
        help='Generate this many instances for each sampler distribution',
    )
    parser.add_argument(
        '--distributions',
        nargs='+',
        default=['uniform', 'gaussian_mixture_2_5', 'gaussian_mixture_3_10', 'clustered', 'mixed'],
        help='Coordinate samplers to use',
    )
    parser.add_argument('--output_file', type=str, default='./data/tsp/eval/val_nodes10.json')
    parser.add_argument('--append_to', type=str, default=None, help='Append generated records to an existing JSON file')
    parser.add_argument(
        '--replace_num_nodes',
        type=int,
        default=None,
        help='Drop existing records with this num_nodes before appending',
    )
    parser.add_argument('--save_pkl', action='store_true', default=False)
    parser.add_argument('--rl_data', action='store_true', default=False, help='Include raw instance coordinates')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_c', type=int, default=3, help='Number of cluster centers')
    parser.add_argument('--std_cluster', type=float, default=0.07, help='Cluster spread for clustered/mixed samplers')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n_instances is None and args.n_instances_per_distribution is None:
        raise SystemExit("Provide --n_instances or --n_instances_per_distribution")

    tsp_env = TSPEnv(
        n_node_range=[args.min_nodes, args.max_nodes],
        distributions=args.distributions,
        seed=args.seed,
        n_c=args.n_c,
        std_cluster=args.std_cluster,
    )

    if args.n_instances_per_distribution is not None:
        tsp_env.generate_per_distribution_and_save(
            n_instance_per_distribution=args.n_instances_per_distribution,
            file_name=args.output_file,
            save_pkl=args.save_pkl,
            rl_data=args.rl_data,
            append_to=args.append_to,
            replace_num_nodes=args.replace_num_nodes,
        )
    else:
        tsp_data = tsp_env.generate_instances_and_save(
            n_instance=args.n_instances,
            file_name=args.output_file,
            save_pkl=args.save_pkl,
            rl_data=args.rl_data,
        )
        if args.append_to is not None:
            TSPEnv._append_json_records(
                tsp_data,
                args.append_to,
                replace_num_nodes=args.replace_num_nodes,
            )
            print(f">> Appended {len(tsp_data)} records to {args.append_to}")

    # Example:
    # python Envs/TSPEnv/TSPEnv.py \
    #   --min_nodes 10 --max_nodes 10 \
    #   --n_instances_per_distribution 10 \
    #   --rl_data \
    #   --append_to ./data/tsp/eval/test.json \
    #   --replace_num_nodes 10
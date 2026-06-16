import argparse
import json
import os
import pickle
import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from tqdm import tqdm


def check_extension(filename):
	if os.path.splitext(filename)[1] != ".pkl":
		return filename + ".pkl"
	return filename


def save_dataset(dataset, filename, disable_print=False):
	filedir = os.path.split(filename)[0]
	if filedir and not os.path.isdir(filedir):
		os.makedirs(filedir)
	with open(check_extension(filename), 'wb') as f:
		pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
	if not disable_print:
		print(f">> Save dataset to {filename}")


def check_gurobi_license():
	"""Fail fast when Gurobi cannot acquire a license (e.g. on Anvil login nodes)."""
	try:
		model = gp.Model("cs_license_check")
		model.setParam("OutputFlag", 0)
		model.setParam("LogToConsole", 0)
		model.dispose()
	except gp.GurobiError as exc:
		raise SystemExit(
			f"Gurobi license check failed: {exc}\n"
			"On Anvil: module load gurobi/9.5.1 && "
			'export GRB_LICENSE_FILE="$GUROBI_HOME/license/gurobi.lic"\n'
			"Then run this script on a compute node via sinteractive or sbatch, "
			"not on a login node."
		) from exc


def solve_cs_gurobi(throughputs, gpu_counts, num_gpus):
	"""Exact 0/1 cluster scheduling via one-shot Gurobi MIP."""
	throughputs = np.asarray(throughputs, dtype=np.float64)
	gpu_counts = np.asarray(gpu_counts, dtype=np.float64)
	n = len(throughputs)
	if n == 0:
		return 0.0, []

	try:
		model = gp.Model("cluster_scheduling")
		model.setParam("OutputFlag", 0)
		model.setParam("LogToConsole", 0)

		x = model.addVars(n, vtype=GRB.BINARY, name="x")
		model.addConstr(
			gp.quicksum(float(gpu_counts[j]) * x[j] for j in range(n)) <= float(num_gpus),
			name="gpu_capacity",
		)
		model.setObjective(
			gp.quicksum(float(throughputs[j]) * x[j] for j in range(n)),
			GRB.MAXIMIZE,
		)
		model.optimize()

		if model.Status != GRB.OPTIMAL:
			return None, None

		selected = [j for j in range(n) if x[j].X > 0.5]
		return float(model.ObjVal), selected
	except gp.GurobiError as exc:
		print(f"Gurobi error in solve_cs_gurobi: {exc}")
		return None, None


def _format_job_descriptions(jobs, top_k_density=3):
	"""Build input string listing jobs and top-k throughput/GPU density hints."""
	throughputs = jobs[:, 0]
	gpu_counts = jobs[:, 1]
	num_jobs = len(jobs)

	density = throughputs / np.maximum(gpu_counts, 1.0)
	top_indices = np.argsort(-density)[: min(top_k_density, num_jobs)]
	top_hint = ", ".join(
		f"job {int(j)} (density={density[j]:.2f})" for j in top_indices
	)

	job_parts = []
	for j in range(num_jobs):
		job_parts.append(
			f"Job {j}, throughput: {int(throughputs[j])}, gpus: {int(gpu_counts[j])}"
		)

	return "; ".join(job_parts) + f". Top density jobs: {top_hint}."


def tag_prompt_and_transform_to_json_cs(jobs, num_gpus, selected, opt_throughput):
	"""Create a JSON-ready dict for cluster scheduling."""
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

	output_str = (
		f"Schedule: {list(map(int, selected))}, "
		f"Objective: {int(round(opt_throughput))}"
	)

	return {
		"num_jobs": str(num_jobs),
		"num_nodes": str(num_jobs),
		"num_gpus": str(int(num_gpus)),
		"instruction": instruction,
		"input": _format_job_descriptions(jobs),
		"output": output_str,
	}


class ClusterSchedulingEnv:
	"""Generate random cluster scheduling instances with Gurobi optimal labels."""

	def __init__(
		self,
		n_job_range,
		num_gpus,
		throughput_range=(0, 50),
		gpu_range=(1, 20),
		seed=None,
	):
		self.n_job_range = list(n_job_range)
		self.num_gpus = int(num_gpus)
		self.throughput_range = tuple(throughput_range)
		self.gpu_range = tuple(gpu_range)
		self.seed = seed

		if self.seed is not None:
			random.seed(self.seed)
			np.random.seed(self.seed)

	def uniform_random_choice(self, start, end):
		return random.randint(start, end)

	def generate_one_instance(self):
		num_jobs = self.uniform_random_choice(self.n_job_range[0], self.n_job_range[1])
		throughputs = np.random.randint(
			self.throughput_range[0],
			self.throughput_range[1] + 1,
			size=num_jobs,
		).astype(np.float64)
		gpu_counts = np.random.randint(
			self.gpu_range[0],
			self.gpu_range[1] + 1,
			size=num_jobs,
		).astype(np.float64)
		jobs = np.column_stack([throughputs, gpu_counts])
		return jobs, self.num_gpus

	def generate_instances(self, n_instance):
		return [self.generate_one_instance() for _ in range(n_instance)]

	def generate_instances_and_save(
		self,
		n_instance,
		file_name,
		save_pkl=False,
		rl_data=False,
		pkl_path="./instances.pkl",
	):
		valid_json_data = []
		valid_instances_pkl = []
		count_valid = 0

		pbar = tqdm(total=n_instance, desc="Generating CS instances")
		while count_valid < n_instance:
			jobs, num_gpus = self.generate_one_instance()
			try:
				opt_throughput, selected = solve_cs_gurobi(
					jobs[:, 0], jobs[:, 1], num_gpus
				)
				if opt_throughput is None or selected is None:
					continue

				cs_json = tag_prompt_and_transform_to_json_cs(
					jobs, num_gpus, selected, opt_throughput
				)
				if rl_data:
					cs_json["instance"] = [
						jobs[:, 0].tolist(),
						jobs[:, 1].tolist(),
						float(num_gpus),
					]

				valid_json_data.append(cs_json)
				valid_instances_pkl.append((
					jobs[:, 0].tolist(),
					jobs[:, 1].tolist(),
					float(num_gpus),
					selected,
					float(opt_throughput),
				))
				count_valid += 1
				pbar.update(1)
			except Exception:
				continue

		pbar.close()

		out_dir = os.path.split(file_name)[0]
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir)

		with open(file_name, 'w') as f:
			json.dump(valid_json_data, f, indent=4)
		print(f"Generated {len(valid_json_data)} valid CS instances -> {file_name}")

		if save_pkl:
			save_dataset(valid_instances_pkl, pkl_path)

		return valid_json_data


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate cluster scheduling datasets for LLMCO"
	)
	parser.add_argument('--n_instance', type=int, default=100,
		help='Number of valid instances to generate')
	parser.add_argument('--output', type=str, required=True,
		help='Output JSON file path')
	parser.add_argument('--rl_data', action='store_true',
		help='Include instance field for RL training')
	parser.add_argument('--save_pkl', action='store_true',
		help='Also save raw instances as pickle')
	parser.add_argument('--pkl_path', type=str, default='./instances.pkl',
		help='Pickle output path when --save_pkl is set')
	parser.add_argument('--n_job_range', type=int, nargs=2, default=[20, 20],
		help='Min and max number of jobs per instance')
	parser.add_argument('--num_gpus', type=int, default=100,
		help='Cluster GPU capacity N')
	parser.add_argument('--throughput_range', type=int, nargs=2, default=[0, 10],
		help='Min and max integer throughput per job (inclusive)')
	parser.add_argument('--gpu_range', type=int, nargs=2, default=[1, 20],
		help='Min and max GPUs requested per job')
	parser.add_argument('--seed', type=int, default=42,
		help='Random seed')
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	check_gurobi_license()
	env = ClusterSchedulingEnv(
		n_job_range=args.n_job_range,
		num_gpus=args.num_gpus,
		throughput_range=tuple(args.throughput_range),
		gpu_range=tuple(args.gpu_range),
		seed=args.seed,
	)
	env.generate_instances_and_save(
		n_instance=args.n_instance,
		file_name=args.output,
		save_pkl=args.save_pkl,
		rl_data=args.rl_data,
		pkl_path=args.pkl_path,
	)

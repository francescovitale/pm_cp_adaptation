from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
import pm4py
from pm4py.objects.log.obj import EventLog
from sklearn.impute import SimpleImputer

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.mixture import BayesianGaussianMixture

import sys
import os
import pandas as pd
import numpy as np

from itertools import combinations,permutations

import time

input_dir = "Input/OA/"
input_data_dir = input_dir + "Data/"
input_petrinets_dir = input_dir + "PetriNets/"

output_dir = "Output/OA/"
output_diagnoses_dir = output_dir + "Diagnoses/"
output_petrinets_dir = output_dir + "PetriNets/"
output_timing_dir = output_dir + "Timing/"

def read_petri_nets():

	petri_nets = {}

	for petri_net in os.listdir(input_petrinets_dir):
		petri_net_name = petri_net.split(".pnml")[0]
		petri_nets[petri_net_name] = {}
		petri_nets[petri_net_name]["network"], petri_nets[petri_net_name]["initial_marking"], petri_nets[petri_net_name]["final_marking"] = pnml_importer.apply(input_petrinets_dir + petri_net)
		
	return petri_nets

def read_event_log():

	event_log = xes_importer.apply(input_data_dir + "L_TR.xes")

	return event_log
	
def compute_diagnoses(petri_nets, event_log):
	import pandas as pd
	from pm4py.objects.log.obj import EventLog

	n_traces = len(event_log)
	event_log_activities = set(get_event_log_activities(event_log))

	final_rows = []

	for trace_idx, trace in enumerate(event_log):
		best_row = None
		best_fitness = -float('inf')

		for petri_net_name, petri_net in petri_nets.items():
			petri_net_activities = set(get_petri_net_activities(petri_net))
			all_activities = sorted(list(petri_net_activities.union(event_log_activities)))

			log = EventLog([trace])
			try:
				trace_diagnoses = generate_ab_diagnoses(log, petri_net, all_activities)
			except:
				continue

			fitness = trace_diagnoses["Fitness"]
			if fitness > best_fitness:
				best_fitness = fitness
				best_row = {**trace_diagnoses}

		final_rows.append(best_row)

	final_df = pd.DataFrame(final_rows)

	return final_df

def get_event_log_activities(event_log):
	
	activities = []
	for trace in event_log:
		for event in trace:
			if event["concept:name"] not in activities:
				activities.append(event["concept:name"])	
					
	activites = list(set(activities))

	return activities

def get_petri_net_activities(petri_net):
	activities = []
	transitions = list(petri_net["network"]._PetriNet__get_transitions())

	for transition in transitions:
		transition = transition._Transition__get_label()
		if transition != None:
			activities.append(transition)

	return activities

def generate_ab_diagnoses(log, petri_net, activities):

	ab_diagnoses = {}

	trace_activities = get_event_log_activities(log)
	last_log_activity = trace_activities[-1]

	for activity in activities:
		ab_diagnoses[activity] = 0;

	fitness, aligned_traces = compute_fitness(petri_net, log)
	temp = []
	for aligned_trace in aligned_traces:
		temp.append(list(aligned_trace.values())[0])
	aligned_traces = temp
	misaligned_activities = compute_misaligned_activities(log, aligned_traces)	
	for misaligned_activity in misaligned_activities:
		ab_diagnoses[misaligned_activity] = misaligned_activities[misaligned_activity]
		
	ab_diagnoses["Fitness"] = fitness

	return ab_diagnoses

def compute_fitness(petri_net, event_log):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
	log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	

	return log_fitness, aligned_traces
	
def compute_misaligned_activities(event_log, aligned_traces):
	
	misaligned_activities = {}
	events = {}
	
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
						events[log_behavior] = events[log_behavior]+1
				elif model_behavior != None and model_behavior != ">>":
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
						events[model_behavior] = events[model_behavior]+1
	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			misaligned_activities[popped_event[0]] = popped_event[1]

	return misaligned_activities	

def cluster_diagnoses(
	df,
	method="DBSCAN",
	params=None,
	drop_non_numeric=True,
	impute_strategy="constant",
	impute_fill=0,
	dpgmm_reg_covar=1e-3
):

	if drop_non_numeric:
		X = df.select_dtypes(include=[np.number]).copy()
	else:
		X = df.copy()

	if "Fitness" in X.columns:
		X_features = X.drop(columns=["Fitness"])
	else:
		X_features = X

	X_features = X_features.loc[:, X_features.nunique() > 1]

	if X_features.isnull().values.any():
		imputer = SimpleImputer(strategy=impute_strategy, fill_value=impute_fill)
		X_features = pd.DataFrame(
			imputer.fit_transform(X_features),
			columns=X_features.columns,
			index=X_features.index
		)

	default_params = {
		"DBSCAN": {"eps": 0.5, "min_samples": 3, "n_jobs": -1},
		"OPTICS": {"min_samples": 3, "xi": 0.05, "min_cluster_size": 5, "n_jobs": -1},
		"DPGMM": {
			"n_components": 20,
			"weight_concentration_prior": 1e-2,
			"max_iter": 1000,
			"random_state": 42,
			"reg_covar": dpgmm_reg_covar
		}
	}

	if params is None:
		params = default_params.get(method.upper(), {}).copy()
	else:
		defaults = default_params.get(method.upper(), {}).copy()
		defaults.update(params)
		params = defaults

	method = method.upper()
	labels = None

	if method == "DBSCAN":
		model = DBSCAN(**params)
		labels = model.fit_predict(X_features)

	elif method == "OPTICS":
		model = OPTICS(**params)
		labels = model.fit_predict(X_features)

	elif method == "DPGMM":
		unique_rows = X_features.drop_duplicates()
		n_samples = unique_rows.shape[0]

		max_components = max(2, int(n_samples / 2))
		params["n_components"] = min(params.get("n_components", 20), max_components)
		params["reg_covar"] = max(params.get("reg_covar", 1e-3), 1e-3)

		try:
			model = BayesianGaussianMixture(**params)
			labels = model.fit_predict(X_features)

		except ValueError as e:
			print(f"DPGMM failed with reg_covar={params['reg_covar']}: {e}")

			if "ill-defined empirical covariance" in str(e):
				params["reg_covar"] *= 10
				print(f"Retrying with reg_covar={params['reg_covar']}")
				try:
					model = BayesianGaussianMixture(**params)
					labels = model.fit_predict(X_features)
				except ValueError as e2:
					print(f"DPGMM retry failed: {e2}")
					labels = np.full(X_features.shape[0], -1)
			else:
				labels = np.full(X_features.shape[0], -1)

	else:
		raise ValueError(f"Unknown clustering method: {method}")

	df_out = df.copy()
	df_out["Cluster"] = labels

	return df_out

def split_event_log(event_log, clustered_diagnoses):

	event_logs = {}
	
	cluster_labels = list(set(clustered_diagnoses["Cluster"]))
	try:
		cluster_labels.remove(-1)
	except:
		pass
	
	for cluster_label in cluster_labels:
		traces_indices = list(clustered_diagnoses.loc[clustered_diagnoses["Cluster"] == cluster_label].index)
		event_logs[cluster_label] = []
		log = EventLog()
		for idx in traces_indices:
			log.append(event_log[idx])
		event_logs[cluster_label] = log
	
	return event_logs

def process_discovery(event_logs, pd_variant, petri_nets, n_iteration, use_baseline):

	if use_baseline == 0:
		for cluster_label in event_logs:
		
			petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)] = {}
		
			if pd_variant == "im":
				petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["network"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["initial_marking"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs[cluster_label], noise_threshold = 0.75)

			elif pd_variant == "ilp":
				petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["network"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["initial_marking"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["final_marking"] = pm4py.discover_petri_net_ilp(event_logs[cluster_label], alpha=1-0.75)
				
			elif pd_variant == "hm":
				petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["network"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["initial_marking"], petri_nets["PN_" + str(n_iteration) + "_" + str(cluster_label)]["final_marking"] = pm4py.discover_petri_net_heuristics(event_logs[cluster_label], dependency_threshold=0.75)
	elif use_baseline == 1:
		petri_nets = {}
		petri_nets["PN_" + str(n_iteration)] = {}
		
		if pd_variant == "im":
			petri_nets["PN_" + str(n_iteration)]["network"], petri_nets["PN_" + str(n_iteration)]["initial_marking"], petri_nets["PN_" + str(n_iteration)]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs, noise_threshold = 0.75)

		elif pd_variant == "ilp":
			petri_nets["PN_" + str(n_iteration)]["network"], petri_nets["PN_" + str(n_iteration)]["initial_marking"], petri_nets["PN_" + str(n_iteration)]["final_marking"] = pm4py.discover_petri_net_ilp(event_logs, alpha=1-0.75)
				
		elif pd_variant == "hm":
			petri_nets["PN_" + str(n_iteration)]["network"], petri_nets["PN_" + str(n_iteration)]["initial_marking"], petri_nets["PN_" + str(n_iteration)]["final_marking"] = pm4py.discover_petri_net_heuristics(event_logs, dependency_threshold=0.75)
		
	return petri_nets
	
def save_petri_nets(new_petri_nets):

	for petri_net in new_petri_nets:
		pnml_exporter.apply(new_petri_nets[petri_net]["network"], new_petri_nets[petri_net]["initial_marking"], output_petrinets_dir + petri_net + ".pnml", final_marking = new_petri_nets[petri_net]["final_marking"])

	return None
	
def save_diagnoses(pre_adaptation_diagnoses, post_adaptation_diagnoses, n_iteration):

	pre_adaptation_diagnoses = pre_adaptation_diagnoses.fillna(0)
	post_adaptation_diagnoses = post_adaptation_diagnoses.fillna(0)

	pre_adaptation_diagnoses.to_csv(output_diagnoses_dir + "PRE_" + str(n_iteration) + ".csv", index=False)
	post_adaptation_diagnoses.to_csv(output_diagnoses_dir + "POST_" + str(n_iteration) + ".csv", index=False)

	return None

def save_timing(elapsed_time):
    with open(output_timing_dir + "timing.txt", "a") as f:
        f.write(f"{elapsed_time:.4f}\n")
	
try:
	use_baseline = int(sys.argv[1])
	pd_variant = sys.argv[2]
	n_iteration = int(sys.argv[3])
	if use_baseline == 0:
		clustering_type = sys.argv[4]
		
except:
	print("Enter the right number of input arguments")

petri_nets = read_petri_nets()
event_log = read_event_log()
if use_baseline == 0:
	start = time.perf_counter()
	pre_adaptation_diagnoses = compute_diagnoses(petri_nets, event_log)
	clustered_diagnoses = cluster_diagnoses(pre_adaptation_diagnoses, clustering_type)
	event_logs = split_event_log(event_log, clustered_diagnoses)
	new_petri_nets = process_discovery(event_logs, pd_variant, petri_nets, n_iteration, use_baseline)
	end = time.perf_counter()

	post_adaptation_diagnoses = pd.DataFrame()
	for log in event_logs:
		diagnoses = compute_diagnoses(new_petri_nets, event_logs[log])
		post_adaptation_diagnoses = pd.concat([post_adaptation_diagnoses, diagnoses], ignore_index=True)
	save_petri_nets(new_petri_nets)
	save_diagnoses(pre_adaptation_diagnoses, post_adaptation_diagnoses, n_iteration)
	save_timing(end-start)
elif use_baseline == 1:
	start = time.perf_counter()
	new_petri_nets = process_discovery(event_log, pd_variant, petri_nets, n_iteration, use_baseline)
	end = time.perf_counter()
	save_timing(end-start)
	save_petri_nets(new_petri_nets)












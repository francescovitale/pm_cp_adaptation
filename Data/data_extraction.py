import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime
import sys
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py
import os
import random

input_dir = "Input/"

output_dir = "Output/"
output_baseline_dir = output_dir + "Baseline/"
output_method_dir = output_dir + "Method/"

def load_data():

	data = {}
	
	data["COVID_PE"] = None
	data["COVID_ARD"] = None
	data["COVID_SST"] = None
	data["COVID_AVP"] = None
	data["COVID_AB"] = None

	conditions = pd.read_csv(input_dir + "conditions.csv")
	procedures = pd.read_csv(input_dir + "procedures.csv")

	patients_with_covid = conditions[conditions.CODE == 840539006].PATIENT
	patients_with_pe = conditions[conditions.CODE == 87433001].PATIENT
	patients_with_ard = conditions[conditions.CODE == 67782005].PATIENT
	patients_with_sst = conditions[conditions.CODE == 43878008].PATIENT
	patients_with_avp = conditions[conditions.CODE == 195662009].PATIENT
	patients_with_ab = conditions[conditions.CODE == 10509002].PATIENT
	patients_with_mnb =  conditions[conditions.CODE == 254837009].PATIENT
	
	covid_patient_ids = list(set(patients_with_covid)) # COVID-19
	covid_pe_patient_ids = list(set(patients_with_covid) & set(patients_with_pe)) # Pulmonary emphysema
	covid_ard_patient_ids = list(set(patients_with_covid) & set(patients_with_ard)) # Acute respiratory distress syndrome
	covid_sst_patient_ids = list(set(patients_with_covid) & set(patients_with_sst)) # Streptococcal sore throat
	covid_avp_patient_ids = list(set(patients_with_covid) & set(patients_with_avp)) # Acute viral pharyngitis
	covid_ab_patient_ids = list(set(patients_with_covid) & set(patients_with_ab)) # Acute bronchitis
	covid_mnb_patient_ids = list(set(patients_with_covid) & set(patients_with_mnb)) # Malignant neoplasm of breast
	
	
	covid_pe_traces = []
	for i in range(0, len(covid_pe_patient_ids)):
		covid_pe_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_pe_patient_ids[i]]["DESCRIPTION"]))
	data["COVID_PE"] = covid_pe_traces
	covid_ard_traces = []
	for i in range(0, len(covid_ard_patient_ids)):
		covid_ard_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_ard_patient_ids[i]]["DESCRIPTION"]))
	data["COVID_ARD"] = covid_ard_traces
	covid_sst_traces = []
	for i in range(0, len(covid_sst_patient_ids)):
		covid_sst_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_sst_patient_ids[i]]["DESCRIPTION"]))
	data["COVID_SST"] = covid_sst_traces	
	covid_avp_traces = []
	for i in range(0, len(covid_avp_patient_ids)):
		covid_avp_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_avp_patient_ids[i]]["DESCRIPTION"]))
	data["COVID_AVP"] = covid_avp_traces
	covid_ab_traces = []
	for i in range(0, len(covid_ab_patient_ids)):
		covid_ab_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_ab_patient_ids[i]]["DESCRIPTION"]))	
	data["COVID_AB"] = covid_ab_traces
	covid_mnb_traces = []
	for i in range(0, len(covid_mnb_patient_ids)):
		covid_mnb_traces.append(list(procedures.loc[procedures['PATIENT'] == covid_mnb_patient_ids[i]]["DESCRIPTION"]))	
	data["COVID_MNB"] = covid_mnb_traces
	

	return data
	
def calculate_event_log_statistics(data):

	stats = {}

	for key, traces in data.items():
		if traces is None or len(traces) == 0:
			stats[key] = {
				"num_activities": 0,
				"num_traces": 0,
				"trace_length_mean": 0,
				"trace_length_std": 0
			}
			continue

		# Total number of activity instances
		num_activities = sum(len(trace) for trace in traces)

		# Number of traces
		num_traces = len(traces)

		# Trace lengths
		trace_lengths = [len(trace) for trace in traces]

		stats[key] = {
			"num_activities": num_activities,
			"num_traces": num_traces,
			"trace_length_mean": float(np.mean(trace_lengths)),
			"trace_length_std": float(np.std(trace_lengths))
		}

	return stats	

def split_event_logs_method(data, seed=42):
	
	training_event_logs = {}
	test_event_logs = {}

	# Datasets in order: PE (base), ARD, SST, AVP, AB
	datasets = ["COVID_PE", "COVID_ARD", "COVID_SST", "COVID_AVP", "COVID_AB"]
	special_test_dataset = "COVID_MNB"

	prev_test_traces = []

	# Normal iterations (PE → AB)
	for i, dataset in enumerate(datasets):
		traces = data.get(dataset, [])
		if traces is None or len(traces) == 0:
			training_event_logs[f"ITERATION_{i}"] = []
			test_event_logs[f"ITERATION_{i}"] = []
			continue

		# Shuffle and split 75/25
		idx = list(range(len(traces)))
		random.shuffle(idx)
		split_point = int(0.75 * len(traces))
		train_idx = idx[:split_point]
		test_idx = idx[split_point:]

		training_traces = [traces[j] for j in train_idx]
		test_traces = [traces[j] for j in test_idx]

		if i == 0:
			# Initial training set (L_tr = 75% of COVID_PE)
			training_event_logs[f"ITERATION_{i}"] = training_traces
			test_event_logs[f"ITERATION_{i}"] = test_traces
			prev_test_traces.extend(test_traces)
		else:
			# Online adaptation: 75% training, 25% testing + all previous test traces
			training_event_logs[f"ITERATION_{i}"] = training_traces
			combined_test = prev_test_traces + test_traces
			test_event_logs[f"ITERATION_{i}"] = combined_test
			prev_test_traces.extend(test_traces)
			
		print("method, iteration " + str(i))	
		print(len(training_event_logs[f"ITERATION_{i}"]))		

	# --- COVID_MNB: separate final test set ---
	mnb_traces = data.get(special_test_dataset, [])
	test_event_logs["COVID_MNB"] = mnb_traces

	return training_event_logs, test_event_logs
	
def split_event_logs_baseline(data, seed=42):
	import random
	random.seed(seed)

	training_event_logs = {}
	test_event_logs = {}

	# Datasets in order: PE (base), ARD, SST, AVP, AB
	datasets = ["COVID_PE", "COVID_ARD", "COVID_SST", "COVID_AVP", "COVID_AB"]
	special_test_dataset = "COVID_MNB"

	prev_train_traces = []
	prev_test_traces = []

	for i, dataset in enumerate(datasets):
		traces = data.get(dataset, [])
		if traces is None or len(traces) == 0:
			training_event_logs[f"ITERATION_{i}"] = []
			test_event_logs[f"ITERATION_{i}"] = []
			continue

		# Shuffle and split 75/25
		idx = list(range(len(traces)))
		random.shuffle(idx)
		split_point = int(0.75 * len(traces))
		train_idx = idx[:split_point]
		test_idx = idx[split_point:]

		training_traces = [traces[j] for j in train_idx]
		test_traces = [traces[j] for j in test_idx]

		if i == 0:
			# Iteration 0: baseline train/test
			training_event_logs[f"ITERATION_{i}"] = training_traces
			test_event_logs[f"ITERATION_{i}"] = test_traces
			prev_train_traces.extend(training_traces)
			prev_test_traces.extend(test_traces)
		else:
			# Iteration i: cumulative training/testing
			cumulative_train = prev_train_traces + training_traces
			cumulative_test = prev_test_traces + test_traces

			training_event_logs[f"ITERATION_{i}"] = cumulative_train
			test_event_logs[f"ITERATION_{i}"] = cumulative_test

			# Update memory
			prev_train_traces.extend(training_traces)
			prev_test_traces.extend(test_traces)
		print("baseline, iteration " + str(i))	
		print(len(training_event_logs[f"ITERATION_{i}"]))	

	# --- COVID_MNB: separate hold-out test set ---
	mnb_traces = data.get(special_test_dataset, [])
	test_event_logs["COVID_MNB"] = mnb_traces

	return training_event_logs, test_event_logs

def extract_event_logs(training_event_logs, test_event_logs):

	transformed_training_event_logs = {}
	transformed_test_event_logs = {}
	
	for iteration in training_event_logs:
		transformed_training_event_logs[iteration] = build_event_log(training_event_logs[iteration])
		transformed_training_event_logs[iteration] = pm4py.filter_variants_top_k(transformed_training_event_logs[iteration],k=20,activity_key='concept:name',timestamp_key='time:timestamp',case_id_key='case:concept:name')
		
	for iteration in test_event_logs:
		transformed_test_event_logs[iteration] = build_event_log(test_event_logs[iteration])
		transformed_test_event_logs[iteration] = pm4py.filter_variants_top_k(transformed_test_event_logs[iteration],k=20,activity_key='concept:name',timestamp_key='time:timestamp',case_id_key='case:concept:name')
		
	return transformed_training_event_logs, transformed_test_event_logs

def build_event_log(traces):
	
	event_log = []
	for idx,trace in enumerate(traces):
		caseid = idx
		for idx_e, event in enumerate(trace):
			event_timestamp = timestamp_builder(idx_e)
			state_transition = event
			event = [caseid, state_transition, event_timestamp]
			event_log.append(event)
	
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	return event_log

def timestamp_builder(number):
	
	ss = number
	mm, ss = divmod(ss, 60)
	hh, mm = divmod(mm, 60)
	ignore, hh = divmod(hh, 24)
	
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)		

def save_statistics(statistics):

	# Convert dict to DataFrame
	df = pd.DataFrame.from_dict(statistics, orient="index")
	
	# Reset index so log names become a column
	df.reset_index(inplace=True)
	df.rename(columns={"index": "Event Log"}, inplace=True)
	
	# Save to CSV
	df.to_csv(output_dir + "statistics.csv", index=False)

	return None

def save_event_logs(training_event_logs, test_event_logs, training_event_logs_baseline, test_event_logs_baseline):

	for iteration in training_event_logs:
		if not os.path.exists(output_method_dir + iteration):
			os.mkdir(output_method_dir + iteration)
		xes_exporter.apply(training_event_logs[iteration], output_method_dir + iteration + "/L_TR.xes")
		xes_exporter.apply(test_event_logs[iteration], output_method_dir + iteration + "/L_TST_N.xes")
		xes_exporter.apply(test_event_logs["COVID_MNB"], output_method_dir + iteration + "/L_TST_A.xes")
	
		if not os.path.exists(output_baseline_dir + iteration):
			os.mkdir(output_baseline_dir + iteration)
		xes_exporter.apply(training_event_logs_baseline[iteration], output_baseline_dir + iteration + "/L_TR.xes")
		xes_exporter.apply(test_event_logs_baseline[iteration], output_baseline_dir + iteration + "/L_TST_N.xes")
		xes_exporter.apply(test_event_logs_baseline["COVID_MNB"], output_baseline_dir + iteration + "/L_TST_A.xes")
		
			
	return None

data = load_data()
statistics = calculate_event_log_statistics(data)
training_event_logs, test_event_logs = split_event_logs_method(data)
'''
for iteration in training_event_logs:
	print(iteration)
	print("TRAINING: " + str(len(training_event_logs[iteration])))
	print("TEST: " + str(len(test_event_logs[iteration])))
	print()
'''	
training_event_logs_baseline, test_event_logs_baseline = split_event_logs_baseline(data)
'''
for iteration in training_event_logs_baseline:
	print(iteration)
	print("TRAINING: " + str(len(training_event_logs_baseline[iteration])))
	print("TEST: " + str(len(test_event_logs_baseline[iteration])))
	print()
'''	
training_event_logs, test_event_logs = extract_event_logs(training_event_logs, test_event_logs)
training_event_logs_baseline, test_event_logs_baseline = extract_event_logs(training_event_logs_baseline, test_event_logs_baseline)
save_statistics(statistics)
save_event_logs(training_event_logs, test_event_logs, training_event_logs_baseline, test_event_logs_baseline)


from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import pm4py
import sys
import os


from itertools import combinations,permutations

import time

input_dir = "Input/T/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/T/"
output_petrinet_dir = output_dir + "PetriNet/"

variant = ""



def read_data():

	event_log = xes_importer.apply(input_data_dir + "L_TR.xes")

	return event_log
	
	
def process_discovery(event_log, variant):

	petri_net = {}
	if variant == "im":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.75)

	elif variant == "ilp":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.75)

	elif variant == "hm":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_heuristics(event_log, dependency_threshold=0.75)

	return petri_net


def write_petri_net(petri_net):
	pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_petrinet_dir + "PN.pnml", final_marking = petri_net["final_marking"])
	
	return None
	
try:
	variant = sys.argv[1]
except:
	print("Enter the right number of input arguments.")
	sys.exit()
	
event_log = read_data()
petri_net = process_discovery(event_log, variant)
write_petri_net(petri_net)






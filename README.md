# Requirements to run the method

## Packages
This project has been executed on a Windows 11cmd machine with Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.11.2
- scikit-learn 1.3.0

Please note that the list above is not comprehensive and there could be other requirements for running the project.

## Data
The data used in this implementation comes from the COVID-19 Synthea dataset, available at https://synthea.mitre.org/downloads. In our project, part of the data has been included in the Data/Input folder, leaving only the essential information to pre-process and organize the data for executing the experiments.

# Execution instructions and project description

To run the experiment, it is sufficient to execute the experimentation_method.bat script. This script includes experimental parameters to set:

- The process discovery variants to use (pd_variant)
- The clustering techniques to use (clustering_technique)
- The number of repetitions of the experiment (n_reps)
- The iterations at which online adaptation is performed (iterations)

The script organizes the environment and first executes the training.bat script, which is the first phase of the method to initialize the knowledge base. Next, the online_adaptation.bat script is executed for each iteration to adaptively identify new clinical pathways. The results are collected under the "Results" folder.

Please note that the project also includes the experimentation_baseline.bat script, which executes the experiment with the baseline approach that does not employ conformance checking and clustering to adaptively find new process models.

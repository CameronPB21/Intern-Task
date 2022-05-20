# Task
Internship Task for modeling EEG data

Given an EEG dataset containing seizure and non-seizure data from the University of Bonn: 
-Build a model to classify randomized epochs of the signal.
-Summarize results in an ROC curve.

This implementation:
First, recognizes the data from the University of Bonn as having the following structure:
-5 Folders (O, Z, F, N, S,) one per participant (normal, normal, interictal, interictal, ictal) respectively
-100 txt data files per folder, each representing an electrode channel's 4,097 measurements

Second, extracts the data from the folders and organizes it as one dataset of eeg measurements with 
classifications according to the participant's current status (normal, interictal, ictal)

Third, preforms PCA on the dataset to determine the n most informative dimensions, then reduces the dataset to 
those first n dimensions.

Fourth, builds three different models based on the extracted dataset, one svc, one decision tree, and one random forest. Then trains
the models using 80% of the dataset, and tests the model on the remaining 20% of the dataset.

Last, the results of testing each model are plotted on a Receiver Operating Characteristic (ROC) curve, and the 
Area Under the ROC Curve (AUC) is calculated and shown on the graphs.

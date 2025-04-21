# Quantum Machine Learning Benchmark

## Reproducibility Report: Quantum Machine Learning Methods in Fundus Analysis – A Benchmark Study.
Authors: Gajan Mohan Raj, Zayan Hasan

Supervisors: Dr. Mohammad Eslami, Dr. Saber Kazeminisab, Dr. Tobias Elze

## Repository Information
This repository aims to evaluate the performance of various quantum machine learning algorithms compared to classical machine learning algorithms. for the detection of glaucoma and diabetic retinopathy. This benchmark is crucial for the development of novel quantum machine learning techniques for medical diagnosis, specifically for fundus diseases.

There are six notebooks with results for six different fundus datasets: APTOS, G1020, GlaucomaFundus, IDRID, MESSIDOR, and PAPILA. 

Please contact gajanmohanraj@gmail.com or mohammad_eslami@meei.harvard.edu for any questions or concerns.

## TODO: ADD A GRAPHIC SHOWING COMPARISONS BETWEEN DATASETS OR OTHER RELEVANT INFORMATION FOR THIS FIELD


ToDO:
-Move notebooks to notebook directory
-in the data directory, make a readme and describe the source of data and if necessary mention save them here
-fill the scripts:
-- save all model definitions including retfound in the models.py
-- save all the trainers, testers and evaluators definitions in trainers_testers.py
-- put any other function and class definitions in utils.py

-for each dataset make a script e.g. main_end2end_aptos.py and include training and evaluation and saving the pickles
-fill free to have a seperate script for plotting

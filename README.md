# Code for "Reproducibility Report: Quantum Machine Learning Methods in Fundus Analysis – A Benchmark Study"

This project benchmarks quantum and classical machine learning algorithms for detecting glaucoma and diabetic retinopathy using six public fundus image datasets. It includes:

    Quantum models: Hybrid QNN-ResNet18, QNN-Parallel, Quantum SVM, Quantum KNN

    Classical counterparts: ResNet18, CNN, SVM, KNN

    Datasets: APTOS, IDRID, MESSIDOR (DR); G1020, GlaucomaFundus, PAPILA (Glaucoma)

This is the first open-source benchmark in ophthalmic AI comparing quantum and classical approaches, with fully reproducible code for medical imaging tasks.

This repository contains the code and resources for the project described above at the Harvard Opthalmology-AI Lab (supervised by Dr. Mohammad Eslami).

Please contact gajanmohanraj@gmail.com or mohammad_eslami@meei.harvard.edu for any questions or concerns.

### QML Model Overview
<div align="center">
  <img src="https://github.com/user-attachments/assets/580ccd62-5f33-4804-ad75-b3f05582b201" alt="QMLModelFigures (1)">
</div>

### Repository Information

|    Directory    |      Description     |
| ------------- | ------------- |
| Data | Contains Google Drive Links to the Six Datasets that were downloaded to be used in this study as well as information about the dataset. |
| Notebooks | Contains the Google Colab Notebooks that we ran to obtain our results for this study.  |

## Notice:
1-Download the listed in Data/ReadMe.md  </br>
  Look at the *tree_output.txt* to organize your data correctly. 

2- The tested libraries are python 3.9.22, torch 2.7, PennyLane 0.38, opencv 4.11, timm 1.0.15. </br> Full list available in *piplist.txt*

3-Download the retfound pretrained model (RETFound_cfp_weights.pth) and place it correctly (follow the tree_output.txt)



## Citation:
If you found this repository and benchamrks usefull, please cite the following paper:

"Reproducibility Report: Quantum Machine Learning Methods in Fundus Analysis – A Benchmark Study", Nature Eye, 2025

## References:
RETFound



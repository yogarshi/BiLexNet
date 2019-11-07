# Requirements (version used in our experiments)

* Python (2.7)
* Pytorch (0.4.1)
* Numpy (1.15.4)
* Scikit-learn (0.20.0)
* cuda (8.0.61)
* cudnn (7.5.0)



# Code 

The main code that runs our model is `train_student-teacher-withmonoen.py` while the model itself is implemented in `paths_student_teacher_pooling.py`. However, it is advisable to run the code through one of the two scripts described below

# Scripts

- `train_integrated_xlingual_st.sh`: Main script that calls the code. Takes in various parameters like temperature, distillation alpha, seed, and language
- `meta_runner.sh`: A meta-script that launches all experiments on the CLIP Cluster


# Data

The data directory contains two folders, one for en-hi, and en-zh. Each folder contains the following files :

1) train-clean.tsv : The monolingual training data, with dictionary based translations
2) val-entrans.tsv : The validation data
3) en-trans-sorted.tsv : The gold test data
4) wiki.*.align.vec : The word embeddings for the two languages \[Not included in this repo]
5) Path files \[Not included in this repo]: 
  - all.xlingual.count : Cross-lingual paths for the training/dev/test data extracted from parallel data
  - all.mono.count : Monolingual paths for the training data extracted from monolingual corpora

# Reference

[Paper](https://www.aclweb.org/anthology/D19-1532/)

[Citation](https://www.aclweb.org/anthology/D19-1532.bib)

Write to `yogarshi@amazon.com` for questions.


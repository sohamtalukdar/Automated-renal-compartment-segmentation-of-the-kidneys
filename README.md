# Automated renal compartment segmentation of the kidneys
## Thesis project to automatically segment the kidney tissue to separate the medulla and cortex based on the T1-weighted MRI scans in human subjects

*NB: This project is a part of ongoing Application of Functional Renal MRI to improve the assessment of chronic kidney disease (AFiRM) study. The dataset used in this project contains MRI images of kidneys taken from live voluntary subjects.*

[The report for this project can be found here](https://github.com/sohamtalukdar/Segmentation-of-Kidneys-in-MRI/blob/main/Report.pdf)



## Running the Code

### Prerequesites

Make sure you have Python > 3.9 and TF > 2.8 installed
Navigate to the root of this repository
You can Download the Dataset from [here]()
When running the code you can directly use the 

### Processing datasets

First you need to collect and process the datasets, so run
datasets/collect_datasets.py
datasets/preprocess_hatexplain.py
You may need a HuggingFace login to download the datasets using their API
Training models
To train models, the models/hyperparameter_sweep.py file can be called. This will iterate over the different model configs set out in that file.

Otherwise, you can call each of the base files (e.g bert_with_attention_entropy.py) and adjust parameter settings inside that to change your model configs.

The other files contain helper functions and structures to minimise the amount of code that clutters up the main files, in an effort to make the logic more readable.

### Running Tests 

# Automated renal compartment segmentation of the kidneys
## Thesis project to automatically segment the kidney tissue to separate the medulla and cortex based on the T1-weighted MRI scans in human subjects

*NB: This project is a part of ongoing Application of Functional Renal MRI to improve the assessment of chronic kidney disease (AFiRM) study. The dataset used in this project contains MRI images of kidneys taken from live voluntary subjects.*

[The report for this project can be found here](https://github.com/sohamtalukdar/Segmentation-of-Kidneys-in-MRI/blob/main/Report.pdf)



## Running the Code

### Prerequesites

* Make sure you have Python > 3.9 and TF > 2.8 installed
* Navigate to the root of this repository
* You can Download the Dataset from [here](https://github.com/sohamtalukdar/Automated-renal-compartment-segmentation-of-the-kidneys/tree/main/Dataset)
* Code for dataset analsyis and mask generation you can find [here](https://github.com/sohamtalukdar/Automated-renal-compartment-segmentation-of-the-kidneys/blob/main/dataset_image.py)
* Code for model execution you can find [here](https://github.com/sohamtalukdar/Automated-renal-compartment-segmentation-of-the-kidneys/blob/main/unet.py)
* [FMRIB Automated Segmentation Tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST) was used to evaluate the dataset in 3D and for generating automated ground truth. 


# Segmentation-of-Kidneys-in-MRI

Magnetic resonance imaging (MRI) has become increasingly important in the clinical work-up of renal diseases such as chronic kidney disease. This project shows the use of a UNet to automatically segment the kidney tissue to separate the medulla and cortex based on the T1-weighted MRI scans in human subjects. The project aims to automate the segmentation process for T1-weighted (T1w) scans. First masks are generated of the cortex and medulla from time-consuming (~5 minutes) MPRAGE scans of the kidneys using a K-means clustering technique. These masks are then used to train a UNet to segment the cortex and medullary of T1w images collected in a single breath-hold. UNet CNN architectures was assessed. The mean IOU score compared to the manual masks for renal cortex and medulla was 0.9974872 and 0.84625 for UNet. The availability of such an automated method will reduce the time required and the difficulty associated with manual segmentation of the cortex and medulla and will have a direct impact on medical trials, such as the ongoing Application of Functional Renal MRI to improve the assessment of chronic kidney disease (AFiRM) study.


<figure>
<img src="https://user-images.githubusercontent.com/103217802/207931588-1f7f2f96-2892-4461-8d69-1bdb1e1d27ef.png"  style="width: 500px">
<b>Fig. 1 Schematic of the kidney showing the different compartments of the cortex and medulla</b>
</figure>

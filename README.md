----Overview----

This repository contains scripts to build and evaluate post-fire debris-flow prediction models using two-feature combinations under different algorithms, rainfall intensities, and weighting schemes. Below are instructions for running each script and the required input files.

----Data----

Input files:

two_feature_rained.xlsx — used by most scripts.
three_feature_rained.xlsx — used for SWA features + KF and SWB features + KF.
data.xlsx — dataset without rainfall multiplied; the second sheet includes watershed longitude and latitude.

two_feature_rained.xlsx and three_feature_rained.xlsx share the same structure:
Sheet 1 – available feature combinations
Sheets 2–7 – feature-multiplied datasets for rainfall windows:
i2, i5, i10, i15, i30, i60

----Instructions and File Information----

models_full.py — builds all 3,840 models.
Add the path to two_feature_rained.xlsx.
Runs 80 models at a time.
Specify the rainfall intensity (i2, i5, i10, i15, i30, i60), weighting (none, balanced, square root) and, algorithm (LR, LDA, RF, XGB) you want to execute. Set for i15, square root, LR.

models_two.py — builds SWA and SWB feature models with four algorithms, i15 and, square root weighting.
Add the path to two_feature_rained.xlsx.
Other weighting options can be used with adjustments.
Otherrain intenisities can be used with adjustments.

SWA_SWB_MODELS.py — outputs model coefficients for square root weight logistic regression using i15 rainfall.
Add the path to two_feature_rained.xlsx.

models_plus_kf.py — builds SWA and SWB feature sets plus the KF soil factor using four algorithms, i15 rainfall, and square root weighting.
Add the path to three_feature_rained.xlsx.
Other weighting options can be used with adjustments.
Otherrain intenisities can be used with adjustments.

----Notes----

The SWA and SWB models use logistic regression (LR), i15 rainfall intensity, and square root weighting by default.
The scripts can be modified to use other algorithms, weighting schemes, or rainfall intensities.
All outputs include multiple accuracy metrics.
 


<<<<<<< Updated upstream
# CVVB
Folder: VAFC vs Hybrid VAFC, and posterior predictive densities
Folder: CVVB - Simulation Study
Folder: CVVB - Real data

=======
---
title: "CVVB"
output: pdf_document
---


## Remarks on the running time and Initialization

It is well-known that the running time of fixed-form VB depends on an initial value as well as VB setting (stopping rule, learning rate, etc.). A good initial value can speed up the process greatly whereas a bad one can lead to the failure. Choosing a good initial value is a challenging problem, especially for complex models like LBA models.
In my Matlab code, I propose an automatic way to set an initial value for VB. My proposed method is to run MCMC for a number of iterations (100 iterations work well in all case study) and take the last iteration or the average as an initial value for the VB. This will not take too much time as we just need to run MCMC for a small number of iterations but the gain is huge. Most of the time the initial value from this method is very good.
>>>>>>> Stashed changes

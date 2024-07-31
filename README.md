# hMeNet
MeNets and hierarchical likelihood

# Update 231022
1. Launch homoscedastic HGLM version 1.0.0
 - Mean pretrain : batch sampling + NJLL loss 
 - Variance pretrain : Method of Moments Estimators (MMEs)
 - M-step : batch sampling + NJLL loss
 - V-step : full batch + NHLL loss 

## Main code structure
1. *HGLM_homo_1.0.0_within.ipynb* : Main Jupyter notebook file
2. *hglm_homo.py* : main code
3. *h_likelihood.py* : NHLL loss function for gradient descent 
4. *likelihood.py* : full NLL, NJLL, NHLL function for evaluation.
5. *util.py* : Utility function

## Comments
 - NHLL loss ignoring c(theta;y) term is equivalent to NJLL loss. 
 - We may consider a subject-sampling technique not to ignore the c(theta;y) term in mean training, however, subject sampling is currently worse than batch-sampling and slowly converges. So homoscedastic HGLM ver 1.0.0 does not emply the batch sampling technique. 


<!-- # Update 231014
1. Split MeNet
 - MeNets_1/2/3/4/5 + loocv_menets_rev2 + likelihood

2. Homoscedastic HGLM (validation set)
 - LOOCV_homo_rev + loocv_homo + likelihood + h_likelihod_rev

3. Homoscedastic HGLM (without validation set)
 - LOOCV_homo_rev2 + loocv_homo_rev + likelihood + h_likelihod_rev

# Update (231012)
1. consider p = 500+3 as a hyperparameter
2. Remove 'MeNets' and use 'MeNets_rev' (revised)
 - We remove a validation dataset in MeNets, because in the original codes, the authors does not emply a validation dataset. 
 - It seems that they check the convergence of EM algorithm by computing train (negative) joint log-likelihood. 
So instead of computing validation njll, we compute train njll. 
 - Updated codes are denoted as 'MeNets_rev'. 
 - Previous 'MeNets' will not be used anymore. 

'loocv / within + MeNEts_rev' is the most recent version. 
deepHGLM will be updated soon.  -->
import torch
import numpy as np

def multivariate_njll_correlated_i(y_i, f_hat_i, Gamma_i, v_i, log_phi, log_lamb, arctan_rho) : 
    '''
    Compute 2 x negative joint log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x 2]   torch tensor        response
    f_hat_i         [n_i x 2]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x 2]     torch tensor        estimated random slopes for i-th subject
    log_phi         [n_i x 2]   torch tensor        estimated variance of e
    log_lamb        [p x 2]     torch tensor        estimated variance matrix of random slope v
    arctan_rho      []          torch tensor        correlation between random slopes
    '''

    rho = torch.tanh(arctan_rho)
    sqrt_lamb = torch.exp(0.5 * log_lamb)
    dist_i = v_i / sqrt_lamb

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / torch.exp(log_phi)) 
    term_2_1 = torch.sum(torch.pow(v_i,2)) 
    term_2_2 = (rho * (torch.pow(dist_i[0,0],2) + torch.pow(dist_i[0,1],2)) - 2 * dist_i[0,0] * dist_i[0,1]) * rho / (1-torch.pow(rho, 2)) 
    term_3 = torch.sum(log_phi + np.log(2 * np.pi)) 
    term_4_1 = torch.sum(log_lamb + np.log(2 * np.pi)) 
    term_4_2 = torch.log(1-torch.pow(rho,2)) 

    return term_1 + term_2_1 + term_2_2 + term_3 + term_4_1 + term_4_2

def multivariate_nhll_correlated_i(y_i, f_hat_i, Gamma_i, v_i, log_phi, log_lamb, arctan_rho) : 
    '''
    Compute 2 x negative hierarchical log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x 2]   torch tensor        response
    f_hat_i         [n_i x 2]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x 2]     torch tensor        estimated random slopes for i-th subject
    log_phi         [n_i x 2]   torch tensor        estimated variance of e
    log_lamb        [p x 2]     torch tensor        estimated variance matrix of random slope v
    arctan_rho      []          torch tensor        correlation between random slopes
    '''
    
    _, p = Gamma_i.shape
    rho = torch.tanh(arctan_rho)
    sqrt_lamb = torch.exp(0.5 * log_lamb)
    dist_i = v_i / sqrt_lamb

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / torch.exp(log_phi)) 
    term_2_1 = torch.sum(torch.pow(dist_i,2)) 
    term_2_2 = (rho * (torch.pow(dist_i[0,0],2) + torch.pow(dist_i[0,1],2)) - 2 * dist_i[0,0] * dist_i[0,1]) * rho / (1-torch.pow(rho, 2)) 
    term_3 = torch.sum(log_phi + np.log(2 * np.pi)) 
    term_4_1 = torch.sum(log_lamb + np.log(2 * np.pi)) 
    term_4_2 = torch.log(1-torch.pow(rho,2)) 

    big_inv_Sigma = torch.zeros(2*p, 2*p, device=log_lamb.device)
    big_inv_Sigma[:p,:p] = torch.diag(torch.exp(-log_lamb[:,0]))
    big_inv_Sigma[p:,p:] = torch.diag(torch.exp(-log_lamb[:,1]))
    big_inv_Sigma[0,0]  /= (1-torch.pow(rho,2))
    big_inv_Sigma[p,p]  /= (1-torch.pow(rho,2))
    big_inv_Sigma[p,0]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))
    big_inv_Sigma[0,p]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))

    big_inv_Sigma[:p, :p] += Gamma_i.T @ torch.diag(torch.exp(-log_phi[:,0])) @ Gamma_i
    big_inv_Sigma[p:, p:] += Gamma_i.T @ torch.diag(torch.exp(-log_phi[:,1])) @ Gamma_i
    term_5 = torch.linalg.slogdet(big_inv_Sigma)[1] 

    # print(f'Terms : {term_1.item()}, {term_2_1.item()}, {term_2_2.item()}, {term_3.item()}, {term_4_1.item()}, {term_4_2.item()}, {term_5.item()}')
    # print(term_5)
        

    return term_1 + term_2_1 + term_2_2 + term_3 + term_4_1 + term_4_2 + term_5


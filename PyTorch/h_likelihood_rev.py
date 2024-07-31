import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def nhll_homo_diag_subject(N, m, y, y_fixed, Gamma_i, v_i, log_phi, log_lamb, update='V', verbose=False) : 
    '''
    Compute -2 * log h-likelihood (called the nhll) / N for random slope model with
    homoscedastic random noises / diagonal random slope covariance / without correlation between random intercepts. 
    To obtain an unbiased gradient for the full nhll, 
    we divide the full nhll by N and hence enable the batch-wise training for mean. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, phi_k)
    v_ik  ind.   ~ N(0, Sigma_k),    where Sigma_k = diag(lamb_k1, ..., lamb_kp)
    
    NAME            SIZE        TYPE                INFO                    
    y               [n_i x K]   torch tensor        Responses       
    y_fixed         [n_i x K]   torch tensor        y_hat = fixed + random
    v_i             [p x K]     torch parameter     i-th random slope
    Gamma_i         [n_i x p]   torch tensor        Design matrix for i-th subject
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    verbose                     boolean             print the terms

    '''
    n_i, K = y.shape

    if update == 'pretrain' : 
        term_1 = torch.sum(torch.pow(y - y_fixed - Gamma_i @ v_i, 2), dim=0) / torch.exp(log_phi) * m / N
        term_3 = torch.sum(torch.pow(v_i, 2) / torch.exp(log_lamb), dim=0) * m / N
        
        if verbose : 
            print(f'Terms : {term_1}, {term_3}')

        nhll = torch.sum(term_1 + term_3)

    elif update == 'M' : 
        term_1 = torch.sum(torch.pow(y - y_fixed - Gamma_i @ v_i, 2), dim=0) / torch.exp(log_phi) * m / N
        term_3 = torch.sum(torch.pow(v_i, 2) / torch.exp(log_lamb), dim=0) * m / N
        term_5 = torch.linalg.slogdet(
                    torch.cat([(
                        torch.diag(torch.exp(-log_lamb[:,k])) + 
                        Gamma_i.T @ Gamma_i / torch.exp(log_phi[k])
                    ).unsqueeze(0) for k in range(K)], dim=0)
                )[1] * m / N
        
        if verbose : 
            print(f'Terms : {term_1}, {term_3}, {term_5}')

        nhll = torch.sum(term_1 + term_3 + term_5)

    elif update == 'V' : 
        term_1 = torch.sum(torch.pow(y - y_fixed - Gamma_i @ v_i, 2), dim=0) / torch.exp(log_phi) * m / N
        term_2 = n_i * (log_phi + np.log(2 * np.pi)) * m / N
        term_3 = torch.sum(torch.pow(v_i, 2) / torch.exp(log_lamb), dim=0) * m / N
        term_4 = torch.sum(log_lamb + np.log(2 * np.pi), dim=0) * m / N
        term_5 = torch.linalg.slogdet(
                    torch.cat([(
                        torch.diag(torch.exp(-log_lamb[:,k])) + 
                        Gamma_i.T @ Gamma_i / torch.exp(log_phi[k])
                    ).unsqueeze(0) for k in range(K)], dim=0)
                )[1] * m / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)
    
    else : 
        nhll=1e8


    return nhll



def nhll_homo_diag(N, y, y_fixed, y_random, v_list, log_phi, log_lamb, Gamma_list=None, update='M', weighted = False, n_list=None, batch_n_list=None, verbose=False) : 
    '''
    Compute -2 * log h-likelihood (called the nhll) / N for random slope model with
    homoscedastic random noises / diagonal random slope covariance / without correlation between random intercepts. 
    To obtain an unbiased gradient for the full nhll, 
    we divide the full nhll by N and hence enable the batch-wise training for mean. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, phi_k)
    v_ik  ind.   ~ N(0, Sigma_k),    where Sigma_k = diag(lamb_k1, ..., lamb_kp)
    

    M-step (pretrain / train)
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    update                      string              M
    weighted                    boolean             used weighted mean for random effects
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of selected peoples in the batch
    verbose                     boolean             print the terms

    
    V-step (train)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]     torch tensor        Responses       
    y_fixed         [N x K]     torch tensor        y_hat = fixed + random
    y_random        [N x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Design matrix for i-th subject
    update                      string              V
    verbose                     boolean             print the terms
    '''

    B, K = y.shape
    m = len(v_list)

    if update == 'M' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0) / torch.exp(log_phi)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) / torch.exp(log_lamb) * (batch_n_list[i] / n_list[i]) for i in range(m)]), dim=0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i,2) / torch.exp(log_lamb) for v_i in v_list]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
        nhll = torch.sum(term_1 + term_2) 

    elif update == 'M-wrong' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0) / torch.exp(log_phi)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) / torch.exp(log_lamb) * (batch_n_list[i] / B) for i in range(m)]), dim=0) * m / N
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i,2) / torch.exp(log_lamb) for v_i in v_list]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
        nhll = torch.sum(term_1 + term_2) 
    
    elif update == 'V' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0) / torch.exp(log_phi)
        term_2 = torch.sum(sum([torch.pow(v_i,2) / torch.exp(log_lamb) for v_i in v_list]), dim=0) / N
        term_3 = log_phi + np.log(2 * np.pi)
        term_4 = torch.sum(log_lamb + np.log(2 * np.pi), dim=0) * m / N
        term_5 = sum([
            torch.linalg.slogdet(
                torch.cat([(
                    torch.diag(torch.exp(-log_lamb[:,k])) + 
                    Gamma_i.T @ Gamma_i / torch.exp(log_phi[k])
                ).unsqueeze(0) for k in range(K)], dim=0)
            )[1] for Gamma_i in Gamma_list
        ]) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')
        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

    else : 
        nhll = 1e8
    return nhll

def nhll_hetero_diag(N, y, y_fixed, y_random, v_list, log_phi, log_lamb, Gamma_list=None, update='M', weighted = False, batch_n_list=None, verbose=False) : 
    '''
    Compute -2 * log h-likelihood (called the nhll) / N for random slope model with
    heteroscedastic random noises / diagonal random slope covariance / without correlation between random intercepts. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, phi_ijk^)
    v_ik  ind.   ~ N(0, Sigma_k),    where Sigma_k = diag(lamb_k1, ..., lamb_kp)

    M-step (pretrain / train)
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [B x K]     torch parameter     Variance of e (homoscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    update                      string              M
    weighted                    boolean             used weighted mean for random effects
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of selected peoples in the batch
    verbose                     boolean             print the terms
    '''

    B, K = y.shape
    m = len(v_list)

    if update == 'M' :
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0) 
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) / torch.exp(log_lamb) * (batch_n_list[i] / B) for i in range(m)]), dim=0) * m / N
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i,2) / torch.exp(log_lamb) for v_i in v_list]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')

        nhll = torch.sum(term_1)

    if update == 'V' : 
        nhll = None

    else : 
        nhll = 1e8

    return nhll

    






class Sigma_module(nn.Module) :
    def __init__(self, p=503, K=2, device = torch.device('cpu')) :
        super(Sigma_module, self).__init__()
        '''
        This module aims to parametrize the arbitary covariance matrix Sigma_k with size [p x p] for k=1,...,K. 
        That is, this module includes the information about Sigma of size [p x K x K], where
        Sigma[k] is symmetric and positive definite for each k. 

        We employ the Cholesky decomposition. 
        Sigma[k] = L[k] @ L[k].T
        '''

        self.p = p
        self.K = K
        self.device = device
        
        # To enable the gradient-based update, we have to initialize each compoent as non-zero, except the log-diagonal term. 
        L_wo_diag = torch.zeros(K, p, p, device = device)
        for l in range(p) : 
            L_wo_diag[:, (l+1):, l] = torch.randn(K, p-l-1, device=device) / np.sqrt(p)

        self.L_wo_diag = nn.Parameter(L_wo_diag)
        self.L_log_diag = nn.Parameter(torch.zeros(K, p, device=device))




def nhll_homo_arbitrary(N, y, y_fixed, y_random, v_list, log_phi=None, Sigma=None, Gamma_list=None, update='M') : 
    '''
    compute -2 * log h-likelihood / N for random slope model with
    homoscedastic random noises / arbitrary random slope covariance / without correlation between random intercepts

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, sigma_k^2)
    v_ik  ind.   ~ N(0, Sigma_k)

    M-step (pretrain / train)
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi    [K]         torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    update                      string              M

    V-step (train)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]     torch tensor        Responses               
    y_fixed         [N x K]     torch tensor        y_hat = fixed + random
    y_random        [N x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi    [K]         torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    update                      string              M
    Gamma_list      [m]         tuple
     - Gamma_i      [n_i x p]   torch tensor        Constructed design matrix
    update                      string              V
    '''
    m = len(v_list)
    p, K = v_list[0].shape

    L = Sigma.L_wo_diag + torch.cat([torch.diag(torch.exp(Sigma.L_log_diag[k])).unsqueeze(0) for k in range(K)])
    
    if update == 'M' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0) / torch.exp(log_phi)
        term_2 = sum([torch.sum(torch.pow(torch.linalg.solve_triangular(L, v_i.T.unsqueeze(2), upper=False).squeeze(2), 2), dim=1) for v_i in v_list]) / N
        
        nhll = torch.sum(term_1 + term_2)
        
    
    elif update == 'V' : 
        L_inv = torch.linalg.solve_triangular(L, torch.eye(p, device=L.device), upper=False)

        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0).detach() / torch.exp(log_phi)
        term_2 = sum([torch.sum(torch.pow(torch.linalg.solve_triangular(L, v_i.T.unsqueeze(2), upper=False).squeeze(2), 2), dim=1) for v_i in v_list]) / N
        term_3 = log_phi + np.log(2 * np.pi)
        term_4 = torch.sum(2 * Sigma.L_log_diag + np.log(2 * np.pi), dim=1) * m / N
        term_5 = sum([
            torch.linalg.slogdet(
                torch.bmm(L_inv.transpose(1,2), L_inv) + 
                (Gamma_i.T @ Gamma_i).detach().repeat(K,1,1) / torch.exp(log_phi).view(-1,1,1)
            )[1] for Gamma_i in Gamma_list
        ]) / N

        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

    else : 
        nhll = 1e8
        
    return nhll


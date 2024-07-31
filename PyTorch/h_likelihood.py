import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def nhll_homo_diag(N, y, y_fixed, y_random, v_list, log_phi, log_lamb, Gamma_list=None, log_phi_list=None, 
                   weighted = False, n_list=None, batch_n_list=None, verbose=False, update='M') : 
    '''
    Compute -2 * log h-likelihood / N (called the NHLL loss) for random slope model with
    homoscedastic random noises / diagonal random slope covariance. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    e_ijk i.i.d. ~ N(0, phi_k)
    v_ik  ind.   ~ N(0, Sigma_k),    where Sigma_k = diag(lamb_k1, ..., lamb_kp)
    for i=1,...,m, j=1,...,n_i, k=1,...,K. 
    

    M-step (pretrain / train)
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    weighted                    boolean             used weighted mean for random effects
    n_list          [m]         tuple
    - n_i                       scalar              number of observations for i-th subject
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of sampled observations for i-th subject in the batch
    verbose                     boolean             print the terms
    update                      string              M

    
    V-step (Full-batch)
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
    verbose                     boolean             print the terms
    update                      string              V
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


def nhll_hetero_diag(N, y, y_fixed, y_random, v_list, log_phi=None, log_lamb=None, 
                     Gamma_list=None, log_phi_list=None, Gamma_i=None, v_i=None, 
                     weighted=False, n_list=None, batch_n_list=None, verbose=False, update='M') : 
    '''
    Compute -2 * log h-likelihood / N (called the NHLL loss) for random slope model with
    heteroscedastic random noises / diagonal random slope covariance. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    e_ijk i.i.d. ~ N(0, phi_ijk)
    v_ik  ind.   ~ N(0, Sigma_k),    where Sigma_k = diag(lamb_k1, ..., lamb_kp)
    for i=1,...,m, j=1,...,n_i, k=1,...,K. 

    Pretrain step
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    weighted                    boolean             used weighted mean for random effects
    n_list          [m]         tuple
    - n_i                       scalar              number of observations for i-th subject
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of sampled observations for i-th subject in the batch
    verbose                     boolean             print the terms
    update                      string              pretrain
    
    M-step
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [B x K]     torch tensor        Variance of e (heteroscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    weighted                    boolean             used weighted mean for random effects
    n_list          [m]         tuple
    - n_i                       scalar              number of observations for i-th subject
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of sampled observations for i-th subject in the batch
    update                      string              M
    verbose                     boolean             print the terms

    V-step (subject-sampling)
    NAME            SIZE        TYPE                INFO                    
    y               [n_i x K]   torch tensor        Responses       
    y_fixed         [n_i x K]   torch tensor        y_hat = fixed + random
    y_random        [n_i x K]   torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    log_phi         [n_i x K]   torch tensor        Variance of e (heteroscedastic)
    
    v_i             [p x K]     torch parameter     selected random slope
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Design matrix for i-th subject
    update                      string              V
    verbose                     boolean             print the terms

    V-step (full-batch)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]     torch tensor        Responses       
    y_fixed         [N x K]     torch tensor        y_hat = fixed + random
    y_random        [N x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    log_phi         [N x K]     torch tensor        Variance of e (heteroscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)

    log_phi_list    [m]         tuple
    - log_phi_i     [n_i x K]   
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Design matrix for i-th subject
    update                      string              V
    verbose                     boolean             print the terms
    '''

    B, K = y.shape
    m = len(v_list)

    if update == 'pretrain' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) * (batch_n_list[i] / n_list[i]) for i in range(m)]), dim=0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i,2) for v_i in v_list]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
        nhll = torch.sum(term_1 + term_2) 

    elif update == 'M' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) / torch.exp(log_lamb) * (batch_n_list[i] / n_list[i]) for i in range(m)]), dim=0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_list[i],2) / torch.exp(log_lamb) for i in range(m)]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
        nhll = torch.sum(term_1 + term_2) 

    elif update == 'V' : 
        term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0) * m / N
        term_2 = torch.sum(torch.pow(v_i,2) / torch.exp(log_lamb), dim=0) * m / N
        term_3 = torch.sum(log_phi + np.log(2 * np.pi), dim=0) * m / N
        term_4 = torch.sum(log_lamb + np.log(2 * np.pi), dim=0) * m / N
        term_5 = torch.linalg.slogdet(
            torch.cat([(
                torch.diag(torch.exp(-log_lamb[:,k])) + 
                Gamma_i.T @ torch.diag(torch.exp(-log_phi[:,k])) @ Gamma_i
            ).unsqueeze(0) for k in range(K)], dim=0)
        )[1] * m / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')
        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

    elif update == 'V-full' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        term_2 = torch.sum(sum([torch.pow(v_i,2) / torch.exp(log_lamb) for v_i in v_list]), dim=0) / N
        term_3 = torch.mean(log_phi + np.log(2 * np.pi), dim=0)
        term_4 = torch.sum(log_lamb + np.log(2 * np.pi), dim=0) * m / N
        term_5 = sum([torch.linalg.slogdet(
            torch.cat([(
                torch.diag(torch.exp(-log_lamb[:,k])) + 
                Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,k])) @ Gamma_list[i]
            ).unsqueeze(0) for k in range(K)], dim=0)
        )[1] for i in range(m)]) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')
        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)


    else : 
        nhll = 1e8
    return nhll





def nhll_correlated_diag(N, y, y_fixed, y_random, v_list, log_phi=None, log_lamb=None, arctan_rho = None, 
                         Gamma_list = None, log_phi_list = None, 
                         weighted=False, n_list=None, batch_n_list=None, verbose=False, update='M') : 
    '''
    Compute -2 * log h-likelihood / N (called the NHLL loss) for random slope model with
    heteroscedastic random noises / diagonal random slope covariance. 
    Furthermore, here we consider a correlation rho between two random intercepts v_i10 and v_i20. 
    Thus this NHLL loss function only works for the case of K=2. 

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    e_ijk i.i.d. ~ N(0, phi_ijk)
    v_ik  ind.   ~ N(0, Sigma_k),   where Sigma_k = diag(lamb_k1, ..., lamb_kp)
    for i=1,...,m, j=1,...,n_i, k=1,2. 

    Pretrain step
    NAME            SIZE        TYPE                INFO                    
    y               [B x 2]     torch tensor        Responses               
    y_fixed         [B x 2]     torch tensor        y_hat = fixed + random
    y_random        [B x 2]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x 2]     torch parameter     i-th random slope
    weighted                    boolean             used weighted mean for random effects
    n_list          [m]         tuple
    - n_i                       scalar              number of observations for i-th subject
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of sampled observations for i-th subject in the batch
    verbose                     boolean             print the terms
    update                      string              pretrain
    
    M-step
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [B x K]     torch tensor        Variance of e (heteroscedastic)
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    weighted                    boolean             used weighted mean for random effects
    n_list          [m]         tuple
    - n_i                       scalar              number of observations for i-th subject
    batch_n_list    [m]         tuple
    - b_i                       scalar              number of sampled observations for i-th subject in the batch
    update                      string              M
    verbose                     boolean             print the terms

    V-step (subject-sampling)
    NAME            SIZE        TYPE                INFO                    
    y               [n_i x K]   torch tensor        Responses       
    y_fixed         [n_i x K]   torch tensor        y_hat = fixed + random
    y_random        [n_i x K]   torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    log_phi         [n_i x K]   torch tensor        Variance of e (heteroscedastic)
    
    v_i             [p x K]     torch parameter     selected random slope
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Design matrix for i-th subject
    update                      string              V
    verbose                     boolean             print the terms

    
    V-step (full-batch)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]   torch tensor        Responses       
    y_fixed         [N x K]   torch tensor        y_hat = fixed + random
    y_random        [N x K]   torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    log_phi         [N x K]   torch tensor        Variance of e (heteroscedastic)
    
    log_lamb        [p x K]     torch parameter     Variance of v_i (diagonal)
    log_phi_list    [m]         tuple
    - log_phi_i     [n_i x K]   
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Design matrix for i-th subject
    update                      string              V-full
    verbose                     boolean             print the terms

    '''

    B, K = y.shape
    m = len(v_list)

    if update == 'pretrain' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i], 2) * (batch_n_list[i] / n_list[i]) for i in range(m)]), dim=0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i,2) for v_i in v_list]), dim=0) / N
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
        nhll = torch.sum(term_1 + term_2) 

    elif update == 'M' : 
        rho = torch.tanh(arctan_rho)
        sqrt_lamb = torch.exp(0.5 * log_lamb)
        dist_list = [v_i / sqrt_lamb for v_i in v_list]

        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0))
        if weighted : 
            term_2_1 = torch.sum(sum([torch.pow(dist_list[i], 2) * (batch_n_list[i] / n_list[i]) for i in range(m)])) / B
            term_2_2 = sum([
                (rho * (torch.pow(dist_list[i][0,0],2) + torch.pow(dist_list[i][0,1],2)) - 2 * dist_list[i][0,0] * dist_list[i][0,1]) * (batch_n_list[i] / n_list[i])
                for i in range(m)
            ]) * rho / (1-torch.pow(rho, 2)) / B
        else : 
            term_2_1 = torch.sum(sum([torch.pow(dist_list[i],2)  for i in range(m)])) / N
            term_2_2 = sum([
                rho * (torch.pow(dist_i[0,0],2) + torch.pow(dist_i[0,1],2)) - 2 * dist_i[0,0] * dist_i[0,1] 
                for dist_i in dist_list
            ]) * rho / (1-torch.pow(rho, 2)) / N

        if verbose : 
            print(f'Terms : {term_1.item()}, {term_2_1.item()}, {term_2_2.item()}')
        nhll = term_1 + term_2_1 + term_2_2

    # elif update == 'V' : 
    #     p = len(log_lamb)
    #     rho = torch.tanh(arctan_rho)
    #     sqrt_lamb = torch.exp(0.5 * log_lamb)
    #     dist_i = v_i / sqrt_lamb

    #     term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi)) * m / N
    #     term_2_1 = torch.sum(torch.pow(dist_i,2)) * m / N
    #     term_2_2 = (rho * (torch.pow(dist_i[0,0],2) + torch.pow(dist_i[0,1],2)) - 2 * dist_i[0,0] * dist_i[0,1]) * rho / (1-torch.pow(rho, 2)) * m / N
    #     term_3 = torch.sum(log_phi + np.log(2 * np.pi)) * m / N
    #     term_4_1 = torch.sum(log_lamb + np.log(2 * np.pi)) * m / N
    #     term_4_2 = torch.log(1-torch.pow(rho,2)) * m / N

    #     big_inv_Sigma = torch.zeros(2*p, 2*p, device=log_lamb.device)
    #     big_inv_Sigma[:p,:p] = torch.diag(torch.exp(-log_lamb[:,0]))
    #     big_inv_Sigma[p:,p:] = torch.diag(torch.exp(-log_lamb[:,1]))
    #     big_inv_Sigma[0,0]  /= (1-torch.pow(rho,2))
    #     big_inv_Sigma[p,p]  /= (1-torch.pow(rho,2))
    #     big_inv_Sigma[p,0]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))
    #     big_inv_Sigma[0,p]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))

    #     big_inv_Sigma[:p, :p] += Gamma_i.T @ torch.diag(torch.exp(-log_phi[:,0])) @ Gamma_i
    #     big_inv_Sigma[p:, p:] += Gamma_i.T @ torch.diag(torch.exp(-log_phi[:,1])) @ Gamma_i
    #     term_5 = torch.linalg.slogdet(big_inv_Sigma)[1] * m / N

    #     if verbose : 
    #         print(f'Terms : {term_1.item()}, {term_2_1.item()}, {term_2_2.item()}, {term_3.item()}, {term_4_1.item()}, {term_4_2.item()}, {term_5.item()}')
    #     nhll = term_1 + term_2_1 + term_2_2 + term_3 + term_4_1 + term_4_2 + term_5

    elif update == 'V' : 
        p = len(log_lamb)
        rho = torch.tanh(arctan_rho)
        sqrt_lamb = torch.exp(0.5 * log_lamb)
        dist_list = [v_i / sqrt_lamb for v_i in v_list]

        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0))
        term_2_1 = torch.sum(sum([torch.pow(dist_i,2) for dist_i in dist_list])) / N
        term_2_2 = sum([
            (rho * (torch.pow(dist_i[0,0],2) + torch.pow(dist_i[0,1],2)) - 2 * dist_i[0,0] * dist_i[0,1]) * rho / (1-torch.pow(rho, 2))
            for dist_i in dist_list
        ]) / N
        term_3 = torch.sum(log_phi + np.log(2 * np.pi)) / N
        term_4_1 = torch.sum(log_lamb + np.log(2 * np.pi)) * m / N
        term_4_2 = torch.log(1-torch.pow(rho,2)) * m / N

        big_inv_Sigma = torch.zeros(2*p, 2*p, device=log_lamb.device)
        big_inv_Sigma[:p,:p] = torch.diag(torch.exp(-log_lamb[:,0]))
        big_inv_Sigma[p:,p:] = torch.diag(torch.exp(-log_lamb[:,1]))
        big_inv_Sigma[0,0]  /= (1-torch.pow(rho,2))
        big_inv_Sigma[p,p]  /= (1-torch.pow(rho,2))
        big_inv_Sigma[p,0]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))
        big_inv_Sigma[0,p]   = -rho * torch.exp(-0.5 * (log_lamb[0,0] + log_lamb[0,1])) / (1-torch.pow(rho,2))

        big_inv_Sigma_repeated = big_inv_Sigma.repeat(m,1,1)
        for i in range(m) : 
            big_inv_Sigma_repeated[i, :p, :p] += Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,0])) @ Gamma_list[i]
            big_inv_Sigma_repeated[i, p:, p:] += Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,1])) @ Gamma_list[i]
        # temp_term_5 = torch.linalg.slogdet(big_inv_Sigma_repeated)[1]
        # print(temp_term_5)
        term_5 = torch.sum(torch.linalg.slogdet(big_inv_Sigma_repeated)[1]) / N

        # if verbose : 
        # print(f'Terms : {term_1.item()}, {term_2_1.item()}, {term_2_2.item()}, {term_3.item()}, {term_4_1.item()}, {term_4_2.item()}, {term_5.item()}')
        nhll = term_1 + term_2_1 + term_2_2 + term_3 + term_4_1 + term_4_2 + term_5

    else : 
        nhll = 1e8
    return nhll

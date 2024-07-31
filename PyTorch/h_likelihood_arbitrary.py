import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Sigma_module_rev(nn.Module) :
    def __init__(self, p=503, K=2, init_log_lamb = 0, device = torch.device('cpu')) :
        super(Sigma_module_rev, self).__init__()
        '''
        Log-Cholesky decomposition of Sigma
        Sigma[k] = L[k] @ L[k].T
        '''

        self.p = p
        self.K = K
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(K, int(p * (p-1) / 2), device = device) * (2/p) + 1/p) * np.exp(0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(K, p, device=device) + 0.5 * init_log_lamb)

    def recover_L(self) : 
        L = torch.cat([torch.diag(torch.exp(self.L_log_diag[k])).unsqueeze(0) for k in range(self.K)], dim=0)
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = self.L_wo_diag

        return L
    
    def inv_L(self) : 
        return torch.linalg.solve_triangular(self.recover_L(), torch.eye(self.p, device=self.device), upper=False)

    def recover_Sigma(self) : 
        L = self.recover_L()
        return torch.bmm(L, L.transpose(1,2))
    
    def MME_initialize(self, v_list, eps = 1e-6) : 
        m = len(v_list)
        # self.L_log_diag.data = torch.log(sum([torch.pow(v_i,2) for v_i in v_list]) / (m-1) + eps).T
        sample_L = torch.linalg.cholesky(
            sum([torch.bmm(v_i.T.unsqueeze(2), v_i.T.unsqueeze(1)) for v_i in v_list]) / (m-1) + 
            torch.eye(self.p, device=self.device).repeat(self.K,1,1) * eps
            )
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        self.L_wo_diag = sample_L[:, tril_indices[0], tril_indices[1]]

        return None
    



class Large_Sigma_module(nn.Module) :
    def __init__(self, p=503, K=2, init_log_lamb=0, device = torch.device('cpu')) :
        super(Large_Sigma_module, self).__init__()

        self.p = p
        self.K = K
        self.Kp = K * p
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(int(K*p * (K*p-1) / 2), device = device) * .2/(K*p) - .1/(K*p)) * np.exp(0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(K*p, device=device) + 0.5 * init_log_lamb)

    def recover_L(self) : 
        L = torch.diag(torch.exp(self.L_log_diag))
        
        tril_indices = torch.tril_indices(row=self.Kp, col=self.Kp, offset=-1)
        L[tril_indices[0], tril_indices[1]] = self.L_wo_diag

        return L
    
    def inv_L(self) : 
        return torch.linalg.solve_triangular(self.recover_L(), torch.eye(self.Kp, device=self.device), upper=False)

    def recover_Sigma(self) : 
        L = self.recover_L()
        return L @ L.T
        
    def MME_initialize(self, v_list, eps = 1e-6) : 
        m = len(v_list)
        self.L_log_diag.data = torch.log(sum([torch.pow(v_i,2).T.flatten() for v_i in v_list]) / (m-1) + eps)
        return None
    





class Corresponding_precision_module(nn.Module) : 
    def __init__(self, p=503, init_log_lamb=0, device = torch.device('cpu')) :
        '''
        In only works when K=2
        '''
        super(Corresponding_precision_module, self).__init__()

        self.p = p
        self.Kp = 2 * p
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(2, int(p * (p-1) / 2), device = device) * (.2/p) - .1/p) / np.exp(0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(2*p, device=device) - 0.5 * init_log_lamb)
        self.arctan_rho = nn.Parameter(torch.ones(p, device=device) * -0.1)


    def recover_precision(self) : 
        L = torch.diag(torch.exp(self.L_log_diag))
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        L[tril_indices[0], tril_indices[1]] = self.L_wo_diag[0]
        L[self.p+tril_indices[0], self.p + tril_indices[1]] = self.L_wo_diag[1]

        prec = L @ L.T
        diag_term = torch.diag(prec).reshape(2,-1)
        off_diag_mat = torch.diag(torch.tanh(self.arctan_rho) * torch.sqrt(diag_term[0] * diag_term[1]))

        prec[:self.p, self.p : self.Kp] += off_diag_mat
        prec[self.p : self.Kp, :self.p] += off_diag_mat

        return prec
        
    


def nhll_homo_arbitrary(N, y, y_fixed, y_random, v_list, log_phi=None, Sigma=None, Gamma_list=None, 
                        weighted=True, n_list=None, batch_n_list=None, L_inv=None, update='M', verbose=False) : 
    '''
    compute -2 * log h-likelihood / N for random slope model with
    homoscedastic random noises / arbitrary random slope covariance / without correlation between random intercepts

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, phi_k])
    v_ik  ind.   ~ N(0, Sigma_k)

    pretrain
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    M-step
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    weighted
    n_list
    - n_i
    batch_n_list
    - b_i

    V-step (full batch)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]     torch tensor        Responses               
    y_fixed         [N x K]     torch tensor        y_hat = fixed + random
    y_random        [N x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [K]         torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    Gamma_list      [m]         tuple
     - Gamma_i      [n_i x p]   torch tensor        Constructed design matrix
    '''
    B = len(y)
    m = len(v_list)
    p, K = v_list[0].shape


    if update == 'pretrain' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i],2) * batch_n_list[i] / n_list[i] for i in range(m)]), dim = 0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i, 2) for v_i in v_list]), dim=0) / N

        if verbose : 
            print(f'Term 1 and 2 : {term_1}, {term_2}')
    
        nhll = torch.sum(term_1 + term_2)

    # elif update == 'pretrain-eval' : 
    #     term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0))
    #     term_2 = torch.sum(sum([torch.pow(v_i, 2)for v_i in v_list])) / N
    #     term_3 = np.log(2 * np.pi) * K
    #     term_4 = np.log(2 * np.pi) * p * K * m / N

    #     Sigma_inv_repeated = torch.eye(K * p, device=y.device).repeat(m,1,1)
    #     for i in range(m) : 
    #         for k in range(K) : 
    #             Sigma_inv_repeated[i, (k*p):(k*p + p), (k*p):(k*p + p)] += Gamma_list[i].T @ Gamma_list[i]

    #     term_5 = torch.sum(torch.linalg.slogdet(Sigma_inv_repeated)[1]) / N

    #     if verbose : 
    #         print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

    #     nhll = term_1 + term_2 + term_3 + term_4 + term_5
    
    elif update == 'M' : 

        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0) / torch.exp(log_phi)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(torch.bmm(L_inv, v_list[i].T.unsqueeze(2)).squeeze(2), 2) * batch_n_list[i] / n_list[i]for i in range(m)]), dim=1) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(torch.bmm(L_inv, v_i.T.unsqueeze(2)).squeeze(2), 2)for v_i in v_list]), dim=1) / N
        
        if verbose : 
            print(f'Term 1 and 2 : {term_1}, {term_2}')

        nhll = torch.sum(term_1 + term_2)
        
    elif update == 'V' or update == 'V-full' : 
        if Sigma is None : 
            # For pretrain
            L_inv = torch.eye(p, device=y.device).repeat(K,1,1)
        else : 
            L_inv = Sigma.inv_L()

        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0).detach() / torch.exp(log_phi)
        term_2 = torch.sum(sum([
            torch.pow(torch.bmm(L_inv, v_i.T.unsqueeze(2)).squeeze(2), 2)
            for v_i in v_list
        ]), dim=1)  / N
        term_3 = log_phi + np.log(2 * np.pi)
        if Sigma is None : 
            term_4 = torch.sum(torch.zeros(K,p, device=y.device) + np.log(2 * np.pi), dim=1) * m / N
        else : 
            term_4 = torch.sum(2 * Sigma.L_log_diag + np.log(2 * np.pi), dim=1) * m / N
        term_5 = sum([
            torch.linalg.slogdet(
                torch.bmm(L_inv.transpose(1,2), L_inv) + 
                (Gamma_i.T @ Gamma_i).detach().repeat(K,1,1) / torch.exp(log_phi).view(-1,1,1)
            )[1] for Gamma_i in Gamma_list
        ]) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

    else : 
        nhll = 1e8
        
    return nhll



def nhll_hetero_arbitrary(N, y, y_fixed, y_random, v_list, log_phi=None, Sigma=None, Gamma_list=None, log_phi_list=None,
                          weighted=True, n_list=None, batch_n_list=None, L_inv=None, update='M', verbose=False) : 
    '''
    compute -2 * log h-likelihood / N for random slope model with
    homoscedastic random noises / arbitrary random slope covariance / without correlation between random intercepts

    y_ijk = Gamma(x_ij).T (beta_k + v_ik) + e_ijk
    for i=1,...,m, j=1,...,n_i, k=1,...,K
    e_ijk i.i.d. ~ N(0, phi_k])
    v_ik  ind.   ~ N(0, Sigma_k)

    pretrain
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope

    M-step
    NAME            SIZE        TYPE                INFO                    
    y               [B x K]     torch tensor        Responses               
    y_fixed         [B x K]     torch tensor        y_hat = fixed + random
    y_random        [B x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [p x K]     torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    weighted
    n_list
    - n_i
    batch_n_list
    - b_i

    V-step (full batch)
    NAME            SIZE        TYPE                INFO                    
    y               [N x K]     torch tensor        Responses               
    y_fixed         [N x K]     torch tensor        y_hat = fixed + random
    y_random        [N x K]     torch tensor        y_hat = fixed + random
    v_list          [m]         tuple               
    - v_i           [p x K]     torch parameter     i-th random slope
    log_phi         [N x K]     torch parameter     Variance of e (homoscedastic)
    Sigma                       torch.nn.Module     Variance of v
    - L_wo_diag     [K x p x p] torch parameter
    - L_log_diag    [K x p]     torch parameter
    Gamma_list      [m]         tuple
    - Gamma_i       [n_i x p]   torch tensor        Constructed design matrix
    log_phi_list    [m]         tuple
    - log_phi_i     [n_i x K]   torch tensor        i-th response random noises variance
    '''
    B = len(y)
    m = len(v_list)
    p, K = v_list[0].shape


    if update == 'pretrain' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i],2) * batch_n_list[i] / n_list[i] for i in range(m)]), dim = 0) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i, 2) for v_i in v_list]), dim=0) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
    
        nhll = torch.sum(term_1 + term_2)
    
    elif update == 'M' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([
                torch.pow(torch.bmm(L_inv, v_list[i].T.unsqueeze(2)).squeeze(2), 2) * batch_n_list[i] / n_list[i]
                for i in range(m)
            ]), dim=1) / B
        else : 
            term_2 = torch.sum(sum([
                torch.pow(torch.bmm(L_inv, v_i.T.unsqueeze(2)).squeeze(2), 2)
                for v_i in v_list
            ]), dim=1) / N
        
        if verbose : 
            print(f'Terms : {term_1}, {term_2}')

        nhll = torch.sum(term_1 + term_2)
        
    elif update == 'V' or update == 'V-full' : 
        if Sigma is None : 
            # For pretrain
            L_inv = torch.eye(p, device=y.device).repeat(K,1,1)
        else : 
            L_inv = Sigma.inv_L()

        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        term_2 = torch.sum(sum([
            torch.pow(torch.bmm(L_inv, v_i.T.unsqueeze(2)).squeeze(2), 2)
            for v_i in v_list
        ]), dim=1)  / N
        term_3 = torch.mean(log_phi + np.log(2 * np.pi), dim=0)
        if Sigma is None : 
            term_4 = torch.sum(torch.zeros(K,p, device=y.device) + np.log(2 * np.pi), dim=1) * m / N
        else : 
            term_4 = torch.sum(2 * Sigma.L_log_diag + np.log(2 * np.pi), dim=1) * m / N
        term_5 = sum([
            torch.linalg.slogdet(
                torch.bmm(L_inv.transpose(1,2), L_inv) + 
                torch.cat([
                    (Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,k])) @ Gamma_list[i]).unsqueeze(0)
                    for k in range(K)
                ], dim=0)
            )[1] for i in range(m)
        ]) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

    else : 
        nhll = 1e8
        
    return nhll



def nhll_correlated_arbitrary(N, y, y_fixed, y_random, v_list, log_phi=None, Large_Sigma=None, 
                              Gamma_list=None, log_phi_list=None,weighted=True, n_list=None, batch_n_list=None, L_inv=None,
                              update='M', verbose=False) : 
    B = len(y)
    m = len(v_list)
    p, K = v_list[0].shape

    if update == 'pretrain' : 
        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0))
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i],2) * batch_n_list[i] / n_list[i] for i in range(m)])) / B
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i, 2) for v_i in v_list])) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
    
        nhll = term_1 + term_2

    elif update == 'pretrain-eval' : 

        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2), dim=0))
        term_2 = torch.sum(sum([torch.pow(v_i, 2)for v_i in v_list])) / N
        term_3 = np.log(2 * np.pi) * K
        term_4 = np.log(2 * np.pi) * p * K * m / N

        Sigma_inv_repeated = torch.eye(K * p, device=y.device).repeat(m,1,1)
        for i in range(m) : 
            for k in range(K) : 
                Sigma_inv_repeated[i, (k*p):(k*p + p), (k*p):(k*p + p)] += Gamma_list[i].T @ Gamma_list[i]

        term_5 = torch.sum(torch.linalg.slogdet(Sigma_inv_repeated)[1]) / N

        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

        nhll = term_1 + term_2 + term_3 + term_4 + term_5

    
    elif update == 'M' : 


        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0))

        if weighted : 
            term_2 = torch.sum(sum([
                torch.pow(L_inv @ v_list[i].T.reshape(-1,1), 2) * batch_n_list[i] / n_list[i]
                for i in range(m)
            ])) / B
        else : 
            term_2 = torch.sum(sum([
                torch.pow(L_inv @ v_i.T.reshape(-1,1), 2) for v_i in v_list
            ])) / N
        
        if verbose : 
            print(f'Terms : {term_1.item()}, {term_2.item()}')

        nhll = term_1 + term_2
        
    elif update == 'V' or update == 'V-full' : 

        L_inv = Large_Sigma.inv_L()

        term_1 = torch.sum(torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0))
        term_2 = torch.sum(sum([
             torch.pow(L_inv @ v_i.T.reshape(-1,1), 2) for v_i in v_list
        ])) / N
        term_3 = torch.sum(torch.mean(log_phi + np.log(2 * np.pi), dim=0))
        term_4 = torch.sum(2 * Large_Sigma.L_log_diag + np.log(2 * np.pi)) * m / N


        Sigma_inv_repeated = (L_inv.T @ L_inv).repeat(m,1,1)
        for i in range(m) : 
            for k in range(K) : 
                Sigma_inv_repeated[i, (k*p):(k*p + p), (k*p):(k*p + p)] += Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,k])) @ Gamma_list[i]

        term_5 = torch.sum(torch.linalg.slogdet(Sigma_inv_repeated)[1]) / N
    
        if verbose : 
            print(f'Terms : {term_1}, {term_2}, {term_3}, {term_4}, {term_5}')

        nhll = term_1 + term_2 + term_3 + term_4 + term_5

    else : 
        nhll = 1e8
        
    return nhll






def nhll_correlated_corresponding(N, y, y_fixed, y_random, v_list, log_phi=None, Precision=None, 
                                  Gamma_list=None, log_phi_list=None,weighted=True, n_list=None, batch_n_list=None, prec=None,
                                  init_log_lamb = 0, update='M', verbose=False) : 
    B = len(y)
    m = len(v_list)
    p, K = v_list[0].shape

    if update == 'pretrain' : 
        term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2)) / B
        if weighted : 
            term_2 = torch.sum(sum([torch.pow(v_list[i],2) * batch_n_list[i] / n_list[i] for i in range(m)])) / B / np.exp(init_log_lamb)
        else : 
            term_2 = torch.sum(sum([torch.pow(v_i, 2) for v_i in v_list])) / N / np.exp(init_log_lamb)

        if verbose : 
            print(f'Terms : {term_1}, {term_2}')
    
        nhll = term_1 + term_2

    elif update == 'pretrain-eval' : 

        term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2)) / N
        term_2 = sum([torch.sum(torch.pow(v_i,2)) for v_i in v_list]) / N / np.exp(init_log_lamb)
        term_3 = np.log(2 * np.pi) * K
        term_4 = np.log(2 * np.pi) * p * K * m / N

        prec_repeated = torch.eye(K*p, K*p, device=y.device).repeat(m,1,1) / np.exp(init_log_lamb)
        for i in range(m) : 
            for k in range(K) : 
                prec_repeated[i, (k*p):(k*p + p), (k*p):(k*p + p)] += Gamma_list[i].T @ Gamma_list[i]

        term_5 = torch.sum(torch.linalg.slogdet(prec_repeated)[1]) / N

        if verbose : 
            print(f'Terms : {term_1.item()}, {term_2.item()}, {term_3.item()}, {term_4.item()}, {term_5.item()}')

        nhll = term_1 + term_2 + term_3 + term_4 + term_5

    
    elif update == 'M' : 

        term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi)) / B
        if weighted : 
            term_2 = sum([torch.sum(v_list[i].T.flatten() * (prec @ v_list[i].T.flatten())) * batch_n_list[i] / n_list[i] for i in range(m)]) / B
        else : 
            term_2 = sum([torch.sum(v_i.T.flatten() * (prec @ v_i.T.flatten())) for v_i in v_list]) / N

        if term_2 < 0 :
            print(f'Warning! Positive definiteness of Precision does not hold : vT P v = {term_2.item():.4f}')
            return None
        
        if verbose : 
            print(f'Terms : {term_1.item()}, {term_2.item()}')

        nhll = term_1 + term_2
        
    elif update == 'V' or update == 'V-full' : 
        prec = Precision.recover_precision()

        term_1 = torch.sum(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi)) / N
        term_2 = sum([torch.sum(v_i.T.flatten() * (prec @ v_i.T.flatten())) for v_i in v_list]) / N

        if term_2 < 0 :
            print(f'Warning! Positive definiteness of Precision does not hold : vT P v = {term_2.item():.4f}')
            return None

        term_3 = torch.sum(log_phi + np.log(2 * np.pi)) / N
        term_4 = -torch.linalg.slogdet(prec / (2 * np.pi))[1] * m / N

        prec_repeated = prec.repeat(m,1,1)
        for i in range(m) : 
            for k in range(K) : 
                prec_repeated[i, (k*p):(k*p + p), (k*p):(k*p + p)] += Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,k])) @ Gamma_list[i]

        term_5 = torch.sum(torch.linalg.slogdet(prec_repeated)[1]) / N

        if verbose : 
            print(f'Terms : {term_1.item()}, {term_2.item()}, {term_3.item()}, {term_4.item()}, {term_5.item()}')

        nhll = term_1 + term_2 + term_3 + term_4 + term_5

    else : 
        nhll = 1e8
        
    return nhll
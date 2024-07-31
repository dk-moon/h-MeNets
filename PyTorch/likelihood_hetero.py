import torch
import numpy as np

def multivariate_njll_hetero_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative joint log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [n_i x K]   torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i = len(y_i)

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / sigma_sq, dim=0)
    term_2 = torch.sum(torch.log(2 * torch.pi * sigma_sq), dim=0)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * Sigma_v)[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4)


def multivariate_nll_hetero_i(y_i, f_hat_i, Gamma_i, v_i=None, sigma_sq=None, Sigma_v=None) : 
    '''
    Compute 2 x negative (integrated) log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [n_i x K]   torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i, K = y_i.shape

    e_hat_i = y_i - f_hat_i
    V_i = torch.cat([torch.diag(sigma_sq[:,k]).unsqueeze(0) for k in range(K)])
    V_i += torch.bmm(Gamma_i.repeat(K,1,1), torch.bmm(Sigma_v, Gamma_i.T.repeat(K,1,1)))
    # print(f'V_i s log det : {torch.linalg.slogdet(V_i)[1]}')

    term_1 = torch.sum(e_hat_i.T * torch.linalg.solve(V_i, e_hat_i.T), dim=1)
    term_2 = torch.linalg.slogdet(2 * torch.pi * V_i)[1]

    return torch.sum(term_1 + term_2)



def multivariate_nhll_hetero_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative hierarchical log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i, K = y_i.shape

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / sigma_sq, dim=0)
    term_2 = torch.sum(torch.log(2 * torch.pi * sigma_sq), dim=0)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * Sigma_v)[1]
    term_5 = torch.linalg.slogdet(
        torch.linalg.inv(Sigma_v) + 
        torch.bmm(
            Gamma_i.T.repeat(K,1,1), 
            torch.bmm(
                torch.cat([torch.diag(1/sigma_sq[:,k]).unsqueeze(0) for k in range(K)]), 
                Gamma_i.repeat(K,1,1)
            ))
        )[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

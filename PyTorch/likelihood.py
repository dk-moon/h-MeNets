import torch
import numpy as np


def njll_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative joint log-likelihood for i-th subject in random slope model.
    y_{ij} = Gamma(x_{ij}).T (beta + v_i) + e_{ij}
           = f_hat_i + Gamma_i @ v_i + e_{ij} 
    e_{ij} ~ N(0, sigma_sq)
    v_i    ~ N(0, Sigma_v)

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i]       torch tensor        response
    f_hat_i         [n_i]       torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p]         torch tensor        estimated random slope for i-th subject
    sigma_sq                    scalar              estimated variance of e
    Sigma_v         [p x p]     torch tensor        estimated variance matrix of random slope v
    '''

    n_i = y_i.shape[0]
    term_1 = torch.norm(y_i - f_hat_i - Gamma_i @ v_i) / sigma_sq
    term_2 = n_i * np.log(2 * np.pi * sigma_sq)
    term_3 = v_i @ torch.linalg.solve(Sigma_v, v_i)
    term_4 = torch.linalg.slogdet(2 * np.pi * Sigma_v)[1]

    return term_1 + term_2 + term_3 + term_4



def multivariate_njll_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative joint log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i = len(y_i)

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / sigma_sq, dim=0)
    term_2 = n_i * torch.log(2 * torch.pi * sigma_sq)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * Sigma_v)[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4)



def multivariate_nll_i(y_i, f_hat_i, Gamma_i, v_i=None, sigma_sq=None, Sigma_v=None) : 
    '''
    Compute 2 x negative (integrated) log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i, K = y_i.shape

    e_hat_i = y_i - f_hat_i
    V_i = torch.eye(n_i, device=sigma_sq.device).repeat(K, 1, 1) * sigma_sq.view(-1,1,1)
    V_i += torch.bmm(Gamma_i.repeat(K,1,1), torch.bmm(Sigma_v, Gamma_i.T.repeat(K,1,1)))

    term_1 = torch.sum(e_hat_i.T * torch.linalg.solve(V_i, e_hat_i.T), dim=1)
    term_2 = torch.linalg.slogdet(2 * torch.pi * V_i)[1]

    return torch.sum(term_1 + term_2)



def multivariate_nhll_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
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
    term_2 = n_i * torch.log(2 * torch.pi * sigma_sq)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * Sigma_v)[1]
    term_5 = torch.linalg.slogdet(torch.linalg.inv(Sigma_v) + (Gamma_i.T @ Gamma_i).repeat(K,1,1) / sigma_sq.view(-1,1,1))[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)



def multivariate_nhll_diag_i(y_i, f_hat_i, Gamma_i, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative hierarchical log-likelihood for i-th subject, with approximating Sigma_v being diagonal

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    f_hat_i         [n_i x K]   torch tensor        estimated fixed part for i-th subject (= Gamma_i @ beta)
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i, K = y_i.shape
    diag_Sigma_v = torch.zeros_like(Sigma_v)
    for k in range(K) : 
        diag_Sigma_v[k] = torch.diag(torch.diag(Sigma_v[k]))

    term_1 = torch.sum(torch.pow(y_i - f_hat_i - Gamma_i @ v_i, 2) / sigma_sq, dim=0)
    term_2 = n_i * torch.log(2 * torch.pi * sigma_sq)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(diag_Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * diag_Sigma_v)[1]
    term_5 = torch.linalg.slogdet(torch.linalg.inv(diag_Sigma_v) + (Gamma_i.T @ Gamma_i).repeat(K,1,1) / sigma_sq.view(-1,1,1))[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)



def EM_update(y_list, Gamma_list, beta, v_list, sigma_sq, Sigma_v, n_list, use_woodbury = False) : 
    '''
    Update v_list, sigma_sq and Sigma_v according to variational SGD + EM algorithm. 

    NAME            SIZE        TYPE                INFO    
    y_list          [m]         tuple               
     - y_i          [n_i x K]   torch tensor        response for i-th subject
    Gamma_list      [m]         tuple
     - Gamma_i      [n_i x p]   torch tensor        design matrix for i-th subject
    beta            [p x K]     torch tensor        coefficient
    v_list          [m]         tuple
     - v_i          [p x K]     torch tensor        random slope for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    n_list          [m]         tuple           
     - n_i                      scalar              number of observations for i-th subject
    '''
    K = sigma_sq.shape[0]
    m = len(y_list)

    f_hat_list = [Gamma_i @ beta for Gamma_i in Gamma_list]

    if use_woodbury : 
        inv_V_list = [
            torch.eye(n_list[i]).repeat(K,1,1) / sigma_sq.view(-1,1,1) - torch.bmm(
                Gamma_list[i].repeat(K,1,1), 
                torch.bmm(
                    torch.linalg.inv(
                        torch.linalg.inv(Sigma_v) + 
                        torch.bmm(Gamma_list[i].T.repeat(K,1,1), Gamma_list[i].repeat(K,1,1)) / sigma_sq.view(-1,1,1)
                    ), 
                    Gamma_list[i].T.repeat(K,1,1)
                )
            ) / sigma_sq.pow(2).view(-1,1,1) for i in range(m)]
    else : 
        inv_V_list = [torch.linalg.inv(
            torch.eye(n_list[i]).repeat(K,1,1) * sigma_sq.view(-1,1,1) + 
            torch.bmm(Gamma_list[i].repeat(K,1,1), torch.bmm(Sigma_v, Gamma_list[i].T.repeat(K,1,1)))
        ) for i in range(m)]

    new_v_list = [
        torch.bmm(
            torch.bmm(
                torch.bmm(
                    Sigma_v, 
                    Gamma_list[i].T.repeat(K,1,1)
                ), inv_V_list[i]
            ), (y_list[i] - Gamma_list[i] @ beta).T.unsqueeze(2)
        ).squeeze(2).T for i in range(m)
    ]
    
    new_eps_list = [
        y_list[i] - f_hat_list[i] - Gamma_list[i] @ new_v_list[i]
        for i in range(m)
    ]

    new_sigma_sq = sum([
        torch.sum(torch.pow(new_eps_list[i], 2), dim=0) + 
        sigma_sq * (n_list[i] - sigma_sq * torch.sum(torch.diagonal(inv_V_list[i], offset=0, dim1=1, dim2=2), dim=1))
        for i in range(m)
    ]) / sum(n_list)

    new_Sigma_v = sum([
        torch.bmm(new_v_list[i].T.unsqueeze(2), new_v_list[i].T.unsqueeze(1)) + Sigma_v - 
        torch.bmm(
            Sigma_v, 
            torch.bmm(
                Gamma_list[i].T.repeat(K,1,1), 
                torch.bmm(
                    inv_V_list[i], 
                    torch.bmm(
                        Gamma_list[i].repeat(K,1,1), 
                        Sigma_v
                    )
                )
            )
        ) for i in range(m)
    ]) / m

    return new_v_list, new_sigma_sq, new_Sigma_v
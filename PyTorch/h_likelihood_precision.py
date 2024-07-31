import torch
import numpy as np
import torch.nn as nn

from precision_module import precision_module

def nhll_hetero_precision(N, y, y_fixed, y_random, v_list, log_phi=None, precision=None, Gamma_list=None, log_phi_list=None,
                          weighted=True, n_list=None, batch_n_list=None, LT=None, init_log_lamb=0, update='M', verbose=False) : 
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
            print(f'Terms : ({term_1[0].item():.4f}, {term_1[1].item():.4f}), ({term_2[0].item():.4f}, {term_2[1].item():.4f})')
    
        nhll = torch.sum(term_1 + term_2)

    elif update == 'pretrain-eval' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2))
        term_2 = sum([torch.sum(torch.pow(v_i,2)) for v_i in v_list]) / N / np.exp(init_log_lamb)

        term_3 = np.log(2 * np.pi) * K
        term_4 = np.log(2 * init_log_lamb + 2 * np.pi) * p * K * m / N

        term_5 = sum([
            torch.linalg.slogdet(
                torch.eye(p, p, device=y.device) + Gamma_list[i].T @ Gamma_list[i]
            )[1] for i in range(m)
        ]) * K / N

        if verbose : 
            print(f'Terms : {term_1.item():.4f}, {term_1.item():.4f}, {term_1.item():.4f}, {term_1.item():.4f}, {term_1.item():.4f}')

        nhll = term_1 + term_2 + term_3 + term_4 + term_5

    
    elif update == 'M' : 
        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        if weighted : 
            term_2 = torch.sum(sum([
                torch.pow(torch.bmm(LT, v_list[i].T.unsqueeze(2)).squeeze(2), 2) * batch_n_list[i] / n_list[i]
                for i in range(m)
            ]), dim=1) / N
        else : 
            term_2 = torch.sum(sum([torch.pow(torch.bmm(LT, v_i.T.unsqueeze(2)).squeeze(2), 2) for v_i in v_list]), dim=1) / N

        if verbose : 
            print(f'Terms : ({term_1[0].item():.4f}, {term_1[1].item():.4f}), ({term_2[0].item():.4f}, {term_2[1].item():.4f})')

        nhll = torch.sum(term_1 + term_2)


    elif update == 'V' : 
        LT = precision.recover_LT()

        term_1 = torch.mean(torch.pow(y - y_fixed - y_random, 2) / torch.exp(log_phi), dim=0)
        term_2 = torch.sum(sum([torch.pow(torch.bmm(LT, v_i.T.unsqueeze(2)).squeeze(2), 2) for v_i in v_list]), dim=1) / N

        term_3 = torch.mean(log_phi + np.log(2 * np.pi), dim=0)
        term_4 = torch.sum(-2 * precision.L_log_diag + np.log(2 * np.pi), dim=1) * m / N

        term_5 = sum([
            torch.linalg.slogdet(
                LT.mT @ LT + 
                torch.cat([
                    (Gamma_list[i].T @ torch.diag(torch.exp(-log_phi_list[i][:,k])) @ Gamma_list[i]).unsqueeze(0) 
                    for k in range(K)
                ], dim=0)
            )[1] for i in range(m)
        ]) / N

        if verbose : 
            print(f'Terms : ({term_1[0].item():.4f}, {term_1[1].item():.4f}), ({term_2[0].item():.4f}, {term_2[1].item():.4f}), ({term_3[0].item():.4f}, {term_3[1].item():.4f}), ({term_4[0].item():.4f}, {term_4[1].item():.4f}), ({term_5[0].item():.4f}, {term_5[1].item():.4f})')

        nhll = torch.sum(term_1 + term_2 + term_3 + term_4 + term_5)

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
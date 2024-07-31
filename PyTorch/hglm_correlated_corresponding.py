import time
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression

from h_likelihood_arbitrary import Corresponding_precision_module, nhll_correlated_corresponding
from util import make_reproducibility, TensorDataset, convert_to_spherical, convert_to_xyz, mae

def correlated_corresponding_without_val(
    train_ids, train_images, train_hps, train_gazes, 
    test_ids, test_images, test_hps, test_gazes, 
    mean_network, hidden_features, K=2, init_log_lamb=0, 
    mean_lr=2e-2, variance_lr=5e-3, weight_decay=0, batch_size=1000, 
    pretrain_iter=5, m_pretrain_epoch=10, v_pretrain_epoch=10, max_iter=30, mean_epoch=10, v_step_iter=100, patience=5, 
    device=torch.device('cpu'), experiment_name=1, SEED=None, 
    normalize=True, deg=True, test_unseen=False, weighted=True, variance_check=True, verbose=True, bins=40, large_test=False, reset_opt=True, betas=(0.9, 0.999)) : 
    
    torch.cuda.empty_cache()
    if SEED is not None : 
        make_reproducibility(SEED)
    
    if normalize : 
        train_images = train_images / 255.0
    train_ids_unique = np.unique(train_ids)
    m = len(train_ids_unique)
    train_hps = convert_to_spherical(train_hps, deg=deg).float()
    train_y = convert_to_spherical(train_gazes, deg=deg).float()
    train_y_cuda = train_y.to(device)
    train_N = len(train_gazes)
    train_cluster = [np.where(train_ids == idx)[0] for idx in train_ids_unique]
    train_n_list = [len(cluster) for cluster in train_cluster]

    train_dataset = TensorDataset(train_images, train_hps, train_y, train_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if normalize : 
        test_images = test_images / 255.0
    test_hps = convert_to_spherical(test_hps, deg=deg).float()
    test_y = convert_to_spherical(test_gazes, deg=deg).float()
    test_y_cuda = test_y.to(device)
    test_N = len(test_gazes)
    if test_unseen : 
        if large_test : 
            test_ids_unique = np.unique(test_ids)
            test_cluster = [np.where(test_ids == idx)[0] for idx in test_ids_unique]
            test_m=len(test_ids_unique)
        else : 
            test_cluster = [np.arange(test_N)]
            test_m=1
    else : 
        test_cluster = [np.where(test_ids == idx)[0] for idx in train_ids_unique]
        test_m = len(test_cluster)
    test_n_list = [len(cluster) for cluster in test_cluster]


    # Initialize a mean neural network and ligression coefficients for log variance
    mean_model=mean_network(hidden_features=hidden_features, out_features=K).to(device)
    p = mean_model.p
    log_phi_layer=nn.Linear(in_features=p, out_features=K, bias=False).to(device)

    # Initialize other parameters
    # Xavier (uniform) initialization for v_i with approximation \sqrt{504 / 6} ~ 9. 
    v_list  = [nn.Parameter(torch.rand(p, K, device=device) * 2/9 - 1/9) for _ in range(m)]
    precision = Corresponding_precision_module(p, init_log_lamb=init_log_lamb, device=device).to(device)

    # Initialize optimizers
    mean_optimizer     = optim.Adam(list(mean_model.parameters())    + v_list,     lr=mean_lr,     weight_decay=weight_decay)
    variance_optimizer = optim.Adam(list(log_phi_layer.parameters()) + list(precision.parameters()), lr=variance_lr, weight_decay=weight_decay)

    # Ready for save
    prediction = np.zeros((pretrain_iter + max_iter, test_N, 3))
    prediction_adjusted = np.zeros((pretrain_iter + max_iter, test_N, 3))

    pretrain_m_loss_list = []
    pretrain_v_loss_list = []
    train_m_loss_list = []
    train_v_loss_list = []
    train_nhll_loss_list = []

    train_loss_list = np.zeros((5,  pretrain_iter + max_iter))
    test_loss_list  = np.zeros((10, pretrain_iter + max_iter))

    v_list_list     = np.zeros((pretrain_iter + max_iter, m, p, K))
    beta_list       = np.zeros((pretrain_iter + max_iter, p, K))
    w_list          = np.zeros((pretrain_iter + max_iter, p, K))
    
    precision_list = np.zeros((3, K * p, K * p))
    train_Gamma_list_list = np.zeros((3, train_N, p))
    test_Gamma_list_list  = np.zeros((3, test_N,  p))

    # Visualization of sample variance of initial v_i
    if variance_check : 
        temp_log_lamb = torch.log(sum([torch.pow(v_i,2) for v_i in v_list]) / (m-1) + 1e-10)
        fig = plt.figure(figsize = (7, 7))
        ax = fig.add_subplot(2,2,1)
        ax.hist(temp_log_lamb[:,0].detach().cpu().numpy(), bins=bins)
        ax.set_title('Initial sample variance 1 (log)')
        ax = fig.add_subplot(2,2,2)
        ax.hist(temp_log_lamb[:,1].detach().cpu().numpy(), bins=bins)
        ax.set_title('Initial sample variance 2 (log)')
        ax = fig.add_subplot(2,2,3)
        ax.hist(temp_log_lamb[:,0].detach().exp().cpu().numpy(), bins=bins)
        ax.set_title('Initial sample variance 1')
        ax = fig.add_subplot(2,2,4)
        ax.hist(temp_log_lamb[:,1].detach().exp().cpu().numpy(), bins=bins)
        ax.set_title('Initial sample variance 2')
        plt.show()
    
    # Pretrain 
    print('Mean Pretrain starts')
    pretrain_start = time.time()
    for iter in tqdm(range(pretrain_iter)) : 
        mean_model.train()
        precision.eval()
        log_phi_layer.eval()
        for epoch in range(m_pretrain_epoch) : 
            for _, (image, hp, y, batch_ids) in enumerate(train_loader) : 
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)
                batch_cluster = [np.where(np.asarray(batch_ids) == idx)[0] for idx in train_ids_unique]
                batch_n_list = [len(cluster) for cluster in batch_cluster]

                batch_Gamma = mean_model.get_feature_map(image, hp)
                batch_fixed = mean_model.fc2(batch_Gamma)       
                batch_random = torch.zeros_like(batch_fixed)
                for i in range (m) : 
                    batch_random[batch_cluster[i]] += batch_Gamma[batch_cluster[i]] @ v_list[i]

                mean_optimizer.zero_grad()
                variance_optimizer.zero_grad()
                pretrain_loss = nhll_correlated_corresponding(train_N, y, batch_fixed, batch_random, v_list, weighted=weighted, n_list=train_n_list, batch_n_list=batch_n_list, init_log_lamb=init_log_lamb, update='pretrain', verbose=verbose)
                pretrain_loss.backward()
                mean_optimizer.step()
                pretrain_m_loss_list.append(pretrain_loss.item())
                if verbose : 
                    print(f'Pretrain h-lik loss (m-step) : {pretrain_loss.item()}')

            print(f'{epoch}-th epoch last batch Pretrain h-lik loss (m-step) : {pretrain_loss.item()}')

        # Pretrain evaluation
        mean_model.eval()
        with torch.no_grad() : 
            # Evaluation 1 : random effect adjustment
            train_Gamma = torch.zeros(train_N, p).to(device)
            
            train_fixed = torch.zeros_like(train_y_cuda)
            train_random = torch.zeros_like(train_y_cuda)
            for i in range(m) : 
                train_Gamma[train_cluster[i]]  = mean_model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)).detach()
                train_fixed[train_cluster[i]]  = mean_model.fc2(train_Gamma[train_cluster[i]])
                train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ v_list[i]

            w = LinearRegression(fit_intercept=False)
            w.fit(X=train_Gamma.cpu(), y=train_random.cpu())
            w_beta = torch.as_tensor(w.coef_).T.to(device)

            # Evaluation 2 : Train MAE / MSE / NLL / NJLL / NHLL
            # prec = precision.recover_precision()
            # train_y_list = [train_y_cuda[cluster] for cluster in train_cluster]
            train_Gamma_list = [train_Gamma[cluster] for cluster in train_cluster]
            # train_fixed_list = [train_fixed[cluster] for cluster in train_cluster]

            train_mae  = mae(train_y_cuda, train_fixed + train_random, is_3d=False, deg=deg).item()
            train_mse  = F.mse_loss(train_y_cuda, train_fixed + train_random).item()
            train_nhll = nhll_correlated_corresponding(train_N, train_y_cuda, train_fixed, train_random, v_list, Gamma_list = train_Gamma_list, init_log_lamb=init_log_lamb, update='pretrain-eval', verbose=verbose)

            train_loss_list[0, iter] = train_mae
            train_loss_list[1, iter] = train_mse
            train_loss_list[4, iter] = train_nhll
            print(f'{iter}-th Pretrain train MAE, MSE, NHLL : {train_mae:.4f} deg, {train_mse:.4f}, {train_nhll:.4f}')

            # Evaluation 3 : Test MAE / MSE / NLL / NJLL / NHLL
            if large_test : 
                test_Gamma = torch.zeros(test_N, p, device=device)
                test_fixed = torch.zeros_like(test_y_cuda)
                for cluster in test_cluster : 
                    test_Gamma[cluster] = mean_model.get_feature_map(test_images[cluster].to(device), test_hps[cluster].to(device)).detach()
                    test_fixed[cluster] = mean_model.fc2(test_Gamma[cluster])
            else : 
                test_Gamma = mean_model.get_feature_map(test_images.to(device), test_hps.to(device)).detach()
                test_fixed = mean_model.fc2(test_Gamma)
            
            test_adjusted = test_Gamma @ w_beta
            if test_unseen is False : 
                test_random = torch.zeros_like(test_fixed)
                for i in range(m) : 
                    test_random[test_cluster[i]] = test_Gamma[test_cluster[i]] @ v_list[i]

            # test_y_list        = [test_y_cuda[cluster]   for cluster in test_cluster]
            test_Gamma_list    = [test_Gamma[cluster]    for cluster in test_cluster]
            # test_fixed_list    = [test_fixed[cluster]    for cluster in test_cluster]
            # test_adjusted_list = [test_adjusted[cluster] for cluster in test_cluster]

            if test_unseen is True : 
                # LOOCV
                test_mae = mae(test_y_cuda, test_fixed, is_3d=False, deg=deg).item()
                test_mse = F.mse_loss(test_y_cuda, test_fixed).item()
                
                test_mae_adjusted = mae(test_y_cuda, test_fixed + test_adjusted, is_3d=False, deg=deg).item()
                test_mse_adjusted = F.mse_loss(test_y_cuda, test_fixed + test_adjusted).item()
                
                test_loss_list[0, iter] = test_mae
                test_loss_list[1, iter] = test_mse
                test_loss_list[3, iter] = test_mae_adjusted
                test_loss_list[4, iter] = test_mse_adjusted
                print(f'{iter}-th Pretrain test MAE, MSE : {test_mae:.4f} deg, {test_mse:.4f}')
                print(f'{iter}-th Pretrain test MAE, MSE (adjusted y_hat) : {test_mae_adjusted:.4f} deg, {test_mse_adjusted:.4f}')

                prediction[iter]          = convert_to_xyz(test_fixed,                 deg=deg).cpu().numpy()
                prediction_adjusted[iter] = convert_to_xyz(test_fixed + test_adjusted, deg=deg).cpu().numpy()
            else : 
                # Within evaluation
                test_mae  = mae(test_y_cuda, test_fixed + test_random, is_3d=False, deg=deg).item()
                test_mse  = F.mse_loss(test_y_cuda, test_fixed + test_random).item()
                test_nhll = nhll_correlated_corresponding(test_N, test_y_cuda, test_fixed, test_random, v_list,Gamma_list = test_Gamma_list, init_log_lamb=init_log_lamb, update='pretrain-eval', verbose=verbose).item()


                test_mae_adjusted  = mae(test_y_cuda, test_fixed + test_adjusted, is_3d=False, deg=deg).item()
                test_mse_adjusted  = F.mse_loss(test_y_cuda, test_fixed + test_adjusted).item()
                # test_nll_adjusted  = sum([multivariate_nll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], 0, torch.ones(test_n_list[i],K, device=device), Sigma_v) for i in range(m)]).item() / test_N
                # test_njll_adjusted = sum([multivariate_njll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], v_list[i] - w_beta, torch.ones(test_n_list[i],K, device=device), Sigma_v) for i in range(m)]).item() / test_N
                # test_nhll_adjusted = sum([multivariate_nhll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], v_list[i] - w_beta, torch.ones(test_n_list[i],K, device=device), Sigma_v) for i in range(m)]).item() / test_N
            
                test_loss_list[0, iter] = test_mae
                test_loss_list[1, iter] = test_mse
                # test_loss_list[2, iter] = test_nll
                # test_loss_list[3, iter] = test_njll
                test_loss_list[4, iter] = test_nhll
                test_loss_list[5, iter] = test_mae_adjusted
                test_loss_list[6, iter] = test_mse_adjusted
                # test_loss_list[7, iter] = test_nll_adjusted
                # test_loss_list[8, iter] = test_njll_adjusted
                # test_loss_list[9, iter] = test_nhll_adjusted
                print(f'{iter}-th Pretrain test MAE, MSE, NHLL : {test_mae:.4f} deg, {test_mse:.4f}, {test_nhll:.4f}')
                print(f'{iter}-th Pretrain test MAE, MSE, NLL (adjusted y_hat) : {test_mae_adjusted:.4f} deg, {test_mse_adjusted:.4f}')

                prediction[iter]          = convert_to_xyz(test_fixed + test_random,   deg=deg).cpu().numpy()
                prediction_adjusted[iter] = convert_to_xyz(test_fixed + test_adjusted, deg=deg).cpu().numpy()
        
        # Save the parameters
        v_list_list[iter]   = torch.cat([v_i.data.unsqueeze(0).detach() for v_i in v_list]).cpu().numpy()
        beta_list[iter]     = mean_model.fc2.weight.T.detach().cpu().numpy()
        w_list[iter]        = w_beta.cpu().numpy()

        # Pretrain variance check
        if variance_check : 
            temp_log_lamb = torch.log(sum([torch.pow(v_i,2) for v_i in v_list]) / (m-1) + 1e-10)
            fig = plt.figure(figsize = (7, 7))
            ax = fig.add_subplot(2,2,1)
            ax.hist(temp_log_lamb[:,0].detach().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 1 (log)')
            ax = fig.add_subplot(2,2,2)
            ax.hist(temp_log_lamb[:,1].detach().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 2 (log)')
            ax = fig.add_subplot(2,2,3)
            ax.hist(temp_log_lamb[:,0].detach().exp().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 1')
            ax = fig.add_subplot(2,2,4)
            ax.hist(temp_log_lamb[:,1].detach().exp().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 2')
            plt.show()

    # Mean Pretrain end
    pretrain_end = time.time()
    print(f'Mean Pretrain spends {(pretrain_end - pretrain_start):.4f} sec')

    # Variance model Pretrain 
    print('Variance Pretrain starts')
    pretrain_start = time.time()
    mean_model.eval()
    log_phi_layer.train()
    precision.train()
    with torch.no_grad() : 
        train_y_hat = torch.zeros(train_N, K, device=device)
        train_eps = torch.zeros(train_N, K, device=device)
        for i in range(m) : 
            temp_Gamma = mean_model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device))
            train_y_hat[train_cluster[i]] = mean_model.fc2(temp_Gamma)
            train_y_hat[train_cluster[i]] += temp_Gamma @ v_list[i]
            train_eps[train_cluster[i]] = train_y[train_cluster[i]].to(device) - train_y_hat[train_cluster[i]]

    train_log_eps_sq = torch.log(torch.pow(train_eps, 2) + 1e-10).cpu()

    v_pretrain_dataset = TensorDataset(train_images, train_hps, train_log_eps_sq)
    v_pretrain_loader  = torch.utils.data.DataLoader(v_pretrain_dataset, batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(v_pretrain_epoch)) : 
        for _, (image, hp, log_eps_sq) in enumerate(v_pretrain_loader) : 
            image=image.to(device)
            hp = hp.to(device)

            mean_optimizer.zero_grad()
            variance_optimizer.zero_grad()
            log_phi_hat = log_phi_layer(mean_model.get_feature_map(image, hp).detach())
            v_pretrain_loss = F.mse_loss(log_phi_hat, log_eps_sq.to(device))
            v_pretrain_loss.backward()
            variance_optimizer.step()
            pretrain_v_loss_list.append(v_pretrain_loss.item())

        print(f'{epoch}-th epoch last batch Pretrain h-lik loss (v-step) : {v_pretrain_loss.item()}')

    print(f'Initialize Sigma_v by MMEs')
    # precision.MME_initialize(v_list)
    
    # Variance Pretrain ends
    pretrain_end = time.time()
    print(f'Variance Pretrain spends {(pretrain_end - pretrain_start):.4f} sec')


    # Save the models
    train_Gamma_list_list[0] = train_Gamma.detach().cpu().numpy()
    test_Gamma_list_list[0] = test_Gamma.detach().cpu().numpy()
    precision_list[0] = precision.recover_precision().detach().cpu().numpy()
    torch.save(mean_model.state_dict(), f'./Model/correlated_arbit_{experiment_name}_pretrained_mean_model.pt')
    torch.save(log_phi_layer.state_dict(), f'./Model/correlated_arbit_{experiment_name}_pretrained_variance_model.pt')
    torch.save(precision.state_dict(), f'./Model/hetero_arbit_{experiment_name}_pretrained_precision.pt')

    # Early stopping criterion 
    best_nhll = train_loss_list[4, pretrain_iter-1]
    best_nhll_mean_model = copy.deepcopy(mean_model)
    best_nhll_log_phi_layer = copy.deepcopy(log_phi_layer)
    # best_nhll_v_list = copy.deepcopy(v_list)
    best_nhll_precision = copy.deepcopy(precision)
    train_Gamma_list_list[2] = train_Gamma.detach().cpu().numpy()
    test_Gamma_list_list[2] = test_Gamma.detach().cpu().numpy()
    precision_list[2] = precision.recover_precision().detach().cpu().numpy()
    nhll_update_count = 0
    nhll_update_stop = False
    best_nhll_index = pretrain_iter-1

    # Re-Initialize optimizers
    if reset_opt is False : 
        mean_optimizer     = optim.Adam(list(mean_model.parameters())    + v_list,     lr=mean_lr,     betas=betas, weight_decay=weight_decay)
        variance_optimizer = optim.Adam(list(log_phi_layer.parameters()) + list(precision.parameters()), lr=variance_lr, weight_decay=weight_decay)
    
    # Main training    
    print('Main train starts')
    train_start = time.time()
    for iter in tqdm(range(max_iter)) : 
        # Early stopping
        if nhll_update_stop : 
            break 

        if reset_opt is True : 
            mean_optimizer     = optim.Adam(list(mean_model.parameters())    + v_list,     lr=mean_lr,     betas=betas, weight_decay=weight_decay)
            variance_optimizer = optim.Adam(list(log_phi_layer.parameters()) + list(precision.parameters()), lr=variance_lr, weight_decay=weight_decay)

        # M-STEP
        precision.eval()
        prec = precision.recover_precision().detach()
        for epoch in range(mean_epoch) : 
            mean_model.train()
            log_phi_layer.eval()

            for _, (image, hp, y, batch_ids) in enumerate(train_loader) : 
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)
                batch_cluster = [np.where(np.asarray(batch_ids) == idx)[0] for idx in train_ids_unique]
                batch_n_list = [len(cluster) for cluster in batch_cluster]

                batch_Gamma = mean_model.get_feature_map(image, hp)
                batch_fixed = mean_model.fc2(batch_Gamma)
                batch_random = torch.zeros_like(batch_fixed)
                for i in range (m) : 
                    batch_random[batch_cluster[i]] += batch_Gamma[batch_cluster[i]] @ v_list[i]

                with torch.no_grad() : 
                    batch_log_phi = log_phi_layer(batch_Gamma).detach()

                mean_optimizer.zero_grad()
                variance_optimizer.zero_grad()
                train_loss = nhll_correlated_corresponding(train_N, y, batch_fixed, batch_random, v_list, batch_log_phi, precision, weighted=weighted, n_list=train_n_list, batch_n_list=batch_n_list, prec=prec, update='M', verbose=verbose)
                train_loss.backward()
                mean_optimizer.step()
                train_m_loss_list.append(train_loss.item())


            mean_model.eval()
            with torch.no_grad() : 
                train_Gamma  = torch.zeros(train_N, p, device=device)
                train_fixed  = torch.zeros_like(train_y_cuda)
                train_random = torch.zeros_like(train_y_cuda)
                for i in range(m) : 
                    train_Gamma[train_cluster[i]]  = mean_model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)).detach()
                    train_fixed[train_cluster[i]]  = mean_model.fc2(train_Gamma[train_cluster[i]])
                    train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ v_list[i]
                train_Gamma_list = [train_Gamma[cluster] for cluster in train_cluster]
                train_log_phi = torch.zeros_like(train_y_cuda)
                for i in range(m) : 
                    train_log_phi[train_cluster[i]] = log_phi_layer(train_Gamma_list[i])
                train_log_phi_list = [train_log_phi[cluster] for cluster in train_cluster]

                mean_optimizer.zero_grad()
                variance_optimizer.zero_grad()
                train_loss = nhll_correlated_corresponding(train_N, train_y_cuda, train_fixed, train_random, v_list, train_log_phi, precision, 
                                                       train_Gamma_list, train_log_phi_list, update='V-full', verbose=verbose)

                train_nhll_loss_list.append(train_loss.item())
                print(f'{epoch}-th epoch full h-lik loss (M-step) : {train_loss.item()}')

        # V-STEP
        mean_model.eval()
        log_phi_layer.train()
        precision.train()

        train_Gamma  = torch.zeros(train_N, p, device=device)
        train_fixed  = torch.zeros_like(train_y_cuda)
        train_random = torch.zeros_like(train_y_cuda)
        with torch.no_grad() : 
            for i in range(m) : 
                train_Gamma[train_cluster[i]]  = mean_model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)).detach()
                train_fixed[train_cluster[i]]  = mean_model.fc2(train_Gamma[train_cluster[i]])
                train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ v_list[i]
            train_Gamma_list = [train_Gamma[cluster] for cluster in train_cluster]

        for v_iter in range(v_step_iter) : 
            train_log_phi = torch.zeros_like(train_y_cuda)
            for i in range(m) : 
                train_log_phi[train_cluster[i]] = log_phi_layer(train_Gamma_list[i])
            train_log_phi_list = [train_log_phi[cluster] for cluster in train_cluster]

            mean_optimizer.zero_grad()
            variance_optimizer.zero_grad()
            train_loss = nhll_correlated_corresponding(train_N, train_y_cuda, train_fixed, train_random, v_list, train_log_phi, precision, 
                                                   train_Gamma_list, train_log_phi_list, update='V-full', verbose=verbose)
            train_loss.backward()
            variance_optimizer.step()
            train_v_loss_list.append(train_loss.item())
            
            if v_iter % 10 == 9 : 
                train_nhll_loss_list.append(train_loss.item())
                print(f'{v_iter}-th V-step train loss : {train_loss.item()}')
            
        # Loss plot
        fig = plt.figure(figsize = (12, 4))
        ax = fig.add_subplot(1,3,1)
        ax.plot(train_m_loss_list)
        ax.set_title('M-loss plot')
        ax = fig.add_subplot(1,3,2)
        ax.plot(train_v_loss_list)
        ax.set_title('V-loss plot')
        ax = fig.add_subplot(1,3,3)
        ax.plot(train_nhll_loss_list)
        ax.set_title('NHLL-loss plot')
        ax.set_ylim(-2, 10)
        plt.show()

        # Train evaluation
        mean_model.eval()
        log_phi_layer.eval()
        precision.eval()
        with torch.no_grad() : 
            # Evaluation 1 : random effect adjustment
            '''
            Since we already computed train_Gamma, fixed and random, we do not compute those here. 
            Instead we use the above results again. 
            '''
            # train_Gamma = torch.zeros(train_N, p).to(device)
            # train_fixed = torch.zeros_like(train_y_cuda)
            # train_random = torch.zeros_like(train_y_cuda)
            # for i in range(m) : 
            #     train_Gamma[train_cluster[i]] = mean_model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)).detach()
            #     train_fixed[train_cluster[i]]  = mean_model.fc2(train_Gamma[train_cluster[i]])
            #     train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ v_list[i]

            w = LinearRegression(fit_intercept=False)
            w.fit(X=train_Gamma.cpu(), y=train_random.cpu())
            w_beta = torch.as_tensor(w.coef_).T.to(device)

            # Evaluation 2 : Train MAE / MSE / NLL / NJLL / NHLL
            # Sigma_v = precision.recover_precision()
            # train_y_list = [train_y_cuda[cluster] for cluster in train_cluster]
            train_Gamma_list = [train_Gamma[cluster] for cluster in train_cluster]
            # train_fixed_list = [train_fixed[cluster] for cluster in train_cluster]

            train_log_phi_list = [log_phi_layer(mean_model.get_feature_map(train_images[cluster].to(device), train_hps[cluster].to(device))) for cluster in train_cluster]

            train_mae = mae(train_y_cuda, train_fixed + train_random, is_3d=False, deg=deg).item()
            train_mse = F.mse_loss(train_y_cuda, train_fixed + train_random).item()
            train_nhll = nhll_correlated_corresponding(train_N, train_y_cuda, train_fixed, train_random, v_list, train_log_phi, precision, 
                                               train_Gamma_list, train_log_phi_list, update='V-full', verbose=verbose).item()
            train_loss_list[0, pretrain_iter + iter] = train_mae
            train_loss_list[1, pretrain_iter + iter] = train_mse
            train_loss_list[4, pretrain_iter + iter] = train_nhll

            print(f'{iter}-th main train train MAE, MSE, NHLL : {train_mae:.4f} deg, {train_mse:.4f}, {train_nhll:.4f}')

            # Evaluation 3 : Test MAE / MSE / NLL / NJLL / NHLL
            if large_test : 
                test_Gamma = torch.zeros(test_N, p, device=device)
                test_fixed = torch.zeros_like(test_y_cuda)
                for cluster in test_cluster : 
                    test_Gamma[cluster] = mean_model.get_feature_map(test_images[cluster].to(device), test_hps[cluster].to(device)).detach()
                    test_fixed[cluster] = mean_model.fc2(test_Gamma[cluster])
            else : 
                test_Gamma = mean_model.get_feature_map(test_images.to(device), test_hps.to(device)).detach()
                test_fixed = mean_model.fc2(test_Gamma)
            
            test_adjusted = test_Gamma @ w_beta
            if test_unseen is not True : 
                test_random = torch.zeros_like(test_fixed)
                for i in range(m) : 
                    test_random[test_cluster[i]] = test_Gamma[test_cluster[i]] @ v_list[i]

            # test_y_list        = [test_y_cuda[cluster]   for cluster in test_cluster]
            test_Gamma_list    = [test_Gamma[cluster]    for cluster in test_cluster]
            # test_fixed_list    = [test_fixed[cluster]    for cluster in test_cluster]
            # test_adjusted_list = [test_adjusted[cluster] for cluster in test_cluster]

            test_log_phi = log_phi_layer(mean_model.get_feature_map(test_images.to(device), test_hps.to(device)))
            test_log_phi_list = [test_log_phi[cluster] for cluster in test_cluster]

            if test_unseen is True : 
                test_mae = mae(test_y_cuda, test_fixed, is_3d=False, deg=deg).item()
                test_mse = F.mse_loss(test_y_cuda, test_fixed).item()
                
                test_mae_adjusted = mae(test_y_cuda, test_fixed + test_adjusted, is_3d=False, deg=deg).item()
                test_mse_adjusted = F.mse_loss(test_y_cuda, test_fixed + test_adjusted).item()
                
                test_loss_list[0, pretrain_iter + iter] = test_mae
                test_loss_list[1, pretrain_iter + iter] = test_mse

                test_loss_list[3, pretrain_iter + iter] = test_mae_adjusted
                test_loss_list[4, pretrain_iter + iter] = test_mse_adjusted
                print(f'{iter}-th main train test MAE, MSE : {test_mae:.4f} deg, {test_mse:.4f}')
                print(f'{iter}-th main train test MAE, MSE : {test_mae_adjusted:.4f} deg, {test_mse_adjusted:.4f}')

                
                prediction[pretrain_iter + iter] = convert_to_xyz(test_fixed, deg=deg).cpu().numpy()
                prediction_adjusted[pretrain_iter + iter] = convert_to_xyz(test_fixed + test_adjusted, deg=deg).cpu().numpy()
            else : 
                test_mae = mae(test_y_cuda, test_fixed + test_random, is_3d=False, deg=deg).item()
                test_mse = F.mse_loss(test_y_cuda, test_fixed + test_random).item()
                test_nhll = nhll_correlated_corresponding(test_N, test_y_cuda, test_fixed, test_random, v_list, test_log_phi, precision, test_Gamma_list, test_log_phi_list, update='V', verbose=verbose).item()
            
                test_mae_adjusted = mae(test_y_cuda, test_fixed + test_adjusted, is_3d=False, deg=deg).item()
                test_mse_adjusted = F.mse_loss(test_y_cuda, test_fixed + test_adjusted).item()
                # test_nll_adjusted = sum([multivariate_nll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], 0, torch.exp(test_log_phi_list[i]), Sigma_v) for i in range(m)]).item() / test_N
                # test_njll_adjusted = sum([multivariate_njll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], v_list[i] - w_beta, torch.exp(test_log_phi_list[i]), Sigma_v) for i in range(m)]).item() / test_N
                # test_nhll_adjusted = sum([multivariate_nhll_hetero_i(test_y_list[i], test_fixed_list[i] + test_adjusted_list[i], test_Gamma_list[i], v_list[i] - w_beta, torch.exp(test_log_phi_list[i]), Sigma_v) for i in range(m)]).item() / test_N
            
                test_loss_list[0, pretrain_iter + iter] = test_mae
                test_loss_list[1, pretrain_iter + iter] = test_mse
                test_loss_list[4, pretrain_iter + iter] = test_nhll
                test_loss_list[5, pretrain_iter + iter] = test_mae_adjusted
                test_loss_list[6, pretrain_iter + iter] = test_mse_adjusted
                print(f'{iter}-th main train test MAE, MSE, NLL, NJLL, NHLL : {test_mae:.4f} deg, {test_mse:.4f}, {test_nhll:.4f}')
                print(f'{iter}-th main train test MAE, MSE, NLL : {test_mae_adjusted:.4f} deg, {test_mse_adjusted:.4f}')
                
                prediction[pretrain_iter + iter] = convert_to_xyz(test_fixed, deg=deg).cpu().numpy()
                prediction_adjusted[pretrain_iter + iter] = convert_to_xyz(test_fixed + test_adjusted, deg=deg).cpu().numpy()


        if train_nhll < best_nhll : 
            best_nhll = train_nhll
            best_nhll_mean_model = copy.deepcopy(mean_model)
            best_nhll_log_phi_layer = copy.deepcopy(log_phi_layer)
            # best_nhll_v_list = copy.deepcopy(v_list)
            best_nhll_precision = copy.deepcopy(precision)
            train_Gamma_list_list[2] = train_Gamma.detach().cpu().numpy()
            test_Gamma_list_list[2] = test_Gamma.detach().cpu().numpy()
            precision_list[2] = precision.recover_precision().detach().cpu().numpy()
            nhll_update_count = 0
            best_nhll_index = pretrain_iter + iter

        else :
            nhll_update_count += 1

        if nhll_update_count == patience :
            nhll_update_stop = True
            print(f"Main train may be stopped at {iter-patience}th iter based on Train NHLL")

        # Save the parameters
        v_list_list[pretrain_iter + iter] = torch.cat([v_i.data.unsqueeze(0).detach() for v_i in v_list]).cpu().numpy()
        beta_list[pretrain_iter + iter] = mean_model.fc2.weight.T.detach().cpu().numpy()
        w_list[pretrain_iter + iter] = w_beta.cpu().numpy()

        # Main train Variance check
        if variance_check :
            temp_log_lamb = torch.log(sum([torch.pow(v_i,2) for v_i in v_list]) / (m-1) + 1e-10)
            fig = plt.figure(figsize = (7, 7))
            ax = fig.add_subplot(2,2,1)
            ax.hist(temp_log_lamb[:,0].detach().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 1 (log)')
            ax = fig.add_subplot(2,2,2)
            ax.hist(temp_log_lamb[:,1].detach().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 2 (log)')
            ax = fig.add_subplot(2,2,3)
            ax.hist(temp_log_lamb[:,0].detach().exp().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 1')
            ax = fig.add_subplot(2,2,4)
            ax.hist(temp_log_lamb[:,1].detach().exp().cpu().numpy(), bins=bins)
            ax.set_title('Sample variance 2')
            plt.show()

        random_intercept = np.concatenate([v_i.data[0].detach().cpu().unsqueeze(1).numpy() for v_i in v_list], axis=1)
        plt.scatter(random_intercept[0], random_intercept[1])
        plt.title('Histogram of random intercept')
        plt.show()

        train_phi = np.concatenate([np.exp(log_phi_i.detach().cpu().numpy()) for log_phi_i in train_log_phi_list])
        test_phi = np.exp(test_log_phi.detach().cpu().numpy())
        
        fig = plt.figure(figsize = (7, 7))
        ax = fig.add_subplot(2,2,1)
        ax.hist(train_phi[:,0], bins=bins)
        ax.set_title('Train phi 1')
        ax = fig.add_subplot(2,2,2)
        ax.hist(train_phi[:,1], bins=bins)
        ax.set_title('Train phi 2')
        ax = fig.add_subplot(2,2,3)
        ax.hist(test_phi[:,0], bins=bins)
        ax.set_title('Test phi 1')
        ax = fig.add_subplot(2,2,4)
        ax.hist(test_phi[:,1], bins=bins)
        ax.set_title('Test phi 2')
        plt.show()

    train_end = time.time()
    print(f'Main train spends {(train_end - train_start):.4f} sec')

    # Best model evaluation
    if test_unseen is True : 
        print(f'NHLL-selected model train MAE, MSE, NHLL : {train_loss_list[0,best_nhll_index]:.4f} deg, {train_loss_list[1,best_nhll_index]:.4f}, {train_loss_list[4,best_nhll_index]:.4f}')
        print(f'NHLL-selected model test MAE, MSE : {test_loss_list[0,best_nhll_index]:.4f} deg, {test_loss_list[1,best_nhll_index]:.4f}')
        print(f'NHLL-selected model test MAE, MSE (adjusted y_hat) : {test_loss_list[3,best_nhll_index]:.4f} deg, {test_loss_list[4,best_nhll_index]:.4f}')

    else : 
        print(f'NHLL-selected model train MAE, MSE, NHLL : {train_loss_list[0,best_nhll_index]:.4f} deg, {train_loss_list[1,best_nhll_index]:.4f}, {train_loss_list[2,best_nhll_index]:.4f}, {train_loss_list[3,best_nhll_index]:.4f}, {train_loss_list[4,best_nhll_index]:.4f}')
        print(f'NHLL-selected model test MAE, MSE, NHLL : {test_loss_list[0,best_nhll_index]:.4f} deg, {test_loss_list[1,best_nhll_index]:.4f}, {test_loss_list[4,best_nhll_index]:.4f}')
        print(f'NHLL-selected model test MAE, MSE (adjusted y_hat) : {test_loss_list[5,best_nhll_index]:.4f} deg, {test_loss_list[6,best_nhll_index]:.4f}')

    train_Gamma_list_list[1] = train_Gamma.detach().cpu().numpy()
    test_Gamma_list_list[1] = test_Gamma.detach().cpu().numpy()
    precision_list[1] = precision.recover_precision().detach().cpu().numpy()

    np.save(f'./Prediction/correlated_arbit_{experiment_name}_pred', prediction)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_pred_adjusted', prediction_adjusted)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_train_loss', train_loss_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_test_loss', test_loss_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_v_list', v_list_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_precision', precision_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_beta', beta_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_w', w_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_train_Gamma', train_Gamma_list_list)
    np.save(f'./Prediction/correlated_arbit_{experiment_name}_test_Gamma', test_Gamma_list_list)

    torch.save(mean_model.state_dict(), f'./Model/correlated_arbit_{experiment_name}_trained_mean_model.pt')
    torch.save(best_nhll_mean_model.state_dict(), f'./Model/correlated_arbit_{experiment_name}_nhll_selected_mean_model.pt')
    
    torch.save(log_phi_layer.state_dict(), f'./Model/correlated_arbit_{experiment_name}_trained_variance_model.pt')
    torch.save(best_nhll_log_phi_layer.state_dict(), f'./Model/correlated_arbit{experiment_name}_nhll_selected_variance_model.pt')
    
    torch.save(precision.state_dict(), f'./Model/correlated_arbit{experiment_name}_trained_precision.pt')
    torch.save(best_nhll_precision.state_dict(), f'./Model/correlated_arbit_{experiment_name}_nhll_selected_precision.pt')
    

    return pretrain_m_loss_list, pretrain_v_loss_list, train_m_loss_list, train_v_loss_list, train_nhll_loss_list

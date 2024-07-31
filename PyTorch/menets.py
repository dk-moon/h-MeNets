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

from likelihood import multivariate_njll_i, multivariate_nll_i, multivariate_nhll_i, EM_update
from util import make_reproducibility, TensorDataset, convert_to_spherical, convert_to_xyz, mae

def MeNets(
        train_ids, train_images, train_hps, train_gazes, 
        test_ids, test_images, test_hps, test_gazes, 
        network, hidden_features=500, K=2, 
        MAXITER=320000, snapshot=300, batch_size=1000, 
        base_lr=0.1, weight_decay=0.0005, momentum=0.9, power=1.0, 
        max_iter=10, patience=3, 
        device=torch.device('cpu'), SEED=None, experiment_number = 1, 
        deg=False, test_unseen=True, verbose=True): 
    '''
    Python implementation of MeNets. 
    Hyperparameters are selected according to the original MeNets code (implemented via MATLAB and Caffe). 

    Network architecture : ResNet-18

    batch_size = 1000
    MAXITER : 320000

    Caffe's default optimizer : SGD
    base_lr = 0.1
    weight_decay = 0.0005
    momentum = 0.9
    power = 1.0
    lr scheduler type : 'poly', that is,  lr_i = base_lr * (1-i/MAXITER)*power
    '''
    torch.cuda.empty_cache()
    if SEED is not None : 
        make_reproducibility(SEED)

    train_ids_unique = np.unique(train_ids)
    m = len(train_ids_unique)
    train_hps = convert_to_spherical(train_hps, deg=deg).float()
    train_y = convert_to_spherical(train_gazes, deg=deg).float()
    train_y_cuda = train_y.to(device)
    train_N = len(train_gazes)
    train_cluster = [np.where(train_ids == idx)[0] for idx in train_ids_unique]
    train_n_list = [len(cluster) for cluster in train_cluster]
    train_dataset = TensorDataset(train_images, train_hps, train_y, train_ids)

    test_hps = convert_to_spherical(test_hps, deg=deg).float()
    test_gazes = test_gazes.float()
    test_y = convert_to_spherical(test_gazes, deg=deg).float()
    test_y_cuda = test_y.to(device)
    test_N = len(test_gazes)
    if test_unseen is True : 
        test_cluster = [np.arange(test_N)]
    else : 
        test_cluster = [np.where(test_ids == idx)[0] for idx in train_ids_unique]
        # test_n_list = [len(cluster) for cluster in test_cluster]

    train_nll_list = []
    train_njll_list = []
    train_nhll_list = []

    test_nll_list = []
    test_nll_adjusted_list = []

    train_loss_list = np.zeros((5, 100 * max_iter))
    test_loss_list = np.zeros((10, 100 * max_iter))

    # Main part
    print(f"EM algorithm starts")

    # Initialize neural networks
    model = network(hidden_features=hidden_features, out_features=K).to(device)
    p = model.p

    # Initialize other parameters
    v_list = [torch.zeros(p, K) for _ in range(m)]
    sigma_sq = torch.ones(K)
    Sigma_v = torch.eye(p).repeat(K,1,1)

    # Setup for early stopping
    best_njll = 1e8
    best_model = copy.deepcopy(model)
    best_v_list = copy.deepcopy(v_list)
    best_sigma_sq = copy.deepcopy(sigma_sq)
    best_Sigma_v = copy.deepcopy(Sigma_v)
    update_count = 0
    update_stop = False

    # Initialize fixed parts of responses
    train_y_fixed = train_y

    for iter in tqdm(range(max_iter)) : 
        if update_stop : 
            break

        opt = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda = lambda i: (1-i/MAXITER) ** power)
        train_dataset = TensorDataset(train_images, train_hps, train_y_fixed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training : SGD step
        num_iter = 0
        epoch = 0
        exceed_MAXITER = False
        while exceed_MAXITER is not True : 
            model.train()
            for _, (image, hp, y) in enumerate(train_loader) :
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)

                opt.zero_grad()
                train_loss = F.mse_loss(model(image, hp), y)
                train_loss.backward()
                opt.step()
                scheduler.step()
                num_iter += 1
                if num_iter >= MAXITER : 
                    exceed_MAXITER=True
                    break

            epoch += 1

            # Evaluation
            if epoch % snapshot == 0 : 
                model.eval()
                test_Gamma = model.get_feature_map(test_images.to(device), test_hps.to(device))
                if test_unseen is True : 
                    y_hat = model.fc2(test_Gamma).cpu()

                    test_mae = mae(test_y, y_hat, is_3d=False, deg=deg).item()
                    test_nll = multivariate_nll_i(test_y, y_hat, test_Gamma.cpu(), v_i=0, sigma_sq=sigma_sq, Sigma_v=Sigma_v).item() / test_N
                    test_nll_list.append(test_nll)
                    print(f'{epoch}-th epoch test MAE, NLL : {test_mae:.4f} deg, {test_nll:.4f}')  
                else : 
                    y_fixed = model.fc2(test_Gamma)
                    y_random = torch.zeros_like(y_fixed)
                    for i in range(m) : 
                        y_random[test_cluster[i]] = test_Gamma[test_cluster[i]] @ v_list[i]

                    test_mae = mae(test_y_cuda, y_fixed + y_random, is_3d=False, deg=deg).item()

                    test_y_list = [test_y_cuda[cluster] for cluster in test_cluster]
                    test_fixed_list = [y_fixed[cluster] for cluster in test_cluster]
                    test_Gamma_list = [test_Gamma[cluster] for cluster in test_cluster]
            
                    test_nhll = sum([multivariate_nhll_i(test_y_list[i], test_fixed_list[i], test_Gamma_list[i], v_list[i].to(device), sigma_sq.to(device), Sigma_v.to(device)) for i in range(m)]).item() / test_N
                    print(f'{epoch}-th epoch test MAE, NHLL : {test_mae:.4f} deg, {test_nhll:.4f}')  

        # Training : M-step
        model.eval()
        image = train_images.to(device)
        hp = train_hps.to(device)

        with torch.no_grad() : 
            beta = model.fc2.weight.data.clone().T.cpu()
            train_y_list = [train_y[cluster] for cluster in train_cluster]
            train_Gamma_list = [model.get_feature_map(image[cluster], hp[cluster]).detach().cpu() for cluster in train_cluster]
            train_f_hat_list = [train_Gamma_list[i] @ beta for i in range(m)]

            # Update other parameters
            v_list, sigma_sq, Sigma_v = EM_update(train_y_list, train_Gamma_list, beta, v_list, sigma_sq, Sigma_v, train_n_list, use_woodbury=False)

            # Update fixed parts of responses
            train_y_fixed = torch.zeros_like(train_y)
            for i in range(m-1) : 
                train_y_fixed[train_cluster[i]] = train_y[train_cluster[i]].detach() - train_Gamma_list[i] @ (beta + v_list[i])
            
            # Train set evaluation
            train_nll = sum([multivariate_nll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
            train_njll = sum([multivariate_njll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
            train_nhll = sum([multivariate_nhll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
            train_nll_list.append(train_nll)
            train_njll_list.append(train_nhll)
            train_nhll_list.append(train_njll)
            print(f'{iter}-th iter train NLL, NJLL, NHLL : {train_nll:.4f}, {train_njll:.4f}, {train_nhll:.4f}')

            # Random effect adjustment

            # Evaluation
            test_pred = convert_to_xyz(model(test_images.to(device), test_hps.to(device)).detach().cpu())
            test_error = mae(test_pred, test_gazes)
            print(f'{iter}-th iter test MAE, sigma_sq : {test_error:.4f} deg / {sigma_sq}')
            
            # Update the best model if validation NJLL is improved
            if train_njll < best_njll :
                best_njll = train_njll
                best_model = copy.deepcopy(model)
                best_v_list = copy.deepcopy(v_list)
                best_sigma_sq = copy.deepcopy(sigma_sq)
                best_Sigma_v = copy.deepcopy(Sigma_v)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience : 
                update_stop = True
                print(f"EM algorithm stopped training at {1+iter-patience}th iteration")

    # Post-hoc training process
    best_model.eval()
    with torch.no_grad() : 
        train_Gamma = torch.zeros(len(train_y), p)
        train_random = torch.zeros_like(train_y)
        for i in range(m-1) : 
            train_Gamma[train_cluster[i]] = best_model.get_feature_map(
                train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)
            ).detach().cpu()
            train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ best_v_list[i]

        w1 = LinearRegression(fit_intercept=False)
        w1.fit(X=train_Gamma, y=train_random)
        w1_beta = torch.as_tensor(w1.coef_).T.to(device)

    # Test step
    with torch.no_grad() : 
        image = test_images.to(device)
        hp = test_hps.to(device)
        test_Gamma = best_model.get_feature_map(image, hp)

        f_hat = best_model.fc2(test_Gamma).cpu()
        f_hat_adjusted1 = (best_model.fc2(test_Gamma) + test_Gamma @ w1_beta).cpu()
        test_nll = multivariate_nll_i(test_y, f_hat, test_Gamma.cpu(), v_i=0, sigma_sq=best_sigma_sq, Sigma_v=best_Sigma_v).item() / test_N
        test_nll_adjusted1 = multivariate_nll_i(test_y, f_hat_adjusted1, test_Gamma.cpu(), v_i=0, sigma_sq=best_sigma_sq, Sigma_v=best_Sigma_v).item() / test_N
        test_nll_list.append(test_nll)
        test_nll_adjusted_list.append(test_nll_adjusted1)
        print(f'Leave-{looid} out test nll / adjusted test nll : {test_nll:.4f}, {test_nll_adjusted1:.4f}')  

        test_pred = convert_to_xyz(f_hat)
        test_pred_adjusted = convert_to_xyz(f_hat_adjusted1)
            
    test_error = mae(test_pred, test_gazes)
    print(f'Leave-{looid}-out MAE without adjustment : {test_error:.4f} deg')
    test_error_adjusted = mae(test_pred_adjusted, test_gazes)
    print(f'Leave-{looid}-out MAE with adjustment : {test_error_adjusted:.4f} deg')

    np.savetxt(f'Prediction/exp_{experiment_number}_MeNet_{looid}.csv', test_pred.numpy())
    np.savetxt(f'Prediction/exp_{experiment_number}_MeNets_adjustment_{looid}.csv', test_pred_adjusted.numpy())
    return test_pred, test_pred_adjusted, train_nll_list, train_njll_list, train_nhll_list, test_nll_list, test_nll_adjusted_list






# def loocv_MeNets_rev2(
#         network, id_list, image_list, hp_list, gaze_list, hidden_features=500, K=2, m=15, 
#         max_iter=20, patience=5, 
#         device=torch.device('cpu'), SEED=10, use_woodbury = False, experiment_number = 1) : 
    
#     batch_size = 1000
#     weight_decay = 0.0005
#     momentum = 0.9
#     MAXITER = 320000
#     base_lr = 0.1
#     power = 1.0

#     make_reproducibility(SEED)

#     prediction = torch.zeros(len(np.concatenate(id_list)), 3, dtype=torch.float32)
#     prediction_adjusted = torch.zeros(len(np.concatenate(id_list)), 3, dtype=torch.float32)

#     train_nll_list = [[] for _ in range(m)]
#     train_njll_list = [[] for _ in range(m)]
#     train_nhll_list = [[] for _ in range(m)]
#     train_nhll_diag_list = [[] for _ in range(m)]

#     test_nll_list = [[] for _ in range(m)]
#     test_nll_adjusted_list = [[] for _ in range(m)]

#     for looid in range(m) : 
#         print(f"Leave-{looid}-out EM algorithm starts")
#         torch.cuda.empty_cache()

#         full_ids = np.concatenate(id_list)
#         test_indice = np.where(full_ids == id_list[looid][0])[0]

#         train_ids = np.concatenate(id_list[:looid] + id_list[(looid + 1):])
#         train_images = torch.cat(image_list[:looid] + image_list[(looid + 1):]).float()
#         train_hps = convert_to_spherical(torch.cat(hp_list[:looid] + hp_list[(looid + 1):])).float()
#         train_y = convert_to_spherical(torch.cat(gaze_list[:looid] + gaze_list[(looid + 1):])).float()
#         train_cluster = [np.where(train_ids == id)[0] for id in np.unique(train_ids)]
#         train_n_list = [len(cluster) for cluster in train_cluster]
#         train_N = len(train_ids)

#         # Test set
#         test_ids = id_list[looid]
#         test_images = image_list[looid].float()
#         test_hps = convert_to_spherical(hp_list[looid]).float()
#         test_gazes = gaze_list[looid].float()
#         test_y = convert_to_spherical(test_gazes)
#         # test_cluster and test_n_list do not make sense, as in LOOCV, test observations are from new subject. 
#         test_N = len(test_ids)

#         # Initialize neural networks
#         model = network(hidden_features=hidden_features, out_features=K).to(device)
#         opt = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda = lambda iter: (1-iter/MAXITER) ** power)
#         p = model.p

#         # Initialize other parameters
#         v_list = [torch.zeros(p, K) for _ in range(m-1)]
#         sigma_sq = torch.ones(K)
#         Sigma_v = torch.eye(p).repeat(K,1,1)

#         # Setup for early stopping
#         best_njll = 1e8
#         best_model = copy.deepcopy(model)
#         best_v_list = copy.deepcopy(v_list)
#         best_sigma_sq = copy.deepcopy(sigma_sq)
#         best_Sigma_v = copy.deepcopy(Sigma_v)
#         update_count = 0
#         update_stop = False

#         # Initialize fixed parts of responses
#         # In the first iteration, y_fixed = y_fixed as initial random effects are all zeros
#         train_y_fixed = train_y
#         for iter in tqdm(range(max_iter)) : 
#             if update_stop : 
#                 break

#             train_dataset = TensorDataset(train_images, train_hps, train_y_fixed)
#             train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#             # Training : SGD step
#             num_iter = 0
#             epoch=0
#             exceed_MAXITER = False
#             while exceed_MAXITER is not True : 
#                 model.train()
#                 for _, (image, hp, y) in enumerate(train_loader) :
#                     image = image.to(device)
#                     hp = hp.to(device)
#                     y = y.to(device)

#                     opt.zero_grad()
#                     train_loss = F.mse_loss(model(image, hp), y)
#                     train_loss.backward()
#                     opt.step()
#                     scheduler.step()
#                     num_iter += 1
#                     if num_iter >= MAXITER : 
#                         exceed_MAXITER=True
#                         break
#                 epoch += 1
#                 if epoch % 50 == 0 : 
#                     print(f'{epoch}-th epoch ended')
#             # Training : M-step
#             model.eval()
#             image = train_images.to(device)
#             hp = train_hps.to(device)

#             with torch.no_grad() : 
#                 beta = model.fc2.weight.data.clone().T.cpu()
#                 train_y_list = [train_y[cluster] for cluster in train_cluster]
#                 train_Gamma_list = [model.get_feature_map(image[cluster], hp[cluster]).detach().cpu() for cluster in train_cluster]
#                 train_f_hat_list = [train_Gamma_list[i] @ beta for i in range(m-1)]

#                 # Update other parameters
#                 v_list, sigma_sq, Sigma_v = EM_update(train_y_list, train_Gamma_list, beta, v_list, sigma_sq, Sigma_v, train_n_list, use_woodbury)

#                 # Update fixed parts of responses
#                 train_y_fixed = torch.clone(train_y)
#                 for i in range(m-1) : 
#                     train_y_fixed[train_cluster[i]] -= train_Gamma_list[i] @ (beta + v_list[i])

#                 # Compute various negative log-likelihoods in train set
#                 train_nll = sum([multivariate_nll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
#                 train_njll = sum([multivariate_njll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
#                 train_nhll = sum([multivariate_nhll_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
#                 train_nhll_diag = sum([multivariate_nhll_diag_i(train_y_list[i], train_f_hat_list[i], train_Gamma_list[i], v_list[i], sigma_sq, Sigma_v) for i in range(m-1)]).item() / train_N
#                 train_nll_list[looid].append(train_nll)
#                 train_njll_list[looid].append(train_nhll)
#                 train_nhll_list[looid].append(train_njll)
#                 train_nhll_diag_list[looid].append(train_nhll_diag)
#                 print(f'Leave-{looid} out {iter}th iteration train nll, njll, nhll, nhll(diag) : {train_nll:.4f}, {train_njll:.4f}, {train_nhll:.4f}, {train_nhll_diag:.4f}')

                
#                 # Update the best model if validation NJLL is improved
#                 if train_njll < best_njll :
#                     best_njll = train_njll
#                     best_model = copy.deepcopy(model)
#                     best_v_list = copy.deepcopy(v_list)
#                     best_sigma_sq = copy.deepcopy(sigma_sq)
#                     best_Sigma_v = copy.deepcopy(Sigma_v)
#                     update_count = 0
#                 else :
#                     update_count += 1

#                 if update_count == patience :
#                     update_stop = True
#                     print(f"Leave-{looid}-out CV stopped training at {1+iter-patience}th iteration")


#         best_model.eval()
#         # Post-hoc training process
#         with torch.no_grad() : 
#             train_Gamma = torch.zeros(len(train_y), p)
#             train_random = torch.zeros_like(train_y)
#             for i in range(m-1) : 
#                 train_Gamma[train_cluster[i]] = best_model.get_feature_map(
#                     train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)
#                 ).detach().cpu()
#                 train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ best_v_list[i]

#         # w1 : Linear regressiom
#         w1 = LinearRegression(fit_intercept=False)
#         w1.fit(X=train_Gamma, y=train_random)
#         w1_beta = torch.as_tensor(w1.coef_).T.to(device)

#         # Test step
#         with torch.no_grad() : 
#             image = test_images.to(device)
#             hp = test_hps.to(device)
#             test_Gamma = best_model.get_feature_map(image, hp)

#             f_hat = best_model.fc2(test_Gamma).cpu()
#             f_hat_adjusted1 = (best_model.fc2(test_Gamma) + test_Gamma @ w1_beta).cpu()
#             test_nll = multivariate_nll_i(test_y, f_hat, test_Gamma.cpu(), v_i=0, sigma_sq=best_sigma_sq, Sigma_v=best_Sigma_v).item() / test_N
#             test_nll_adjusted1 = multivariate_nll_i(test_y, f_hat_adjusted1, test_Gamma.cpu(), v_i=0, sigma_sq=best_sigma_sq, Sigma_v=best_Sigma_v).item() / test_N
#             test_nll_list[looid].append(test_nll)
#             test_nll_adjusted_list[looid].append(test_nll_adjusted1)
#             print(f'Leave-{looid} out test nll / adjusted test nll : {test_nll:.4f}, {test_nll_adjusted1:.4f}')  

#             test_pred = convert_to_xyz(f_hat)
#             test_pred_adjusted = convert_to_xyz(f_hat_adjusted1)
                
#         prediction[test_indice] = test_pred
#         prediction_adjusted[test_indice] = test_pred_adjusted

#         test_error = mae(test_pred, test_gazes)
#         print(f'Leave-{looid}-out MAE without adjustment : {test_error:.4f} deg')
#         test_error_adjusted = mae(test_pred_adjusted, test_gazes)
#         print(f'Leave-{looid}-out MAE with adjustment (lin. reg) : {test_error_adjusted:.4f} deg')

#     error = mae(prediction, torch.cat(gaze_list))
#     error_adjusted = mae(prediction_adjusted, torch.cat(gaze_list))

#     print(f'MeNets\' MAE without adjustment ({model.model_name}) : {error:.4f} deg')
#     print(f'MeNets\' MAE with adjustment (lin. reg) ({model.model_name}) : {error_adjusted:.4f} deg')

#     np.savetxt(f'Prediction/exp_{experiment_number}_Reproduce_MeNets.csv', prediction.numpy())
#     np.savetxt(f'Prediction/exp_{experiment_number}_Reproduce_MeNets_adjustment.csv', prediction_adjusted.numpy())
#     return [prediction, prediction_adjusted,  
#             train_nll_list, train_njll_list, train_nhll_list, train_nhll_diag_list, 
#             test_nll_list, test_nll_adjusted_list]

import copy
import torch
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from util import make_reproducibility, TensorDataset, convert_to_spherical, convert_to_xyz, mae, make_arbitrary_masking, k_fold_index

def within_MPIIGaze(network, ids, images, hps, gazes, hidden_features=500, K=2, m=15, num_fold=10, 
                   lr=1e-3, weight_decay=1e-3, batch_size=256, max_epoch=100, patience=15, 
                   device=torch.device('cpu'), SEED=10, experiment_number=1) : 
    
    total_N = len(ids)

    prediction = torch.zeros(total_N, 3, dtype=torch.float32)

    train_loss_list = [[] for _ in range(num_fold)]
    val_loss_list = [[] for _ in range(num_fold)]

    cv_indice_list = k_fold_index(N=total_N, k=num_fold, randomize=True, SEED=SEED)


    for fold in range(num_fold) : 
        print(f'{fold + 1}-th {num_fold}-fold CV starts')
        torch.cuda.empty_cache()

        loo_indice = cv_indice_list[fold][0]
        test_indice = cv_indice_list[fold][1]

        # hps : (translated) hps (2-dimensional)
        # y : translated gazes (2-dimensional)
        # gazes : non-translated gazes (3-dimensional)
        loo_ids = ids[loo_indice]
        loo_images = images[loo_indice].float()
        loo_hps = convert_to_spherical(hps[loo_indice]).float()
        loo_y = convert_to_spherical(gazes[loo_indice]).float()
        loo_N = len(loo_ids)

        train_indice, val_indice = train_test_split(range(loo_N), test_size = 0.1)

        train_images = loo_images[train_indice]
        train_hps = loo_hps[train_indice]
        train_y = loo_y[train_indice]

        train_dataset = TensorDataset(train_images, train_hps, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_images = loo_images[val_indice]
        val_hps = loo_hps[val_indice]
        val_y = loo_y[val_indice]
        
        test_images = images[test_indice].float()
        test_hps = convert_to_spherical(hps[test_indice]).float()
        test_gazes = gazes[test_indice].float()

        model = network(hidden_features=hidden_features, out_features=K).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss = nn.MSELoss().to(device)

        # For early stopping
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(max_epoch)) : 
            if update_stop : 
                break
            
            # training step
            model.train()
            for _, (image, hp, y) in enumerate(train_loader) :
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)

                opt.zero_grad()
                train_loss = loss(model(image, hp), y)
                train_loss.backward()
                opt.step()
                train_loss_list[fold].append(train_loss.item())

            with torch.no_grad() : 
                # validation step
                model.eval()
                image = val_images.to(device)
                hp = val_hps.to(device)
                y = val_y.to(device)

                val_loss = loss(model(image, hp), y).item()
                val_loss_list.append(val_loss)

                if val_loss < best_loss :
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                    update_count = 0
                else :
                    update_count += 1

                if update_count == patience :
                    update_stop = True
                    print(f"{fold + 1}-th fold stopped training at {1+epoch-patience}th epoch")
                
        # test step
        with torch.no_grad() : 
            best_model.eval()
            image = test_images.to(device)
            hp = test_hps.to(device)

            pred = best_model(image, hp)

        # save only last prediction
        prediction[test_indice] = convert_to_xyz(pred.cpu())
        test_error = mae(prediction[test_indice], test_gazes)
        print(f'MPIIGaze\'s {fold + 1}-th fold MAE : {test_error} deg')

    error = mae(prediction, gazes)
    print(f'MPIIGaze\'s MAE ({model.model_name}) : {error} deg')
    np.savetxt(f"Prediction/within_MPIIGaze_{model.model_name}_{experiment_number}.csv", prediction.numpy(), delimiter=",")
    return prediction, error
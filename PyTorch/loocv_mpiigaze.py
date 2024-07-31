import copy
import torch
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from util import make_reproducibility, TensorDataset, convert_to_spherical, convert_to_xyz, mae

def loocv_MPIIGaze(network, id_list, image_list, hp_list, gaze_list, hidden_features=500, K=2, m=15, 
                   lr=1e-3, weight_decay=1e-3, batch_size=256, max_epoch=100, patience=15, 
                   device=torch.device('cpu'), SEED=10, experiment_number=1) : 
    make_reproducibility(SEED)

    prediction = torch.zeros_like(torch.cat(gaze_list), dtype=torch.float32)

    train_loss_list = [[] for _ in range(m)]
    val_loss_list = [[] for _ in range(m)]

    for looid in range(m) : 
        torch.cuda.empty_cache()

        full_ids = np.concatenate(id_list)
        test_indice = np.where(full_ids == id_list[looid][0])[0]

        loo_ids = np.concatenate(id_list[:looid] + id_list[(looid + 1):])
        loo_images = torch.cat(image_list[:looid] + image_list[(looid + 1):])
        loo_hps = convert_to_spherical(torch.cat(hp_list[:looid] + hp_list[(looid + 1):]))
        loo_y = convert_to_spherical(torch.cat(gaze_list[:looid] + gaze_list[(looid + 1):]))
        # loo_gazes = torch.cat(gaze_list[:looid] + gaze_list[(looid + 1):])

        N = len(loo_ids)
        train_indice, val_indice = train_test_split(range(N), test_size = 0.1)

        # train_ids = loo_ids[train_indice]
        train_images = loo_images[train_indice]
        train_hps = loo_hps[train_indice]
        # train_gazes = loo_gazes[train_indice]
        train_y = loo_y(train_indice)
        # train_dataset = TensorDataset(train_images, train_hps, train_gazes)
        train_dataset = TensorDataset(train_images, train_hps, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # val_ids = loo_ids[val_indice]
        val_images = loo_images[val_indice]
        val_hps = loo_hps[val_indice]
        # val_gazes = loo_gazes[val_indice]
        val_y = loo_y(val_indice)

        # test_ids = id_list[looid]
        test_images = image_list[looid]
        test_hps = convert_to_spherical(hp_list[looid])
        test_gazes = gaze_list[looid]

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
                train_loss_list[looid].append(train_loss.item())

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
                    print(f"Leave-{looid}-out CV stopped training at {1+epoch-patience}th epoch")
                
                # test step
                best_model.eval()
                image = test_images.to(device)
                hp = test_hps.to(device)

                pred = best_model(image, y)

        # save only last prediction
        prediction[test_indice] = convert_to_xyz(pred.cpu())    

    error = mae(prediction, torch.cat(gaze_list))
    print(f'MPIIGaze\'s MAE ({model.model_name}) : {error} deg')
    np.savetxt(f"Prediction/MPIIGaze_{model.model_name}_{experiment_number}.csv", prediction.numpy(), delimiter=",")
    return prediction, error
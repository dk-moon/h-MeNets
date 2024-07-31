import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm

from util import make_reproducibility, TensorDataset, convert_to_spherical, convert_to_xyz, mae

def MeNets_simple(
        train_ids, train_images, train_hps, train_gazes, 
        test_ids, test_images, test_hps, test_gazes, 
        network, hidden_features=500, K=2, 
        MAXITER=320000, snapshot=300, batch_size=1000, 
        base_lr=0.1, weight_decay=0.0005, momentum=0.9, power=1.0, 
        max_iter=1, patience=1, 
        device=torch.device('cpu'), SEED=None, experiment_name = 1, 
        deg=False, normalize=False): 
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
    
    if normalize : 
        train_images /= 255.0
        test_images /= 255.0
        
    train_hps = convert_to_spherical(train_hps, deg=deg).float()
    train_y = convert_to_spherical(train_gazes, deg=deg).float()
    
    test_hps = convert_to_spherical(test_hps, deg=deg).float()
    test_gazes = test_gazes.float()

    train_loss_list = []

    # Main part
    print(f"EM algorithm starts")

    # Initialize neural networks
    model = network(hidden_features=hidden_features, out_features=K).to(device)

    # Initialize fixed parts of responses
    train_y_fixed = train_y
    for _ in range(max_iter) : 

        opt = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda = lambda i: (1-i/MAXITER) ** power)
        train_dataset = TensorDataset(train_images, train_hps, train_y_fixed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training : SGD step
        num_iter = 0
        exceed_MAXITER = False
        for epoch in tqdm(range(10000)) : 
            if exceed_MAXITER : 
                break

            model.train()
            for _, (image, hp, y) in enumerate(train_loader) :
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)

                opt.zero_grad()
                train_loss = F.mse_loss(y, model(image, hp))
                train_loss.backward()
                train_loss_list.append(train_loss.item())
                opt.step()
                scheduler.step()
                num_iter += 1
                if num_iter >= MAXITER : 
                    exceed_MAXITER=True
                    break

            if epoch % snapshot == 0 : 
                current_lr = opt.param_groups[0]['lr']
                print(f'Current learning rate : {current_lr}')
                print(f'Last batch mse loss : {train_loss.item()}')

    torch.save(model.state_dict(), f'./Model/MeNets_{experiment_name}_trained_model.pt')
    return model, train_loss_list






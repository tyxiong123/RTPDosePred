import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext import create_mednext_v1
import data_loader_lightning
import yaml
import argparse 
import os
from Loss import L1_DVH_Loss
import numpy as np
import time
import copy


def cal_valid_loss(cfig, model, val_loader, device):  #calculate validation loss
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, data_dict in enumerate(val_loader):
            outputs = model(data_dict['data'].to(device))
            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone()) 
            outputs = outputs * cfig['scale_out']
            loss = L1_DVH_Loss(outputs, data_dict['label'].to(device),data_dict['PTV'].to(device),data_dict['oar_serial'].to(device),data_dict['oar_parallel'].to(device),device)

            val_loss += loss.item()
        val_loss /= len(val_loader)
    model.train()

    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('cfig_path',  type = str)
    parser.add_argument('--phase', default = 'train', type = str)
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device( "cuda:2")
    # ------------ data loader -----------------#
    loaders = data_loader_lightning.GetLoader(cfig = cfig['loader_params'])
    train_loader =loaders.train_dataloader()
    val_loader = loaders.train_val_dataloader()

    # ------------- Network ------------------ # 
    # model = AGUDPnet( in_channels = cfig['model_params']['num_input_channels'],
    # filter_num_list= cfig['model_params']['filter_num_list'],
    # out_channels = cfig['model_params']['out_channels'], 
    # ).to(device)

    model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
    num_classes = cfig['model_params']['out_channels'],
    model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
    kernel_size = cfig['model_params']['kernel_size'],   # 3x3x3 and 5x5x5 were tested in publication
    deep_supervision = cfig['model_params']['deep_supervision']   
    ).to(device)

    # ------------ loss -----------------------# 
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr':cfig['lr']}], lr=cfig['lr'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= len(train_loader) * cfig['num_epochs'], last_epoch=cfig['num_epochs'])

    #criterion = nn.L1Loss()

    # -----------Training loop --------------- #

    nbatch_per_log = max(int(len(train_loader) / 20), 1)  

    if not os.path.exists(cfig['save_model_root']):
        os.makedirs(cfig['save_model_root'])

    top_models = []
    top_valid_models = []
    time_start = time.time()
    for epoch in range(cfig['num_epochs']):
        model.train()
        epoch_loss = 0
        for batch_idx, data_dict in enumerate(train_loader):
            # Forward pass
            outputs = model(data_dict['data'].to(device))

            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone())  
                
            outputs = outputs * cfig['scale_out']

            loss = L1_DVH_Loss(outputs, data_dict['label'].to(device),data_dict['PTV'].to(device),data_dict['oar_serial'].to(device),data_dict['oar_parallel'].to(device),device)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()

            epoch_loss += loss.item()
            
            if batch_idx % nbatch_per_log == 0:
            
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{cfig['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

            if batch_idx == 3 and epoch % 50 == 0:
                os.system('nvidia-smi')
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfig['num_epochs']}] Completed: Avg Loss: {avg_epoch_loss:.4f}")

        model_save_path = os.path.join(cfig['save_model_root'], 'last_model.pth')
        torch.save(model.state_dict(), model_save_path)

        if len(top_models) < 3:
            top_models.append((avg_epoch_loss, copy.deepcopy(model.state_dict())))
        else:
            # Find the model with the highest loss in the top 3
            max_loss, _ = max(top_models, key=lambda x: x[0])
            if avg_epoch_loss < max_loss:
                # Replace the model with the highest loss
                top_models.remove(max(top_models, key=lambda x: x[0]))
                top_models.append((avg_epoch_loss,  copy.deepcopy(model.state_dict())))
        # Sort the list by loss value
        top_models.sort(key=lambda x: x[0])
        
        # validation loss
        val_loss = cal_valid_loss(cfig, model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")

        # save top 3 models on the validation dataset
        if len(top_valid_models) < 3:
            top_valid_models.append((val_loss, copy.deepcopy(model.state_dict())))
        else:
            # Find the model with the highest loss in the top 3
            max_loss, _ = max(top_valid_models, key=lambda x: x[0])
            if val_loss < max_loss:
                # Replace the model with the highest loss
                top_valid_models.remove(max(top_valid_models, key=lambda x: x[0]))
                top_valid_models.append((val_loss, copy.deepcopy(model.state_dict())))

        for i, (loss, state_dict) in enumerate(top_models):
            model_save_path = os.path.join(cfig['save_model_root'], f'top_model_{i+1}.pth')
            torch.save(state_dict, model_save_path)
        
        for i, (loss, state_dict) in enumerate(top_valid_models):
            model_save_path = os.path.join(cfig['save_model_root'], f'top_valid_model_{i+1}.pth')
            torch.save(state_dict, model_save_path)

    time_end = time.time()
    print(f"Training completed in {time_end - time_start:.2f} seconds")
    # # Optionally save the top 5 models to disk
    # for i, (loss, state_dict) in enumerate(top_models):
    #     model_save_path = os.path.join(cfig['save_model_root'], f'top_model_{i+1}.pth')
    #     torch.save(state_dict, model_save_path)
    
    # for i, (loss, state_dict) in enumerate(top_valid_models):
    #     model_save_path = os.path.join(cfig['save_model_root'], f'top_valid_model_{i+1}.pth')
    #     torch.save(state_dict, model_save_path)
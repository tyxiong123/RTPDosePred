import torch
from nnunet_mednext import create_mednext_v1
import data_loader_lightning
import yaml
import argparse 
import os
import pdb
import numpy as np




def offset_spatial_crop(roi_center=None, roi_size=None):
    """
    for crop spatial regions of the data based on the specified `roi_center` and `roi_size`.
    
    get the start and end of the crop
    
    Parameters:
        roi_center (tuple of int, optional): The center point of the region of interest (ROI).
        roi_size (tuple of int, optional): The size of the ROI in each spatial dimension.
        
    Returns:
        start & end: start and end offsets
    """
    
    if roi_center is None or roi_size is None:
        raise ValueError("Both `roi_center` and `roi_size` must be specified.")
    
    roi_center = [int(round(c)) for c in roi_center]
    roi_size = [int(round(s)) for s in roi_size]
    
    start = []
    end = []
    
    for i, (center, size) in enumerate(zip(roi_center, roi_size)):
        
        half_size = size // 2 # int(round(size / 2))
        start_i = max(center - half_size, 0)  # Ensure we don't go below 0
        end_i = max(start_i + size, start_i)
        #end_i = min(center + half_size + (size % 2), ori_size[i])  
        start.append(start_i)
        end.append(end_i)

    return start, end

def cropped2ori(crop_data, ori_size, isocenter, trans_in_size):

    '''
    crop_data: the cropped data
    ori_size: the original size of the data
    isocenter: the isocenter of the original data
    trans_in_size: the in_size parameter in the transfromation of loader
    '''

    assert (np.array(trans_in_size) == np.array(crop_data.shape)).all()

    start_coords, end_coords = offset_spatial_crop(roi_center = isocenter, roi_size = trans_in_size)

    # remove the padding
    crop_start, crop_end = [], []
    for i in range(len(ori_size)):
        if end_coords[i] > ori_size[i]:
            diff = end_coords[i] - ori_size[i]
            crop_start.append(diff // 2)
            crop_end.append(crop_data.shape[i] - diff + diff // 2)
        else:
            crop_start.append(0)
            crop_end.append(crop_data.shape[i])

    
    crop_data = crop_data[crop_start[0]: crop_end[0], crop_start[1]: crop_end[1], crop_start[2]: crop_end[2]]

    pad_out = np.zeros(ori_size)

    pad_out[start_coords[0]: end_coords[0], start_coords[1]: end_coords[1], start_coords[2]: end_coords[2]] = crop_data 
    
    return pad_out

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('cfig_path',  type = str)
    parser.add_argument('--phase', default = 'test', type = str)
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ data loader -----------------#
    loaders = data_loader_lightning.GetLoader(cfig = cfig['loader_params'])
    test_loader = loaders.val_dataloader()


        
    model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
    num_classes = cfig['model_params']['out_channels'],
    model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
    kernel_size = cfig['model_params']['kernel_size'],   
    deep_supervision = cfig['model_params']['deep_supervision']   
    ).to(device)
    # load pretrained model 
    model.load_state_dict(torch.load(cfig['save_model_path'], map_location = device))

    with torch.no_grad():
        model.eval()
        for batch_idx, data_dict in enumerate(test_loader):
            # Forward pass
            outputs = model(data_dict['data'].to(device)) 

            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone())  

            outputs = outputs * cfig['scale_out']

            if 'label' in data_dict.keys():
                print ('L1 error is ', torch.nn.L1Loss()(outputs, data_dict['label'].to(device)).item())
            


            if cfig['loader_params']['in_size'] != cfig['loader_params']['out_size']:
                outputs = torch.nn.functional.interpolate(outputs, size = cfig['loader_params']['in_size'], mode = 'area')


            for index in range(len(outputs)):
                pad_out = np.zeros(data_dict['ori_img_size'][index].numpy().tolist())
                crop_data = outputs[index][0].cpu().numpy() 
                ori_size = data_dict['ori_img_size'][index].numpy().tolist()
                isocenter = data_dict['ori_isocenter'][index].numpy().tolist()
                trans_in_size = cfig['loader_params']['in_size']

                pred2orisize = cropped2ori(crop_data, ori_size, isocenter, trans_in_size) * cfig['loader_params']['dose_div_factor']

                np.save(os.path.join(cfig['save_pred_path'], data_dict['id'][index] + '_pred.npy'), pred2orisize)
                
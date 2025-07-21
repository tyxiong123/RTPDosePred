from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def upkern_init_load_weights(network,pretrained_net):
    pretrn_dct = pretrained_net.state_dict()
    model_dct = network.state_dict()
    
    for k in model_dct.keys():
        # Only common keys and core algorithm in Demo code
        if k in model_dct.keys() and k in pretrn_dct.keys():
            #print(model_dct[k].shape)
            #print(pretrn_dct[k].shape)
            #inc1 , outc1 , * spt_dims1 = model_dct[k].shape
            #inc2 , outc2 , * spt_dims2 = pretrn_dct[k].shape
            spt_dims1 = model_dct[k].shape
            spt_dims2 = pretrn_dct[k].shape
            if spt_dims1 == spt_dims2: # standard init
                model_dct[k] = pretrn_dct[k]
            else : # Upsampled kernel init
                inc1 , outc1 , * spt_dims1 = model_dct[k].shape
                model_dct[k] = F.interpolate(pretrn_dct[k],size = spt_dims1 ,mode ='trilinear')
    network.load_state_dict(model_dct)
    return network




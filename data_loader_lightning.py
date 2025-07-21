

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
from toolkit import *


HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}



class MyDataset(Dataset):
    
    def __init__(self, cfig, phase, dev_split):
        '''
        phase: train, validation, or testing 
        
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        
        df = pd.read_csv(cfig['csv_root'])

        if phase == 'valid' and dev_split == 'valid':
            df = pd.read_csv(cfig['csv_root_validation'])
        
        df = df.loc[(df['phase'] == phase)]

        self.phase = phase
        self.dev_split = dev_split
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        # PatientID = ID.split('+')[0]

        # if len(str(PatientID)) < 3:
        #     PatientID = f"{PatientID:0>3}"

        data_npz = np.load(data_path, allow_pickle=True)


        In_dict = dict(data_npz)['arr_0'].item()

        isocenter = In_dict['isocenter']
        ori_img_size = In_dict['Body'].shape
        
        KEYS = list(In_dict.keys())
        print(In_dict['oar_serial'].shape)
        print(In_dict['Body'].shape)

        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)
        
        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.aug = tt_augmentation(KEYS, self.cfig['in_size'],  self.cfig['out_size'], isocenter)
            else:
                self.aug = tr_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)
            
        if self.phase in ['val', 'test', 'valid', 'external_test'] or self.dev_split in ['test','valid']:
            self.aug = tt_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)


        In_dict = self.aug(In_dict)

        data_dict = dict()  


        if 'label' in In_dict.keys():
            data_dict['label'] = In_dict['label']
            ref_dose = In_dict['label'] * 1
            data_dict['ref_5Gy_mask'] = (ref_dose > 5) & (In_dict['Body'] > 0)

        In_dict['Body'] = (In_dict['Body'] > 0.5).type(torch.FloatTensor)

        # if self.cfig['CatStructures']:
        #     data_dict['data'] = torch.cat((cat_optptv, cat_ptv, cat_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0)
        # else:
        data_dict['data'] = torch.cat((In_dict['comb_optptv'],  In_dict['comb_oar_priority'],  In_dict['comb_oar_distance'], In_dict['Body'], In_dict['mass_density'], In_dict['beam_plate_norm']), axis=0) # , In_dict['obj_2DGy'], In_dict['obj_2DWei']

        data_dict['Body'] = In_dict['Body']

        data_dict['PTV'] = In_dict['PTV'] * In_dict['Body']
        data_dict['oar_serial'] = In_dict['oar_serial'] * In_dict['Body']
        data_dict['oar_parallel'] = In_dict['oar_parallel'] * In_dict['Body']
        
        data_dict['ori_isocenter'] = torch.tensor(isocenter)
        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['id'] = ID
        del In_dict

        return data_dict
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
    def train_dataloader(self):
        dataset = MyDataset(self.cfig,  phase='train', dev_split = 'train') 
        return DataLoader(dataset, batch_size=self.cfig['train_bs'],  shuffle=True, num_workers=self.cfig['num_workers'])

    def train_val_dataloader(self):
        dataset = MyDataset(self.cfig, phase='valid', dev_split = 'valid') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])
    
    def val_dataloader(self):
        dataset = MyDataset(self.cfig, phase='valid', dev_split = 'test') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])
    
    def test_dataloader(self):
        dataset = MyDataset(self.cfig, phase='test', dev_split = 'test') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])


if __name__ == '__main__':

    cfig = {
            'train_bs': 1,
             'val_bs': 1, 
             'num_workers': 0, 
             'csv_root': 'meta_files/meta_data_dummy.csv',
             'csv_HaN_OAR_priority_root': 'meta_files/HaN_OAR_update.csv',
             'csv_LUNG_OAR_priority_root': 'meta_files/LUNG_OAR_update.csv',
             'scale_dose_dict': 'meta_files/PTV_DICT.json',
             'pat_obj_dict': 'meta_files/Pat_Obj_DICT.json',
             'down_HU': -1000,
             'up_HU': 1000,
             'denom_norm_HU': 500,
             'in_size': [96, 128, 160], 
             'out_size': [96, 128, 160], 
             'CatStructures': False,
             'dose_div_factor': 10 
             }
    
    loaders = GetLoader(cfig)
    train_loader = loaders.train_dataloader()



    out_dir = './input_lable_figs'
    for i, data in enumerate(train_loader):
        out_case_dir = f"{out_dir}/{data['id'][0]}"
        if not os.path.exists(out_case_dir):
            os.makedirs(out_case_dir)
       # pdb.set_trace()

        data_array = data['oar_serial'][0,0,:,:,:].numpy()
        print(data_array.shape)
        print(np.max(data_array), np.min(data_array))
        iso_center = data['ori_isocenter'][0].numpy()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data_array[round(iso_center[0]),:,:])
        out_path = f"{out_case_dir}/oar_serial.png"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=600)
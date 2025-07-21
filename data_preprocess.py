

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb

from toolkit import *


HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}






class CreateDataset():
    
    def __init__(self, cfig):
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
        

        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))

        df_han_oar = pd.read_csv(cfig['csv_HaN_OAR_priority_root'])
        self.HaN_OAR_name = df_han_oar['OAR_Name'].tolist()
        self.HaN_OAR_priority = df_han_oar['Priority'].tolist()
        self.HaN_OAR_PRIORITY_DICT = dict(zip(self.HaN_OAR_name, self.HaN_OAR_priority))
        self.HaN_isDmax = df_han_oar['isDmax'].tolist()
        self.HaN_isDmax_DICT = dict(zip(self.HaN_OAR_name, self.HaN_isDmax))

        df_lung_oar = pd.read_csv(cfig['csv_LUNG_OAR_priority_root'])
        self.Lung_OAR_name = df_lung_oar['OAR_Name'].tolist()
        self.Lung_OAR_priority = df_lung_oar['Priority'].tolist()
        self.Lung_OAR_PRIORITY_DICT = dict(zip(self.Lung_OAR_name, self.Lung_OAR_priority))
        self.Lung_isDmax = df_lung_oar['isDmax'].tolist()
        self.Lung_isDmax_DICT = dict(zip(self.Lung_OAR_name, self.Lung_isDmax))

        if not os.path.exists(cfig['dataset_save_root']):
            os.makedirs(cfig['dataset_save_root'])



    
    def create_dataset(self):
        for index in range(len(self.data_list)):
            data_path = self.data_list[index]
            ID = self.data_list[index].split('/')[-1].replace('.npz', '')
            PatientID = ID.split('+')[0]

            if self.site_list[index] < 1.5: #To be validated
                OAR_LIST = self.HaN_OAR_name
                OAR_PRIORITY = self.HaN_OAR_PRIORITY_DICT
                OAR_isDmax = self.HaN_isDmax_DICT
                
            else:
                OAR_LIST = self.Lung_OAR_name
                OAR_PRIORITY = self.Lung_OAR_PRIORITY_DICT
                OAR_isDmax = self.Lung_isDmax_DICT
            if len(str(PatientID)) < 3:
                PatientID = f"{PatientID:0>3}"

            data_npz = np.load(data_path, allow_pickle=True)

            In_dict = dict(data_npz)['arr_0'].item()

            spacing = [2.0,2.5,2.5]
            In_dict['spacing'] = spacing # [z,y,x] in mm
            angle_list = In_dict['angle_list']
            In_dict['ed'] = HU2electron_density(In_dict['img']) * In_dict['Body'] 
            In_dict['md'] = HU2mass_density(In_dict['img']) * In_dict['Body'] 
            In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 


            oar_serial, oar_parallel = oar_mask(In_dict, OAR_LIST, OAR_PRIORITY,OAR_isDmax)
            In_dict['oar_serial'] = oar_serial
            In_dict['oar_parallel'] = oar_parallel   
            if 'dose' in In_dict.keys():
                ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
                In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] 
                PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
                norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5) # D97
                In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
                In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)


            isocenter = In_dict['isocenter']

            

            KEYS = list(In_dict.keys())
            for key in In_dict.keys(): 
                if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                    In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
                else:
                    KEYS.remove(key)



            try:
                need_list = self.pat_obj_dict[ID.split('+')[0]] # list(a.values())[0]
            except:
                need_list = OAR_LIST
                print (ID.split('+')[0],  '-------------not in the pat_obj_dict')
            #comb_oar, cat_oar  = combine_oar(In_dict, need_list, self.cfig['norm_oar'], OAR_DICT)
            comb_oar_priority  = combine_oar_priority(In_dict, need_list, OAR_PRIORITY)

            

            opt_dose_dict = {}

            for key in self.scale_dose_Dict[PatientID].keys():
                if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                    opt_dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']

                
            comb_optptv, prs_opt, cat_optptv = combine_ptv(In_dict, opt_dose_dict)

            if 'dose' in In_dict.keys():
                label = In_dict['dose'] * In_dict['Body'] 


            

            save_data_dict = dict()

            save_data_dict['prompt'] = [In_dict['isVMAT'], len(prs_opt), self.site_list[index], self.cohort_list[index]]

            save_data_dict['comb_optptv'] = np.squeeze(comb_optptv.to('cpu').numpy())
            save_data_dict['comb_oar_priority'] = np.squeeze(comb_oar_priority.to('cpu').numpy())

            #distance_map = calculate_min_distance_to_tumor_surface(save_data_dict['comb_optptv'], spacing) ##mm
            distance_map = calculate_min_distance_to_tumor_surface(save_data_dict['comb_optptv'], In_dict['spacing']) ##mm

            # plt.imshow(distance_map[round(isocenter[0]),:,:])
            # plt.colorbar()
            # plt.show()
            save_data_dict['comb_oar_distance'] = calculate_distance_to_tumor(In_dict,distance_map,need_list, OAR_PRIORITY)

            # plt.imshow(save_data_dict['comb_oar_distance'][round(isocenter[0]),:,:])
            # plt.colorbar()
            # plt.show()

            save_data_dict['Body'] = np.squeeze(In_dict['Body'].to('cpu').numpy())

            
            save_data_dict['electron_density'] = np.squeeze(In_dict['ed'].to('cpu').numpy())
            save_data_dict['mass_density'] = np.squeeze(In_dict['md'].to('cpu').numpy())
            #save_data_dict['beam_plate'] = np.squeeze(beam_plate.to('cpu').numpy())

            if 'dose' in In_dict.keys():
                save_data_dict['label'] = np.squeeze(label.to('cpu').numpy())
            
            if self.site_list[index] < 1.5:
                save_data_dict['PTVHigh'] = np.squeeze(In_dict['PTVHighOPT'].to('cpu').numpy())
            else:
                save_data_dict['PTVHigh'] = np.squeeze(In_dict['PTV'].to('cpu').numpy())
            save_data_dict['PTV'] = save_data_dict['comb_optptv'] > 0
            save_data_dict['PTV_expanded'] = expand_roi(save_data_dict['PTV'], In_dict['spacing'], 15)
            
            save_data_dict['isocenter'] = isocenter

            try:
                beam_plate = np.squeeze(In_dict['beam_plate'].to('cpu').numpy())
                beam_plate_norm = beam_plate / round(beam_plate[int(isocenter[0]),int(isocenter[1]),int(isocenter[2])]) * save_data_dict['Body']
            except: 
                beam_plate = get_allbeam_plate(save_data_dict['PTV'], isocenter, spacing, angle_list, with_distance = True)                
                beam_plate_norm = beam_plate/len(angle_list) * save_data_dict['Body']
            save_data_dict['beam_plate'] = beam_plate
            save_data_dict['beam_plate_norm'] = beam_plate_norm

            save_data_dict['oar_serial'] = np.squeeze(In_dict['oar_serial'].to('cpu').numpy())
            save_data_dict['oar_parallel'] = np.squeeze(In_dict['oar_parallel'].to('cpu').numpy())

            if not os.path.exists(self.cfig['dataset_save_root']):
                os.makedirs(self.cfig['dataset_save_root'])
            np.savez_compressed(f"{self.cfig['dataset_save_root']}/{ID}.npz", arr_0 = save_data_dict)
            
         
            #np.savez_compressed(f"{self.cfig['dataset_save_root']}/{ID}.npz", comb_optptv = save_data_dict['comb_optptv'], comb_oar_priority = save_data_dict['comb_oar_priority'], comb_oar_distance = save_data_dict['comb_oar_distance'], Body = save_data_dict['Body'].astype('bool'), election_density = save_data_dict['election_density'], mass_density = save_data_dict['mass_density'], beam_plate = save_data_dict['beam_plate'], label = save_data_dict['label'], PTVHigh = save_data_dict['PTVHigh'].astype('bool'), PTV = save_data_dict['PTV'].astype('bool'), PTV_expanded = save_data_dict['PTV_expanded'].astype('bool'), isocenter = save_data_dict['isocenter'])
            
            del In_dict



        return 
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig

    
    def dataset_creation(self):
        dataset = CreateDataset(self.cfig)
        dataset.create_dataset()
        return 
    

if __name__ == '__main__':

    cfig = {
             'csv_root': 'meta_files/meta_data_sanity_validation.csv',
            'csv_HaN_OAR_priority_root': 'meta_files/HaN_OAR_update.csv',
            'csv_LUNG_OAR_priority_root': 'meta_files/LUNG_OAR_update.csv',
             'scale_dose_dict': 'meta_files/PTV_DICT.json',
             'pat_obj_dict': 'meta_files/Pat_Obj_DICT.json',
             'dataset_save_root': './data_preprocessed',
             'down_HU': -1000,
             'up_HU': 1000,
             'denom_norm_HU': 500,
             'dose_div_factor': 10 
             }
    
    loaders = GetLoader(cfig)
    loaders.dataset_creation()

    



    

import numpy as np
import json 
import matplotlib.pyplot as plt

sanity_check_evaluation_list = ['0617-259694+2Ac+MOS_33896', '0617-259694+imrt+MOS_33896', 'HNC_001+9Ag+MOS_25934', 'HNC_001+A4Ac+MOS_25934'] #  defined by organizers

PTVHighname_list = ['PTV', 'PTV',  'PTVHighOPT', 'PTVHighOPT']  # provided by organizers, lung site is PTV, HNC site is PTVHighOPT

reference_data_folder = './data' # only accessable to organizers for Phase II and Phase III. 

prediction_folder = 'results' 
#prediction_folder = '../../submission/results/lightning' 

MAE_list = []

for i in range(len(sanity_check_evaluation_list)):

    plan_file_name = sanity_check_evaluation_list[i]
    PTVHighname = PTVHighname_list[i]

    patient_id = plan_file_name.split('+')[0]

    data_path = f'{reference_data_folder}/{plan_file_name}.npz'
    data_npz = np.load(data_path, allow_pickle=True)
    data_dict = dict(data_npz)['arr_0'].item()
    scale_dose_Dict = json.load(open('meta_files/PTV_DICT.json'))
    ref_dose = data_dict['dose'] * data_dict['dose_scale']
    ptv_highdose =  scale_dose_Dict[patient_id]['PTV_High']['PDose']
    norm_scale = ptv_highdose / (np.percentile(ref_dose[data_dict[PTVHighname].astype('bool')], 3) + 1e-5)
    ref_dose = ref_dose * norm_scale

    prediction = np.load(f'{prediction_folder}/{plan_file_name}_pred.npy')

    isodose_5Gy_mask = ((ref_dose > 5) | (prediction > 5)) & (data_dict['Body'] > 0) # the mask include the body AND the region where the dose/prediction is higher than 5Gy

    diff = ref_dose - prediction

    error = np.mean(np.abs(diff)[isodose_5Gy_mask > 0])

    MAE_list.append(error)

print ('the Metric 1 MAE displayed in leaderboard should be: ', np.mean(MAE_list))
print ('the MAE for each case is: ', MAE_list)
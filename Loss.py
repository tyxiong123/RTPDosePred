import torch.nn as nn
import torch


class DVHLoss(nn.Module):
    def __init__(self):
        super(DVHLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, mask, dose):



        vols = torch.sum(mask, axis=(2, 3, 4)) #
        vols = vols.clamp(min=1) # Avoid division by zero


        momentOfStructure = {'PTV': {'moments': [2, 4, 6], 'weights': [1, 1, 1]},
                'serial_OAR': {'moments': [5, 10], 'weights': [1, 1]},
                'parallel_OAR': {'moments': [1, 2], 'weights': [1, 1]}}

        keys = list(momentOfStructure.keys())
        #momentOfStructure = dict([(k, v) for k in keys for v in values])
        oarPredMoment = []
        oarRealMoment = []

        epsilon = 0.00001#Added epsilon as the loss function can become sqrt(0)


        for i in range(mask.shape[1]):
            moments = momentOfStructure[keys[i]]['moments']
            weights = momentOfStructure[keys[i]]['weights']
            for j in range(len(moments)):
                gEUDa = moments[j]
                weight = weights[j]


                oarpreddose = predicted_dose*mask[:, i, :, :, :]
                oarRealDose = dose * mask[:, i, :, :, :]


                oarPredMomenta = weight*torch.pow((1 / vols[:, i]) * (torch.sum(torch.pow(oarpreddose, gEUDa)*mask[:, i, :, :, :], axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                oarRealMomenta = weight*torch.pow((1 / vols[:, i]) * (torch.sum(torch.pow(oarRealDose, gEUDa)*mask[:, i, :, :, :], axis=(2, 3, 4))) + epsilon, 1 / gEUDa)

                oarPredMoment.append(oarPredMomenta)
                oarRealMoment.append(oarRealMomenta)
        oarPredMoment = torch.stack(oarPredMoment)
        oarRealMoment = torch.stack(oarRealMoment)

        return self.loss(oarPredMoment, oarRealMoment)
    
def L1_DVH_Loss(pd_dose, gt_dose,ptv_mask,oar_serial_mask,oar_parallel_mask,device,weight=0.01):

    dvh_loss = DVHLoss().to(device)
    #roi = mask[:, 2, :, :, :].unsqueeze(1).type(torch.bool)
    if torch.sum(oar_serial_mask) == 0 and torch.sum(oar_parallel_mask) > 0:
        oar_serial_mask = oar_parallel_mask
    elif torch.sum(oar_serial_mask) > 0 and torch.sum(oar_parallel_mask) == 0:
        oar_parallel_mask = oar_serial_mask
    elif torch.sum(oar_serial_mask) == 0 and torch.sum(oar_parallel_mask) == 0:
        oar_serial_mask = ptv_mask
        oar_parallel_mask = ptv_mask

    roi = torch.cat((ptv_mask,oar_serial_mask,oar_parallel_mask), dim=1).type(torch.bool)

    dvhloss = dvh_loss(pd_dose,roi,gt_dose)

    L1Loss = nn.L1Loss().to(device)
    mae_loss = L1Loss(pd_dose,gt_dose)

    total_loss = weight*dvhloss + mae_loss


    return total_loss
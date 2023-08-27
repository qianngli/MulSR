# coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable
import os
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from data_utils import  ValsetFromFolder

from option import  opt
from model import Net
from os import listdir
import scipy.io as scio
import pdb
from eval import *

def generate_coarse(model, lr_hsi, hr_rgb):

    data = np.zeros((lr_hsi.shape[1], hr_rgb.shape[2], hr_rgb.shape[3])).astype(np.float32) 
     
    with torch.no_grad():

        if opt.cuda:
            lr_hsi = lr_hsi.cuda()
            hr_rgb = hr_rgb.cuda() 
        
        for i in range(lr_hsi.shape[1]):
            neigbor = []       	    
            if i==0:                                
                neigbor.append(lr_hsi[:,1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,2,:,:].data.unsqueeze(1))
                	                 		                	                
            elif i==lr_hsi.shape[1]-1:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i-2,:,:].data.unsqueeze(1))               	
            else:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i+1,:,:].data.unsqueeze(1))


            neigbor.append(lr_hsi[:,i,:,:].data.unsqueeze(1))               
            neigbor =  Variable(torch.cat(neigbor, 1))
             
            _, SR = model(neigbor, hr_rgb)   
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            flops = FlopCountAnalysis(model, (neigbor, hr_rgb))   
            print("FLOPs: ", flops.total())    
            pdb.set_trace()          
                       
            data[i, :,:] = SR.detach().cpu().numpy()
 	
    return data        	

    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])
    
if __name__ == "__main__":

    cPSNRs = []
    cSSIMs = []
    cSAMs = []  
    cERGASs = []
    cRMESs = []    
            
    input_path = '/data/liqiang/HSI-Fusion/CAVE/test_mat_nosie_6/8/'
    out_path = '/data/liqiang/HSI-Fusion/CAVE/result/8/MSCL/'
    if not os.path.exists(out_path):
        os.makedirs(out_path) 
        
    names = [x for x in listdir(input_path) if is_image_file(x)]    
    opt.upscale_factor = 8
    model = Net(opt)      

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 

    model_path = 'result/CAVE_model_8_4.pth'
    print("=> loading premodel")  

    checkpoint = torch.load(model_path)          
    model.load_state_dict(checkpoint['model'])
 
    CSR = sio.loadmat('P_N_V2.mat')['P']
    P = Variable(torch.unsqueeze(torch.from_numpy(CSR), 0)).type(torch.cuda.FloatTensor) 
             
    for k in range(len(names)):
        mat = scio.loadmat(input_path + names[k])  
        print(names[k]) 
        lr_hsi = mat['LR'].astype(np.float32).transpose(2,0,1)
        hsi = mat['HR'].astype(np.float32).transpose(2,0,1)  
        hr_rgb = mat['RGB'].astype(np.float32).transpose(2,0,1)
            
        lr_hsi = Variable(torch.from_numpy(lr_hsi).unsqueeze(0))         
        hr_rgb = Variable(torch.from_numpy(hr_rgb).unsqueeze(0))      
   
        c_result = generate_coarse(model, lr_hsi, hr_rgb)

        
        c_result[c_result>1.]=1
        c_result[c_result<0]=0
        	
        m_psnr = PSNR(hsi, c_result)
        m_sam = SAM(hsi, c_result)
        m_ssim = SSIM(hsi, c_result)
        m_ergas = ERGAS(hsi, c_result, opt.upscale_factor)
        m_rmes = RMES(hsi, c_result)

        cPSNRs.append(m_psnr)
        cSSIMs.append(m_ssim)  
        cSAMs.append(m_sam) 
        cERGASs.append(m_ergas)  
        cRMESs.append(m_rmes)
        
        #sio.savemat(out_path  + names[k], {'SR':c_result.transpose(1,2,0)})   

        print("===The {}-th picture=====PSNR:{:.4f}=====SSIM:{:.5f}=====SAM:{:.4f}=====ERGAS:{:.4f}=====RMES:{:.4f}=======Name:{}".format(k+1,  m_psnr, m_ssim, m_sam, m_ergas, m_rmes, names[k]))  
          
    print("=====averPSNR:{:.2f}=====averSSIM:{:.4f}=====averSAM:{:.2f}======averERGAS:{:.4f}=====averRMES:{:.4f}".format(np.mean(cPSNRs), np.mean(cSSIMs), np.mean(cSAMs), np.mean(cERGASs),np.mean(cRMESs)))     	
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                       
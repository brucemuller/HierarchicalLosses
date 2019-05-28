import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.tree import getTreeList

def cross_entropy2d(input, target, weight=None):  # , size_average=True   moved this out
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        print("size between input and taget was inconsistent")
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
    #input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    #target = target.view(-1)         

    loss = F.cross_entropy(
        input, target, weight=weight,  ignore_index=250
    )   # size_average=size_average, moved out

    return loss

def tree_loss(input, target, weight=None, root=None, use_hierarchy=True):  
    
    sm_cnn_out = F.softmax(input, dim = 1)
     
    loss_list = [] # for recording value of level losses
    initial_run = True
    
    for level_loss_list in getTreeList(root):
        
        summed_probabilities = sm_cnn_out.clone() 
        for branch in level_loss_list:
            initial = True
            for chan in branch:
                if initial:
                    initial = False
                    sli = sm_cnn_out[:,chan,:,:].unsqueeze(1)
                else:
                    sli = torch.cat((sli, sm_cnn_out[:,chan,:,:].unsqueeze(1)), 1)
            
            summed_branch_slice = sli.sum(1,keepdim=True) 
            for chan in branch:
                summed_probabilities[:,chan:(chan+1),:,:] = summed_branch_slice
                
        if initial_run==True:
            summed_probabilities = summed_probabilities + 0.0001
            total_loss = F.nll_loss(torch.log(summed_probabilities), target)   
            initial_run = False
            loss_list.append(total_loss.data.cpu().numpy())
            if use_hierarchy == False:
                return (total_loss, loss_list)
        else:
            summed_probabilities = summed_probabilities + 0.0001 
            level_loss = F.nll_loss(torch.log(summed_probabilities), target)
            loss_list.append(level_loss.data.cpu().numpy()) 
            total_loss += level_loss

    return (total_loss, loss_list)  # returning the total and list of losses at each level


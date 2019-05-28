import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.tree import *

# This is the original non-tree loss I've been using. cross_entropy combines log softmax with nll_loss
def cross_entropy2d(input, target, weight=None):  # , size_average=True   moved this out
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        print("size between input and taget was inconsistent")
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
   # These lines re shape the input and label before passing to cross entropy but is this really necessary? 
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)   
    target = target.view(-1)         # Tested with and without re-shaping and gives same loss. Efficiency gain?
    
    loss = F.cross_entropy(
        input, target, weight=weight,  ignore_index=250
    )
    
    return loss


def tree_loss(input, target, weight=None, root=None, use_hierarchy=True):  
    
    probs = F.softmax(input, dim = 1)
    
    loss_list = []           # Store losses for individual levels
    initial_run = True
    for levellosslist in getTreeList(root):
        
        probs2 = probs.clone() # Deep copy...cnn output name
        
        for j in levellosslist:
            
            # this seg does not assume sequential numbering in levellosslist
            # Collects the relevant channels using concatenation then sums for inference
            initial = True
            for chan in j:
                if initial:
                    initial = False
                    sli = probs2[:,chan,:,:].unsqueeze(1)   # var names....replicated probs, 
                else:
                    sli = torch.cat((sli,probs2[:,chan,:,:].unsqueeze(1)),1)
            summed_tensor = sli.sum(1,keepdim=True)

            # Swaps those same channels above with the summed version
            for chan in j:
                probs2[:,chan:(chan+1),:,:] = summed_tensor
                
        if initial_run==True:
            probs2 = probs2 + 0.0001 # Handle numerical instability
            total_loss = F.nll_loss(torch.log(probs2), target)
            initial_run = False
            loss_list.append(total_loss.item())
            if use_hierarchy == False:             # Return bottom level if not using tree hierarchy
                return (total_loss, loss_list)
        else:                                      # Compute loss for higher levels
            probs2 = probs2 + 0.0001 
            level_loss = F.nll_loss(torch.log(probs2), target)
            loss_list.append(level_loss.item())
            total_loss += level_loss
            
    total_loss = total_loss/len(getTreeList(root)) # Take mean of level losses
    return (total_loss, loss_list)  # returning the total and list of losses at each level



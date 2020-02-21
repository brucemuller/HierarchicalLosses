import torch.nn.functional as F
from ptsemseg.tree import getTreeList

def hierarchical_loss(cnn_output, target, root):                 # root represents your hierarchy
    probabilities = F.softmax(cnn_output, dim = 1) ; loss = 0

    precomputed_hierarchy_list = getTreeList(root) # see ptsemseg/tree.py
    for level_loss_list in precomputed_hierarchy_list

        probabilities_tosum = probabilities.clone()
        summed_probabilities = probabilities_tosum
        for branch in level_loss_list:

            # Extract the relevant probabilities according to a branch in our hierarchy.
            branch_probs = torch.FloatTensor()
            for channel in branch:
                branch_probs = torch.cat((branch_probs,probabilities_tosum[:,channel,:,:].unsqueeze(1)),1)

            # Sum these probabilities into a single slice; this is hierarchical inference.
            summed_tree_branch_slice = branch_probs.sum(1,keepdim=True)

            # Insert inferred probability slice into each channel of summed_probabilities given by branch. 
            # This duplicates probabilities for easy passing to standard loss functions such as nll_loss.
            for channel in branch:  
                summed_probabilities[:,channel:(channel+1),:,:] = summed_tree_branch_slice
                
        level_loss = F.nll_loss(log(summed_probabilities), target)
        loss = loss + level_loss
    return(loss)

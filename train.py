import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0,7"

import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import timeit
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger, show_images
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.tree import create_tree_from_textfile, add_channels, add_levels, update_channels, find_depth
#from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from validate import validate

def train(cfg, logger):
    
    # Setup seeds   ME: take these out for random samples
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ",device)

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    
    if torch.cuda.is_available():
        data_path = cfg['data']['server_path']
    else:
        data_path = cfg['data']['path']
    
    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)
    
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    number_of_images_training = t_loader.number_of_images
    
    # Setup Hierarchy
    
    if torch.cuda.is_available():
        if cfg['data']['dataset'] == "vistas":
            if cfg['data']['viking']:
                root = create_tree_from_textfile("/users/brm512/scratch/experiments/meetshah-semseg/mapillary_tree.txt")
            else:
                root = create_tree_from_textfile("/home/userfs/b/brm512/experiments/meetshah-semseg/mapillary_tree.txt")
        elif cfg['data']['dataset'] == "faces":
            if cfg['data']['viking']:
                root = create_tree_from_textfile("/users/brm512/scratch/experiments/meetshah-semseg/faces_tree.txt")
            else:
                root = create_tree_from_textfile("/home/userfs/b/brm512/experiments/meetshah-semseg/faces_tree.txt")
    else:
        if cfg['data']['dataset'] == "vistas":
            root = create_tree_from_textfile("/home/brm512/Pytorch/meetshah-semseg/mapillary_tree.txt")
        elif cfg['data']['dataset'] == "faces":
            root = create_tree_from_textfile("/home/brm512/Pytorch/meetshah-semseg/faces_tree.txt")

    add_channels(root,0)
    add_levels(root,find_depth(root))
    
    class_lookup = [0,10,7,8,9,1,6,4,5,2,3]  # correcting for tree channel and data integer class correspondence  # HELEN
    #class_lookup = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,48,51,45,46,47,49,50,52,53,54,55,56,57,58,59,60,61,62,63,64,65] # VISTAS
    update_channels(root, class_lookup)

    # Setup models for Hierarchical and Standard training. Note we use Tree synonymously with hierarchy

    model_nontree = get_model(cfg['model'], n_classes).to(device)
    model_tree = get_model(cfg['model'], n_classes).to(device)
    model_nontree = torch.nn.DataParallel(model_nontree, device_ids=range(torch.cuda.device_count()))
    model_tree = torch.nn.DataParallel(model_tree, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls_nontree = get_optimizer(cfg)
    optimizer_params_nontree = {k:v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    optimizer_nontree = optimizer_cls_nontree(model_nontree.parameters(), **optimizer_params_nontree)
    logger.info("Using non tree optimizer {}".format(optimizer_nontree))

    optimizer_cls_tree = get_optimizer(cfg)
    optimizer_params_tree = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}
    optimizer_tree = optimizer_cls_tree(model_tree.parameters(), **optimizer_params_tree)
    logger.info("Using non tree optimizer {}".format(optimizer_tree))
    
    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    loss_meter_nontree = averageMeter()
    if cfg['training']['use_hierarchy']:
        loss_meter_level0_nontree = averageMeter()
        loss_meter_level1_nontree = averageMeter()
        loss_meter_level2_nontree = averageMeter()
        loss_meter_level3_nontree = averageMeter()
        
    loss_meter_tree = averageMeter()
    if cfg['training']['use_hierarchy']:
        loss_meter_level0_tree = averageMeter()
        loss_meter_level1_tree = averageMeter()
        loss_meter_level2_tree = averageMeter()
        loss_meter_level3_tree = averageMeter()
        
        
    time_meter = averageMeter()
    epoch = 0
    i = 0
    flag = True
    number_epoch_iters = number_of_images_training / cfg['training']['batch_size']
    
# TRAINING
    start_training_time = time.time()
    
    while i < cfg['training']['train_iters'] and flag and epoch < cfg['training']['epochs']:
       
        epoch_start_time = time.time()
        epoch = epoch + 1
        for (images, labels) in trainloader:
            i = i + 1
            start_ts = time.time()
        
            model_nontree.train()
            model_tree.train()
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer_nontree.zero_grad()
            optimizer_tree.zero_grad()
            
            outputs_nontree = model_nontree(images)
            outputs_tree = model_tree(images)

            #nontree loss calculation
            if cfg['training']['use_tree_loss']:
                loss_nontree = loss_fn(input=outputs_nontree, target=labels, root=root, use_hierarchy = cfg['training']['use_hierarchy'])
                level_losses_nontree = loss_nontree[1]
                mainloss_nontree = loss_fn(input=outputs_nontree, target=labels, root=root, use_hierarchy = False)[0]
            else:
                loss_nontree = loss_fn(input=outputs_nontree, target=labels)
                mainloss_nontree = loss_nontree
            
            #tree loss calculation
            if cfg['training']['use_tree_loss']:
                loss_tree = loss_fn(input=outputs_tree, target=labels, root=root, use_hierarchy = cfg['training']['use_hierarchy'])
                level_losses_tree = loss_tree[1]
                mainloss_tree = loss_tree[0]
            else:
                loss_tree = loss_fn(input=outputs_tree, target=labels)
                mainloss_tree = loss_tree
            
            loss_meter_nontree.update(mainloss_nontree.item())
            if cfg['training']['use_hierarchy'] and not cfg['training']['phased']:
                loss_meter_level0_nontree.update(level_losses_nontree[0])
                loss_meter_level1_nontree.update(level_losses_nontree[1])
                loss_meter_level2_nontree.update(level_losses_nontree[2])
                loss_meter_level3_nontree.update(level_losses_nontree[3])
                
            loss_meter_tree.update(mainloss_tree.item())
            if cfg['training']['use_hierarchy'] and not cfg['training']['phased']:
                loss_meter_level0_tree.update(level_losses_tree[0])
                loss_meter_level1_tree.update(level_losses_tree[1])
                loss_meter_level2_tree.update(level_losses_tree[2])
                loss_meter_level3_tree.update(level_losses_tree[3])

            # optimise nontree and tree
            mainloss_nontree.backward()
            mainloss_tree.backward()
            
            optimizer_nontree.step()
            optimizer_tree.step()

            time_meter.update(time.time() - start_ts)
            
            # For printing/logging stats
            if (i) % cfg['training']['print_interval'] == 0:
                fmt_str = "Epoch [{:d}/{:d}] Iter [{:d}/{:d}] IterNonTreeLoss: {:.4f}  IterTreeLoss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(epoch,cfg['training']['epochs'], i % int(number_epoch_iters),
                                           int(number_epoch_iters), mainloss_nontree.item(), 
                                           mainloss_tree.item(),
                                           time_meter.avg / cfg['training']['batch_size'])
    

                print(print_str)
                logger.info(print_str)
                time_meter.reset()
                
# VALIDATION AFTER EVERY EPOCH
            if (i) % cfg['training']['val_interval'] == 0 or (i) % number_epoch_iters == 0 or (i) == cfg['training']['train_iters']:
                validate(cfg, model_nontree, model_tree, loss_fn, device, root)
                # reset meters after validation
                loss_meter_nontree.reset()
                if cfg['training']['use_hierarchy']:
                    loss_meter_level0_nontree.reset()
                    loss_meter_level1_nontree.reset()
                    loss_meter_level2_nontree.reset()
                    loss_meter_level3_nontree.reset()

                loss_meter_tree.reset()     
                if cfg['training']['use_hierarchy']:
                    loss_meter_level0_tree.reset()
                    loss_meter_level1_tree.reset()
                    loss_meter_level2_tree.reset()
                    loss_meter_level3_tree.reset()
            
            # For de-bugging
            if (i) == cfg['training']['train_iters']:
                flag = False
                break
            
        print("EPOCH TIME (MIN): ", epoch, (time.time() - epoch_start_time)/60.0)
        logger.info("Epoch %d took %.4f minutes" % (int(epoch) , (time.time() - epoch_start_time)/60.0))
           
    print("TRAINING TIME: ",(time.time() - start_training_time)/3600.0)






if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Set config file here
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/faces.yml",       
        help="Configuration file to use"
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    if torch.cuda.is_available(): 
        #SHARED
       
    else:
        logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    train(cfg, logger)
    
    print("FINISHED TRAINING")

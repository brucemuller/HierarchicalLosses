import os
import sys
import yaml
import torch
import argparse
import timeit

from torch.utils import data

from tqdm import tqdm

from ptsemseg.loader import get_loader

#torch.backends.cudnn.benchmark = True

from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations


def validate(cfg, model_nontree, model_tree, loss_fn, device, root):
    
    val_loss_meter_nontree = averageMeter()
    if cfg['training']['use_hierarchy']:
        val_loss_meter_level0_nontree = averageMeter()
        val_loss_meter_level1_nontree = averageMeter()
        val_loss_meter_level2_nontree = averageMeter()
        val_loss_meter_level3_nontree = averageMeter()
        
    val_loss_meter_tree = averageMeter()
    if cfg['training']['use_hierarchy']:
        val_loss_meter_level0_tree = averageMeter()
        val_loss_meter_level1_tree = averageMeter()
        val_loss_meter_level2_tree = averageMeter()
        val_loss_meter_level3_tree = averageMeter()
    
    if torch.cuda.is_available():
        data_path = cfg['data']['server_path']
    else:
        data_path = cfg['data']['path']
    
    data_loader = get_loader(cfg['data']['dataset'])
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)
    
    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)
    
    n_classes = v_loader.n_classes
    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'])
    
    
    # Setup Metrics
    running_metrics_val_nontree = runningScore(n_classes)
    running_metrics_val_tree = runningScore(n_classes)
    
    model_nontree.eval()
    model_tree.eval()
    with torch.no_grad():
        print("validation loop")
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs_nontree = model_nontree(images_val)
            outputs_tree = model_tree(images_val)
         
            if cfg['training']['use_tree_loss']:
                val_loss_nontree = loss_fn(input=outputs_nontree, target=labels_val, root=root, use_hierarchy = cfg['training']['use_hierarchy'])
            else:
                val_loss_nontree = loss_fn(input=outputs_nontree, target=labels_val)
                
            if cfg['training']['use_tree_loss']:
                val_loss_tree = loss_fn(input=outputs_tree, target=labels_val, root=root, use_hierarchy = cfg['training']['use_hierarchy'])
            else:
                val_loss_tree = loss_fn(input=outputs_tree, target=labels_val)
            
            
            # Using standard max prob based classification
            pred_nontree = outputs_nontree.data.max(1)[1].cpu().numpy()   
            pred_tree = outputs_tree.data.max(1)[1].cpu().numpy()
            
            gt = labels_val.data.cpu().numpy()
            running_metrics_val_nontree.update(gt, pred_nontree)  # updates confusion matrix
            running_metrics_val_tree.update(gt, pred_tree)
            
            if cfg['training']['use_tree_loss']:
                val_loss_meter_nontree.update(val_loss_nontree[1][0]) # take the 1st level 
            else:
                val_loss_meter_nontree.update(val_loss_nontree.item())
                
            if cfg['training']['use_tree_loss']:
                val_loss_meter_tree.update(val_loss_tree[0].item())
            else:
                val_loss_meter_tree.update(val_loss_tree.item())
            
            if cfg['training']['use_hierarchy']:
                val_loss_meter_level0_nontree.update(val_loss_nontree[1][0])
                val_loss_meter_level1_nontree.update(val_loss_nontree[1][1])
                val_loss_meter_level2_nontree.update(val_loss_nontree[1][2])
                val_loss_meter_level3_nontree.update(val_loss_nontree[1][3])
                
            if cfg['training']['use_hierarchy']:
                val_loss_meter_level0_tree.update(val_loss_tree[1][0])
                val_loss_meter_level1_tree.update(val_loss_tree[1][1])
                val_loss_meter_level2_tree.update(val_loss_tree[1][2])
                val_loss_meter_level3_tree.update(val_loss_tree[1][3])
            
            if i_val == 1:
                break

        score_nontree, class_iou_nontree = running_metrics_val_nontree.get_scores()
        score_tree, class_iou_tree = running_metrics_val_tree.get_scores()

        ### VISUALISE METRICS AND LOSSES HERE

        val_loss_meter_nontree.reset()
        running_metrics_val_nontree.reset()
        val_loss_meter_tree.reset()
        running_metrics_val_tree.reset()
        if cfg['training']['use_hierarchy']:
            val_loss_meter_level0_nontree.reset()
            val_loss_meter_level1_nontree.reset()
            val_loss_meter_level2_nontree.reset()
            val_loss_meter_level3_nontree.reset()
            
        if cfg['training']['use_hierarchy']:
            val_loss_meter_level0_tree.reset()
            val_loss_meter_level1_tree.reset()
            val_loss_meter_level2_tree.reset()
            val_loss_meter_level3_tree.reset()
                    































if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)

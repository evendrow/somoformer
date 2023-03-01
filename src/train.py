import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import collate_batch, batch_process_joints, get_datasets, create_dataset
from evaluate import evaluate_vim, evaluate_mpjpe 
from model import create_model
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import keypoint_mse, keypoint_mae, bone_dist_mse

def evaluate_loss(model, dataloader, config):
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    loss_avg = AverageMeter()
    dataiter = iter(dataloader)

    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                break
                
            in_joints, in_masks, out_joints, out_masks, pelvis, padding_mask = batch_process_joints(joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(config["DEVICE"])
            
            loss, _ = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, pelvis, padding_mask)
            loss_avg.update(loss.item(), len(pelvis))
            
            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            bar.suffix = " | ".join(summary)
            bar.next()

        bar.finish()

    return loss_avg.avg

def compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, pelvis, padding_mask, epoch=None, mode='val', loss_last=True, optimizer=None):
    
    _, in_F, _, _ = in_joints.shape

    metamask = (mode == 'train')

    pred_joints, aux_joints = model(in_joints, pelvis, padding_mask, metamask=metamask)
    
    loss = keypoint_mse(pred_joints[:,in_F:], out_joints, out_masks)
    loss += config['TRAIN']['aux_weight'] * sum([keypoint_mse(aux[:,in_F:], out_joints, out_masks) for aux in aux_joints])
    loss /= config['TRAIN']['aux_weight']*config['MODEL']['num_layers'] + 1 # To take average over auxilliary losses
    
    return loss, pred_joints

def adjust_learning_rate(optimizer, epoch, config):
    """
    From: https://github.com/microsoft/MeshTransformer/
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs*2/3 = 100
    """
    # dct_multi_overfit_3dpw_allsize_multieval_noseg_rot_permute_id
    lr = config['TRAIN']['lr'] * (config['TRAIN']['lr_decay'] ** epoch) #  (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
    if 'lr_drop' in config['TRAIN'] and config['TRAIN']['lr_drop']:
        lr = lr * (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f'Saving checkpoint to {filename}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(ckpt, os.path.join(config['OUTPUT']['ckpt_dir'], filename))

    
def dataloader_for(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)

def train(config, logger, experiment_name=""):
    
    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True, pin_memory=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")
    logger.info(f"Training on a total of {len(dataloader_train)} batches.")

    if config["DATA"]["joints"] == "somof":
        # 3dpw validation set for val metrics
        dataset_3dpw_train_segmented = create_dataset("3dpw", logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_3dpw_train_segmented = dataloader_for(dataset_3dpw_train_segmented, config, shuffle=True, pin_memory=True)
        dataset_3dpw_val = create_dataset("3dpw", logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_3dpw_val = dataloader_for(dataset_3dpw_val, config, shuffle=True, pin_memory=True)
        
        dataloader_val = dataloader_3dpw_val
    
        # SoMoF datasets for training-time VIM metrics
        dataset_somof_train = create_dataset("somof", logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataset_somof_val = create_dataset("somof", logger, split="valid", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_somof_train = dataloader_for(dataset_somof_train, config, shuffle=False)
        dataloader_somof_val = dataloader_for(dataset_somof_val, config, shuffle=False)
        
    elif config["DATA"]["joints"] == "posetrack":
        # 3dpw validation set for val metrics
        dataset_val = create_dataset("posetrack", logger, split="valid", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_val = dataloader_for(dataset_val, config, shuffle=False, pin_memory=True)
    
        # SoMoF datasets for training-time VIM metrics
        dataset_somof_train = create_dataset("posetrack", logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataset_somof_val = create_dataset("posetrack", logger, split="valid", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_somof_train = dataloader_for(dataset_somof_train, config, shuffle=False)
        dataloader_somof_val = dataloader_for(dataset_somof_val, config, shuffle=False)
    
    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid =  SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))
    
    ################################
    # Create model, loss, optimizer
    ################################

    model = create_model(config, logger)

    if config["MODEL"]["checkpoint"] != "":
        logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
        checkpoint = torch.load(os.path.join(config['OUTPUT']['ckpt_dir'], config["MODEL"]["checkpoint"]))
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    ################################
    # Begin Training 
    ################################
    global_step = 0

    for epoch in range(config["TRAIN"]["epochs"]):

        dataiter = iter(dataloader_train)

        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_avg = AverageMeter()
        disc_loss_avg = AverageMeter()
        disc_acc_avg = AverageMeter()

        if config["TRAIN"]["optimizer"] == "adam":
            adjust_learning_rate(optimizer, epoch, config)

        train_steps =  len(dataloader_train)
        if not config['DATA']['segmented']:
            train_steps = train_steps // 29

        bar = Bar(f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1}", fill="#", max=train_steps)
        
        for i in range(train_steps): #len(dataloader_train)): #enumerate(dataloader_train):
            model.train()
            optimizer.zero_grad()

            ################################
            # Load a batch of data
            ################################
            start = time.time()

            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader_train)
                joints, masks, padding_mask = next(dataiter)
            in_joints, in_masks, out_joints, out_masks, pelvis, padding_mask = batch_process_joints(joints, masks, padding_mask, config, training=True)
            padding_mask = padding_mask.to(config["DEVICE"])
            
            timer["DATA"] = time.time() - start

            ################################
            # Forward Pass 
            ################################
            start = time.time()
            loss, pred_joints = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, pelvis, padding_mask, epoch=epoch, mode='train', optimizer=None)
            
            timer["FORWARD"] = time.time() - start

            ################################
            # Backward Pass + Optimization
            ################################
            start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
            optimizer.step()
                
            timer["BACKWARD"] = time.time() - start

            ################################
            # Logging 
            ################################

            loss_avg.update(loss.item(), len(joints))
            
            summary = [
                f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            for key, val in timer.items():
                 summary.append(f"{key}: {val:.2f}")

            bar.suffix = " | ".join(summary)
            bar.next()

            if cfg['dry_run']:
                break
            
        bar.finish()

        ################################
        # Tensorboard logs
        ################################

        global_step += train_steps


        writer_train.add_scalar("loss", loss_avg.avg, global_step)
        
        if (epoch+1) % config["TRAIN"]["val_frequency"] == 0:

                
            # SOMOF VIM calculations
            if config["DATA"]["joints"] == "somof":
                train_vim_somof = evaluate_vim(model, dataloader_somof_train, config, logger, return_all=True, bar_prefix="SoMoF train") 
                writer_train.add_scalar("VIM_100ms/somof", train_vim_somof[1], global_step)
                writer_train.add_scalar("VIM_240ms/somof", train_vim_somof[3], global_step)
                writer_train.add_scalar("VIM_500ms/somof", train_vim_somof[7], global_step)
                writer_train.add_scalar("VIM_640ms/somof", train_vim_somof[9], global_step)
                writer_train.add_scalar("VIM_900ms/somof", train_vim_somof[13], global_step)
                writer_train.add_scalar("VIM/somof", train_vim_somof.mean(), global_step)
                
                val_vim_somof = evaluate_vim(model, dataloader_somof_val, config, logger, return_all=True, bar_prefix="SoMoF valid")
                writer_valid.add_scalar("VIM_100ms/somof", val_vim_somof[1], global_step)
                writer_valid.add_scalar("VIM_240ms/somof", val_vim_somof[3], global_step)
                writer_valid.add_scalar("VIM_500ms/somof", val_vim_somof[7], global_step)
                writer_valid.add_scalar("VIM_640ms/somof", val_vim_somof[9], global_step)
                writer_valid.add_scalar("VIM_900ms/somof", val_vim_somof[13], global_step)
                writer_valid.add_scalar("VIM/somof", val_vim_somof.mean(), global_step)

                # 3DPW vim and loss calculations

                train_vim_3dpw  = evaluate_vim(model, dataloader_3dpw_train_segmented, config, logger, return_all=True, bar_prefix="3dpw train")
                writer_train.add_scalar("VIM_100ms/3dpw", train_vim_3dpw[1], global_step)
                writer_train.add_scalar("VIM_240ms/3dpw", train_vim_3dpw[3], global_step)
                writer_train.add_scalar("VIM_500ms/3dpw", train_vim_3dpw[7], global_step)
                writer_train.add_scalar("VIM_640ms/3dpw", train_vim_3dpw[9], global_step)
                writer_train.add_scalar("VIM_900ms/3dpw", train_vim_3dpw[13], global_step)
                writer_train.add_scalar("VIM/3dpw", train_vim_3dpw.mean(), global_step)

                val_vim_3dpw = evaluate_vim(model, dataloader_3dpw_val, config, logger, return_all=True, bar_prefix="3dpw valid")
                writer_valid.add_scalar("VIM_100ms/3dpw", val_vim_3dpw[1], global_step)
                writer_valid.add_scalar("VIM_240ms/3dpw", val_vim_3dpw[3], global_step)
                writer_valid.add_scalar("VIM_500ms/3dpw", val_vim_3dpw[7], global_step)
                writer_valid.add_scalar("VIM_640ms/3dpw", val_vim_3dpw[9], global_step)
                writer_valid.add_scalar("VIM_900ms/3dpw", val_vim_3dpw[13], global_step)
                writer_valid.add_scalar("VIM/3dpw", val_vim_3dpw.mean(), global_step)
                
                train_mpjpe_3dpw_joint = evaluate_mpjpe(model, dataloader_3dpw_train_segmented, config, logger, return_all=True, bar_prefix="3dpw train", per_joint=True)
                for i in range(len(train_mpjpe_3dpw_joint)):
                    writer_train.add_scalar(f"MPJPE/3dpw/j_{i}", train_mpjpe_3dpw_joint[i], global_step)

                val_mpjpe_3dpw_joint = evaluate_mpjpe(model, dataloader_3dpw_val, config, logger, return_all=True, bar_prefix="3dpw train", per_joint=True)
                for i in range(len(train_mpjpe_3dpw_joint)):
                    writer_valid.add_scalar(f"MPJPE/3dpw/j_{i}", val_mpjpe_3dpw_joint[i], global_step)
                    
            elif config["DATA"]["joints"] == "posetrack":
                train_vim_somof = evaluate_vim(model, dataloader_somof_train, config, logger, return_all=True, bar_prefix="SoMoF train") 
                writer_train.add_scalar("VIM_100ms/somof", train_vim_somof[1], global_step)
                writer_train.add_scalar("VIM_240ms/somof", train_vim_somof[3], global_step)
                writer_train.add_scalar("VIM_500ms/somof", train_vim_somof[7], global_step)
                writer_train.add_scalar("VIM_640ms/somof", train_vim_somof[9], global_step)
                writer_train.add_scalar("VIM_900ms/somof", train_vim_somof[13], global_step)
                writer_train.add_scalar("VIM/somof", train_vim_somof.mean(), global_step)
                
                val_vim_somof = evaluate_vim(model, dataloader_somof_val, config, logger, return_all=True, bar_prefix="SoMoF valid")
                writer_valid.add_scalar("VIM_100ms/somof", val_vim_somof[1], global_step)
                writer_valid.add_scalar("VIM_240ms/somof", val_vim_somof[3], global_step)
                writer_valid.add_scalar("VIM_500ms/somof", val_vim_somof[7], global_step)
                writer_valid.add_scalar("VIM_640ms/somof", val_vim_somof[9], global_step)
                writer_valid.add_scalar("VIM_900ms/somof", val_vim_somof[13], global_step)
                writer_valid.add_scalar("VIM/somof", val_vim_somof.mean(), global_step)

                # 3DPW vim and loss calculations

                val_vim_posetrack = evaluate_vim(model, dataloader_val, config, logger, return_all=True, bar_prefix="PoseTrack valid")
                writer_valid.add_scalar("VIM_100ms/posetrack", val_vim_posetrack[1], global_step)
                writer_valid.add_scalar("VIM_240ms/posetrack", val_vim_posetrack[3], global_step)
                writer_valid.add_scalar("VIM_500ms/posetrack", val_vim_posetrack[7], global_step)
                writer_valid.add_scalar("VIM_640ms/posetrack", val_vim_posetrack[9], global_step)
                writer_valid.add_scalar("VIM_900ms/posetrack", val_vim_posetrack[13], global_step)
                writer_valid.add_scalar("VIM/posetrack", val_vim_posetrack.mean(), global_step)

        val_loss = evaluate_loss(model, dataloader_val, config)
        writer_valid.add_scalar("loss", val_loss, global_step)


        if cfg['dry_run']:
            break

    if not cfg['dry_run']:
        save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
    logger.info("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name)
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run

    # Set the random seed so operations are deterministic if desired
    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    # Compatibility with both gpu and cpu training
    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        cfg["DEVICE"] = "cpu"

    logger = create_logger(cfg["OUTPUT"]["log_dir"])

    logger.info("Hello!")
    logger.info("Initializing with config:")
    logger.info(cfg)

    train(cfg, logger, experiment_name=args.exp_name)







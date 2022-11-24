import argparse
import torch
import numpy as np

from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset import batch_process_joints, create_dataset, collate_batch
from model import create_model
from utils.metrics import VIM, VAM
from utils.utils import create_logger, load_default_config, AverageMeter


def inference(model, config, input_joints, pelvis, padding_mask, out_len=14):
    """Run inference on the model
    input_joints: torch.Tensor of shape (B, F, J, K)
    pelvis: torch.Tensor of shape (B, 1, 1, K)
    """
    model.eval()
    
    input_joints_flat = input_joints.flatten(-2).permute(1, 0, 2) # (F+i, B, J*K)
    pelvis_flat = pelvis.flatten(-2).permute(1, 0, 2) # (1, B, K)

    with torch.no_grad():
        pred_joints, _ = model(input_joints, pelvis, padding_mask)

    output_joints = pred_joints[:,-out_len:]

    return output_joints

def evaluate_vim(model, dataloader, config, logger, return_all=False, bar_prefix="", vam=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_name = "posetrack" if config["DATA"]["joints"] == "posetrack" else "3dpw"
    bar = Bar(f"EVAL VIM", fill="#", max=len(dataloader))

    vim_avg = AverageMeter()

    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch

        num_people = 1
        if len(joints.shape) == 5:
            _, num_people, _, _, _ = joints.shape

        
        in_joints, in_masks, out_joints, out_masks, pelvis, padding_mask = batch_process_joints(joints, masks, padding_mask, config)
        padding_mask = padding_mask.to(config["DEVICE"])

        pred_joints = inference(model, config, in_joints, pelvis, padding_mask, out_len=out_F)
        pred_masks = out_masks #torch.zeros(out_masks.shape) # TODO do this

        out_joints = out_joints.flatten(-2).cpu().numpy()                 # (B, out_F, J*K)
        pred_joints = pred_joints.cpu().numpy().reshape(out_joints.shape) # (B, out_F, J*K)
        pred_masks = pred_masks.cpu().numpy()

        for person in range(num_people):
            JK = out_joints.shape[2] // num_people
            K = out_joints.shape[-1] // pred_masks.shape[-1]
            J = JK // K

            for k in range(len(out_joints)):
                if padding_mask[k,person] != 0:
                    continue
                
                person_out_joints = out_joints[k,:,JK*person:JK*(person+1)]
                assert person_out_joints.shape == (14, J*K)
                person_pred_joints = pred_joints[k,:,JK*person:JK*(person+1)]
                person_masks = pred_masks[k,:,J*person:J*(person+1)]
                if not vam:
                    vim_score = VIM(person_out_joints, person_pred_joints, dataset_name,  person_masks) * 100 # *100 for 3dpw
                else:
                    pred_visib = in_masks[:,-1:,:].repeat(1, 16, 1) 
                    vam_score = VAM(person_out_joints*1000, person_pred_joints*1000, 200,  pred_visib) * 100 # *100 for 3dpw
            
                if return_all:
                    vim_avg.update(vim_score, 1)
                else:
                    vim_100 = vim_score[2]
                    vim_avg.update(vim_100, 1)
        
        if return_all:
            vim_text = "["+(", ".join(["%.2f" % vim for vim in vim_avg.avg]))+"]"
        else:
            vim_text = "%.3f" % vim_avg.avg
        bar.suffix = f"{bar_prefix} VIM: {vim_text}"
        bar.next()

    bar.finish()

    return vim_avg.avg

def evaluate_mpjpe(model, dataloader, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL VIM", fill="#", max=len(dataloader))

    vim_avg = AverageMeter()

    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
        in_joints, in_masks, out_joints, out_masks, pelvis, padding_mask = batch_process_joints(joints, masks, padding_mask, config)
        pred_joints = inference(model, config, in_joints, pelvis, padding_mask, out_len=out_F)
        pred_masks = torch.ones(out_masks.shape) # TODO do this

        # logger.info("Evaluating VIM and VAM...")
        out_joints = out_joints.cpu() # (B, out_F, N*J,K)
        pred_joints = pred_joints.cpu().reshape(out_joints.shape)           # (B, out_F, N*J,K)
        
        num_people = 1
        if len(joints.shape) == 5:
            _, num_people, _, _, _ = joints.shape
        
        for person in range(num_people):
            J = out_joints.shape[2] // num_people

            for k in range(len(out_joints)):
                if padding_mask[k,person] != 0:
                    continue
                
                person_out_joints = out_joints[k,:,J*person:J*(person+1)]
                person_pred_joints = pred_joints[k,:,J*person:J*(person+1)]

                mpjpe = torch.norm(person_out_joints - person_pred_joints, dim=-1)
                if per_joint:
                    mpjpe = mpjpe.mean(dim=0) * 1000.
                    vim_avg.update(mpjpe, len(person_out_joints))
                else:
                    mpjpe = mpjpe.mean(dim=-1).numpy() * 1000.
                    vim_avg.update(mpjpe, 1/out_joints.shape[2])
        
        if show_avg:
            vim_text = "%.3f" % vim_avg.avg.mean()
        else:
            vim_text = "["+(", ".join(["%.3f" % vim for vim in vim_avg.avg]))+"]"
            
        bar.suffix = f"{bar_prefix} MPJPE: {vim_text}"
        bar.next()

    bar.finish()

    return vim_avg.avg

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--somof", action="store_true", help="Run somof validation instead of 3dpw")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    args = parser.parse_args()

    # assert args.split in ['train', 'valid', 'test'], "Split must be one of [train, test, valid]"
    
    ################################
    # Load checkpoint
    ################################

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    
        # Compatibility with both gpu and cpu training
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        config["DEVICE"] = "cpu"


    logger.info("Hello!")
    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################

    model = create_model(config, logger)
    model.load_state_dict(ckpt['model']) 

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 16
    assert out_F == 14

    if args.somof:
        logger.info("Using SOMOF data")
        in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
        dataset = create_dataset("somof", logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader = DataLoader(dataset, batch_size=config['TRAIN']['batch_size'], num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)
    
    else:
        dataset = create_dataset("3dpw", logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader = DataLoader(dataset, batch_size=config['TRAIN']['batch_size'], num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    if args.metric == 'vim':
        avgs = evaluate_vim(model, dataloader, config, logger, return_all=True)
    elif args.metric == 'mpjpe':
        avgs = evaluate_mpjpe(model, dataloader, config, logger, return_all=True)
    else:
        raise ValueError("Metric must be onf of (vim, mpjpe)")

    idx_to_check = [1, 3, 7, 9, 13]
    vim_100 = str(avgs[idx_to_check[0]])
    vim_240 = str(avgs[idx_to_check[1]])
    vim_500 = str(avgs[idx_to_check[2]])
    vim_640 = str(avgs[idx_to_check[3]])
    vim_900 = str(avgs[idx_to_check[4]])

    print(f'100ms: {vim_100}')
    print(f'240ms: {vim_240}')
    print(f'500ms: {vim_500}')
    print(f'640ms: {vim_640}')
    print(f'900ms: {vim_900}')
    


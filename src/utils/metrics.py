import numpy as np
import torch

def VIM(GT, pred, dataset_name, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        dataset_name: Dataset name
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """

    gt_i_global = np.copy(GT)

    if dataset_name == "posetrack":
        mask = np.repeat(mask, 2, axis=-1)
        # print('mask:', mask.shape)
        # print('gt:', gt_i_global.shape)
        errorPose = np.power(gt_i_global - pred, 2) * mask
        #get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask,axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    else:   #3dpw
        errorPose = np.power(gt_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    return errorPose


def VAM(GT, pred, occ_cutoff, pred_visib):
    """
    Visibility Aware Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        occ_cutoff: Maximum error penalty
        pred_visib: Predicted visibilities of pose, array of shape (pred_len, #joint)
    Output:
        seq_err:
    """
    pred_visib = np.repeat(pred_visib, 2, axis=-1)
    # F = 0
    seq_err = []
    if type(GT) is list:
        GT = np.array(GT)
    GT_mask = np.where(abs(GT) < 0.5, 0, 1)

    for frame in range(GT.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, GT.shape[1], 2):
            if GT_mask[frame][j] == 0:
                if pred_visib[frame][j] == 0:
                    dist = 0
                elif pred_visib[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif GT_mask[frame][j] > 0:
                N += 1
                if pred_visib[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_visib[frame][j] == 1:
                    d = np.power(GT[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            f_err += dist
        
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
        # if f_err > 0:
        # F += 1
    return np.array(seq_err)


def keypoint_mse(output, target, mask=None):
    """Implement 2D and 3D MSE loss
    Arguments:
    output -- tensor of predicted keypoints (B, ..., K, d)
    target -- tensor of ground truth keypoints (B, ..., K, d)
    mask   -- (optional) tensor of shape (B, ..., K, 1)
    """
    assert output.shape == target.shape
    assert len(output.shape) >= 3

    B = output.shape[0]
    K = output.shape[-2]

    dims = len(output.shape)

    if mask is None:
        mask = torch.ones(B, *[1] * (dims - 1)).float().to(output.device)
        valids = torch.ones(B, *[1] * (dims - 3)).to(output.device) * K

    else:
        if len(mask.shape) != len(output.shape):  # i.e. shape is (B, ..., K)
            mask = mask.unsqueeze(-1)

        assert mask.shape[:-1] == output.shape[:-1]

        valids = torch.sum(mask.squeeze(), dim=-1)

    norm = torch.norm(output * mask - target * mask, p=2, dim=-1)
    mean_K = torch.sum(norm, dim=-1) / (valids + 1e-6)
    mean_B = torch.mean(mean_K)

    return mean_B

def keypoint_mae(output, target, mask=None):
    """Implement 2D and 3D MAE loss
    Arguments:
    output -- tensor of predicted keypoints (B, ..., K, d)
    target -- tensor of ground truth keypoints (B, ..., K, d)
    mask   -- (optional) tensor of shape (B, ..., K, 1)
    """
    assert output.shape == target.shape
    assert len(output.shape) >= 3

    B = output.shape[0]
    K = output.shape[-2]

    dims = len(output.shape)

    if mask is None:
        mask = torch.ones(B, *[1] * (dims - 1)).float().to(output.device)
        valids = torch.ones(B, *[1] * (dims - 3)).to(output.device) * K

    else:
        if len(mask.shape) != len(output.shape):  # i.e. shape is (B, ..., K)
            mask = mask.unsqueeze(-1)

        assert mask.shape[:-1] == output.shape[:-1]

        valids = torch.sum(mask.squeeze(), dim=-1)

    loss = torch.nn.SmoothL1Loss()(output*mask*1000, target*mask*1000)
    #loss= torch.norm(output * mask - target * mask, p=1, dim=-1)
    #mean_K = torch.sum(loss, dim=-1) / valids
    #mean_B = torch.mean(mean_K)

    return loss

def bone_dist_mse(output, target):
    """
    output:  (B, F, J, K)
    target:  (B, F, J, K)
    """

    num_kps = output.shape[-2]

    if num_kps == 13:
        kp_scheme = [(6, 0),
                     (6, 1),
                     (0, 2),
                     (1, 3),
                     (2, 4),
                     (3, 5),
                     (7, 9),
                     (8, 10),
                     (9, 11),
                     (10, 12)]
        
        # (B, 1, len(kp_scheme))
        gt_bone_dist = torch.stack([(target[:,:,joint_a] - target[:,:,joint_b]).norm(p=2, dim=2).mean(1) for (joint_a, joint_b) in kp_scheme], dim=-1).unsqueeze(1)
        # (B, F, len(kp_scheme))
        pred_bone_dists = torch.stack([(output[:,:,joint_a] - output[:,:,joint_b]).norm(p=2, dim=2) for (joint_a, joint_b) in kp_scheme], dim=-1)

        joint_dist_loss = (pred_bone_dists - gt_bone_dist).norm(p=2,dim=2)

    return joint_dist_loss.mean()

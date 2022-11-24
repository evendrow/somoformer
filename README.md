# SoMoFormer: Multi-Person Pose Forecasting with Transformers
This is the repository for the paper

> **SoMoFormer: Multi-Person Pose Forecasting with Transformers** <br>
> by Edward Vendrow, Satyajit Kumar, Ehsan Adeli, Hamid Rezatofighi <br>
> https://arxiv.org/abs/2208.14023
> 
> **Abstract:** Human pose forecasting is a challenging problem involving complex human body motion and posture dynamics. In cases that there are multiple people in the environment, one's motion may also be influenced by the motion and dynamic movements of others. Although there are several previous works targeting the problem of multi-person dynamic pose forecasting, they often model the entire pose sequence as time series (ignoring the underlying relationship between joints) or only output the future pose sequence of one person at a time. In this paper, we present a new method, called Social Motion Transformer (SoMoFormer), for multi-person 3D pose forecasting. Our transformer architecture uniquely models human motion input as a joint sequence rather than a time sequence, allowing us to perform attention over joints while predicting an entire future motion sequence for each joint in parallel. We show that with this problem reformulation, SoMoFormer naturally extends to multi-person scenes by using the joints of all people in a scene as input queries. Using learned embeddings to denote the type of joint, person identity, and global position, our model learns the relationships between joints and between people, attending more strongly to joints from the same or nearby people. SoMoFormer outperforms state-of-the-art methods for long-term motion prediction on the SoMoF benchmark as well as the CMU-Mocap and MuPoTS-3D datasets. 

The code for this paper will be made available soon.

For inquiries, please contact [evendrow@stanford.edu](mailto:evendrow@stanford.edu)

## Getting Started

Clone the repo:

```
git clone https://github.com/evendrow/somoformer.git
```

(Optional) Create a Conda environment:
```
conda create -n poseforecast python=3.8
```

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

### Requirements

- pytorch
- pillow
- tensorboard
- progress

## Data

First, create a `data/` folder in the repo root directory. We expect the following structure:
```
data/
    3dpw/
        sequenceFiles/
            test/
            train/
            validation/
    somof/
        3dpw_test_frames_in.json
        3dpw_test_in.json
        ...
```
The 3DPW data files for both the `3dpw` and `somof` folders can be downloaded from the 3DPW website [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/).

## Training

The model is trained with the training script `src/train.py`, with the following instructions:
```
usage: train.py [-h] [--exp_name EXP_NAME] [--cfg CFG]

optional arguments:
  -h, --help           show this help message and exit
  --exp_name EXP_NAME  Experiment name. Otherwise will use timestamp
  --cfg CFG            Config name. Otherwise will use default config
```

You can train a model on 3DPW with
```
python src/train.py --cfg src/configs/release.yaml --exp_name train_on_3dpw
```

## Evaluation
We provide a script to evaluate trained SoMoFormer models. You can run
```
python src/evaluate.py --ckpt ./path/to/model/checkpoint.pth
```
to get these metrics.

## Progress

This repository is work-in-progress and will continue to get updated and improved over the coming months. Please let us know if you would like any particular features!

The models used for our paper are additionally trained on data from AMASS which helps greatly with model performance. We are working on releasing thes scripts we use to process this data for training.
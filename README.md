<img align ="right" src="figures/slab-logo-lofi-dark-transparent.png" width="120">

# Spacecraft Pose Estimation via Keypoint Regression


## Introduction

This repository provides the MATLAB implementations of the Object Detection Network (ODN) and Keypoint Regression Network (KRN) introduced in the paper titled [Towards Robust Learning-Based Pose Estimation of Noncooperative Spacecraft](https://slab.sites.stanford.edu/sites/g/files/sbiybj25201/files/media/file/asc2019_parksharmadamico_final.pdf). The network models, training and validation scripts are all implemented using the MATLAB Deep Learning toolbox by [Tae Ha "Jeff" Park](https://taehajeffpark.com/) and Zahra Ahmed at the [Space Rendezvous Laboratory (SLAB)](https://slab.stanford.edu/) of Stanford University. 

## Dataset

The models are trained on SPEED-UE-Cube, a synthetic image dataset that was created by SLAB using Unreal Engine 5. The full dataset can be downloaded [here.](https://purl.stanford.edu/hw812wb1641) The dataset models spaceborne imagery of a 3U CubeSat and consists of two subsets: a training dataset comprised of 30,000 images with a 80/20 training/validation split, and a trajectory dataset of 1,186 images that depict a rendezvous scenario between the CubeSat and a servicer spacecraft. Additional details about the dataset can be found in the paper titled [SPEED-UE-Cube: A Machine Learning Dataset for Autonomous, Vision-Based Navigation](https://slab.stanford.edu/sites/g/files/sbiybj25201/files/media/file/speed_ue_cube_a_machine_learning_dataset_for_autonomous_vision_based_spacecraft_navigation.pdf). If you use any images from SPEED-UE-Cube or any material from this repository, please cite them as indicated in the References section of this README.

Once downloaded, the dataset should be organized in the structure outlined in `DATASET.md`.

## Requirements

The code is developed and trained using MATLAB R2022b and requires the following toolboxes, which can be downloaded via the Add-On Explorer in MATLAB:
- [Image Processing Toolbox](https://www.mathworks.com/products/image.html)
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
- [Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html)
- [Aerospace Toolbox](https://www.mathworks.com/products/aerospace-toolbox.html)

## How to run scripts

First, make sure the `utils` folder is added to the MATLAB search path.

### Pre-processing
To pre-process the labels for the training and trajectory datsets, first specify the path to `dataroot`:
```
dataroot = <Path to Dataset>;
```
Then, run the following from the Command Window:

```
traindata = fullfile(dataroot,'train');
trajdata  = fullfile(dataroot,'trajectory');
camerafn  = fullfile(dataroot, 'camera.json');
camera    = jsondecode(fileread(camerafn));

createCSV(traindata, camera, 'train');
createCSV(traindata, camera, 'validation');
createCSV(trajdata,  camera, 'test');
```
This will convert the `.json` labels into `.csv` files that can be used for training, validation, and testing the ODN and KRN. The remaining pre-processing is built into the training scripts and does not need to be executed separately. 

### Training

The ODN and KRN have separate training scripts, `trainODN.m` and `trainKRN.m`, respectively. 

In `trainODN.m`, specify `dataroot` as the path to `train`. Additionally, specify a checkpoint name where the best checkpoint will be saved during training:
```
% Inputs:
dataroot = <Path to train folder>
chkpName = <Name of output checkpoint file>;
```
Then, simply run `trainODN.m` from the command window or MATLAB editor.

Follow the same process for `trainKRN.m`.

### Testing
Once the ODN and KRN have been trained and their checkpoints have been saved to the specified files, the entire ODN-KRN pipeline can be used for spacecraft pose estimation using `testAll.m`. This script passes the input images through the ODN, KRN, and finally a Perspective-n-Point (PnP) algorithm to produce a final 6D pose.

In `testAll.m`, first specify the path to `trajectory` in `dataroot` as well as the names of the ODN and KRN checkpoints that will be loaded into the neural nets.

```
% Inputs:
dataroot = <Path to trajectory folder>;
chkpt_odn = <ODN_checkpoint.mat>;
chkpt_krn = <KRN_checkpoint.mat>;
```
Then, run `testAll.m` from the command window or MATLAB editor.

## License

This repository is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license, copyright (c) 2023 Stanford's Space Rendezvous Laboratory.

## References
To reference SPEED-UE-Cube or any material in this repository, please cite the paper that introduced SPEED-UE-Cube as well as the dataset itself.

Paper:

```
@inproceedings{ahmed2024speeduecube,
	author={Ahmed, Z., Park, T. H., Bhattacharjee, A., Fazel-Rezai, R., Graves, R., Saarela, O., Teramoto, R., Vemulapalli, K., D'Amico, S.;},
	booktitle={46th Rocky Mountain AAS Guidance, Navigation and Control Conference},
	title={SPEED_UE_Cube: A Machine Learning Dataset for Autonomous, Vision-Based Spacecraft Navigation},
	year={2024},
	month={February 2-7}
}
```

Dataset:

```
@misc{park2024speeduecubedataset,
	author={Park, T. H., Ahmed, Z., Bhattacharjee, A., Fazel-Rezai, R., Graves, R., Saarela, O., Teramoto, R., Vemulapalli, K., D'Amico, S.;},
	title={Spacecraft PosE Estimation Dataset of a 3U CubeSat using Unreal Engine ({SPEED-UE-Cube})},
	note={Available at \url{https://purl.stanford.edu/hw812wb1641}},
	year={2024}
}
```


To reference the original ODN-KRN architecture, cite as:

```
@inproceedings{park2019krn,
	author={Park, Tae Ha and Sharma, Sumant and D'Amico, Simone},
	booktitle={2019 AAS/AIAA Astrodynamics Specialist Conference, Portland, Maine},
	title={Towards Robust Learning-Based Pose Estimation of Noncooperative Spacecraft},
	year={2019},
	month={August 11-15}
}
```

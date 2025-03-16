# 3DSG


## Manual Data Preparation

1. Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Unpack the image sequences for each scan. And include the 3DSSG files as a subdirectory in 3RScan.
2. Download [ScanNet](http://www.scan-net.org/ScanNet/) and split the scans into ```scannet_2d``` and ```scannet_3d```. We use the pre-processed data from [ScanNet ETH preprocessed 3D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip) & [ScanNet ETH preprocessed 2D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip), when using the pre-processed version make sure that you have acknowledged the ScanNet license. When using processed ScanNet ETH preprocessed 2D frames, use the matching [intrinsics](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
3. Download the [3DSSG_subset.zip](http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip) and extract the files in the 3RScan directory for training and evaluation. Additional meta files can be found [here](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
4. Download 3RScan & ScanNet meta data files using ```scripts/download_scannet_meta.sh``` and ```scripts/download_scannet_meta.sh``` and place them in their data directories.
5. Set the path to your data in ```config/config.py```




## Auto Preparation
Download data
```
cd files
bash preparation.sh
```


### Prepare 3RScan dataset
Please make sure you agree the [3RScan Terms of Use](https://forms.gle/NvL5dvB4tSFrHfQH6) first, and get the download script and put it right at the 3RScan main directory.

Then run
```
python scripts/RUN_prepare_dataset_3RScan.py --download --thread 8
```

### Generate Experiment data
```
# For GT
# This script downloads preprocessed data for GT data generation, and generate GT data.
python scripts/RUN_prepare_GT_setup_3RScan.py --thread 16

# For Dense
# This script downloads the inseg.ply files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Dense_setup_3RScan.py -c configs/dataset/config_base_3RScan_inseg_l20.yaml --thread 16

# For Sparse
# This script downloads the 2dssg_orbslam3.[json,ply] files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Sparse_setup_3RScan.py -c configs/dataset/config_base_3RScan_orbslam_l20.yaml --thread 16
```

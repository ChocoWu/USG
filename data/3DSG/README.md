# 3D Scene Graph (3DSG)

This directory contains the 3DSG data used by this project. The data is built from [3DSSG](https://3dssg.github.io/) and [3RScan](https://github.com/WaldJohannaU/3RScan), with helper scripts following the [Open3DSG](https://github.com/boschresearch/Open3DSG) preparation pipeline.





## Data Sources

- [3DSSG](https://3dssg.github.io/): scene graph annotations, object classes, and relationship labels.
- [3RScan](https://github.com/WaldJohannaU/3RScan): 3D reconstructed scans and RGB-D image sequences.
- [3DSSG_subset.zip](http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip): subset files for training and evaluation.


## Manual Preparation

1. Download [3DSSG](https://3dssg.github.io/) and [3RScan](https://github.com/WaldJohannaU/3RScan).
2. Unpack the image sequences for each 3RScan scan.
3. Place the 3DSSG files as a subdirectory inside the 3RScan directory.


## Automatic Preparation

Download the metadata files:

```bash
cd data/3DSG
bash preparation.sh
```

The preparation script downloads:

- `rescans.txt`
- `train_ref.txt`
- `val_ref.txt`
- `3RScan.json`
- `relationships.json`
- `relationships.txt`
- `classes160.txt`
- `references.txt`

## Prepare 3RScan

Before running the scripts, agree to the [3RScan Terms of Use](https://forms.gle/NvL5dvB4tSFrHfQH6), get the 3RScan download script, and place `download.py` at the main 3RScan directory.

Then run:

```bash
cd data/3DSG
python scripts/RUN_prepare_dataset_3RScan.py \
  -c <open3dsg_config.yaml> \
  --download \
  --thread 8
```

This step downloads the required 3RScan files, unzips image sequences, generates aligned instance meshes, and prepares rendered views.


## Generate Experiment Data

Generate ground-truth data:

```bash
python scripts/RUN_prepare_GT_setup_3RScan.py \
  -c <open3dsg_config.yaml> \
  --thread 16
```

Generate dense training data:

```bash
python scripts/RUN_prepare_Dense_setup_3RScan.py \
  -c <config_base_3RScan_inseg_l20.yaml> \
  --thread 16
```

Generate sparse training data:

```bash
python scripts/RUN_prepare_Sparse_setup_3RScan.py \
  -c <config_base_3RScan_orbslam_l20.yaml> \
  --thread 16
```

## Expected File Structure

The local `data/3DSG/` directory should be organized as:

```text
data/3DSG/
|-- 3DSSG/
|   |-- affordances.txt
|   |-- attributes.txt
|   |-- classes.txt
|   |-- objects.json
|   |-- relationships.json
|   |-- relationships.txt
|   `-- wordnet_attributes.txt
|-- 3DSSG_subset/
|   |-- classes.txt
|   |-- relationships.json
|   |-- relationships.txt
|   |-- relationships_train.json
|   `-- relationships_validation.json
|-- 3RScan/
|   `-- <scan_id>/
|       |-- labels.instances.annotated.v2.ply
|       |-- mesh.refined.0.010000.segs.v2.json
|       |-- mesh.refined.mtl
|       |-- mesh.refined.v2.obj
|       |-- mesh.refined_0.png
|       |-- semseg.v2.json
|       `-- sequence.zip
|-- scripts/
|   |-- RUN_prepare_dataset_3RScan.py
|   |-- RUN_prepare_GT_setup_3RScan.py
|   |-- RUN_prepare_Dense_setup_3RScan.py
|   `-- RUN_prepare_Sparse_setup_3RScan.py
|-- split/
|   |-- train_scans.txt
|   |-- validation_scans.txt
|   `-- test_scans.txt
|-- 3RScan.v2 Semantic Classes - Mapping.csv
|-- intrinsics.txt
|-- relationships_custom.txt
`-- preparation.sh
```

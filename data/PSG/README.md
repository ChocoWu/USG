
## PSG addresses many SGG problems
We believe that the biggest problem of classic scene graph generation (SGG) comes from noisy datasets.
Classic scene graph generation datasets adopt a bounding box-based object grounding, which inevitably causes a number of issues:
- **Coarse localization**: bounding boxes cannot reach pixel-level accuracy,
- **Inability to ground comprehensively**: bounding boxes cannot ground backgrounds,
- **Tendency to provide trivial information**: current datasets usually capture frivolous objects like `head` to form trivial relations like `person-has-head`, due to too much freedom given during bounding box annotation.
- **Duplicate groundings**: the same object could be grounded by multiple separate bounding boxes.

All of the problems above can be easily addressed by the PSG dataset, which grounds the objects using panoptic segmentation with an appropriate granularity of object categories (adopted from COCO).

In fact, the PSG dataset contains 49k overlapping images from COCO and Visual Genome. In a nutshell, we asked annotators to annotate relations based on COCO panoptic segmentations, i.e., relations are mask-to-mask.

| ![psg.jpg](https://live.staticflickr.com/65535/52231743087_2bda038ee2_b.jpg) |
|:--:|
| <b>Comparison between the classic VG-150 and PSG.</b>|

## Clear Predicate Definition
We also find that a good definition of predicates is unfortunately ignored in the previous SGG datasets.
To better formulate PSG task, we carefully define 56 predicates for PSG dataset.
We try hard to avoid trivial or duplicated relations, and find that the designed 56 predicates are enough to cover the entire PSG dataset (or common everyday scenarios).

Type    | Predicates  |
---    | ---       |
Positional Relations (6)     | over, in front of, beside, on, in, attached to. |
Common Object-Object Relations (5) | hanging from, on the back of, falling off, going down, painted on.|
Common Actions (31) | walking on, running on, crossing, standing on, lying on, sitting on, leaning on, flying over, jumping over, jumping from, wearing, holding, carrying, looking at, guiding, kissing, eating, drinking, feeding, biting, catching, picking (grabbing), playing with, chasing, climbing, cleaning (washing, brushing), playing, touching, pushing, pulling, opening.|
Human Actions (4)	 | cooking, talking to, throwing (tossing), slicing.
Actions in Traffic Scene (4) |	driving, riding, parked on, driving on.
Actions in Sports Scene (3)	| about to hit, kicking, swinging.
Interaction between Background (3) |	entering, exiting, enclosing (surrounding, warping in)



## Get Started
To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
```
You shall manually install the following dependencies.
```bash
# Install mmcv
## CAUTION: The latest versions of mmcv 1.5.3, mmdet 2.25.0 are not well supported, due to bugs in mmdet.
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# Install mmdet
pip install openmim
mim install mmdet==2.20.0

# Install coco panopticapi
pip install git+https://github.com/cocodataset/panopticapi.git

# For visualization
conda install -c conda-forge pycocotools
pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

# If you're using wandb for logging
pip install wandb
wandb login

# If you develop and run openpsg directly, install it from source:
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

[Datasets](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh) and [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErQ4stbMxp1NqP8MF8YPFG8BG-mt5geOrrJfAkeitjzASw?e=9taAaU) are provided. Please unzip the files if necessary.

**Before October 2022, we only release part of the PSG data for competition, where part of the test set annotations are wiped out. Users should change the `json` filename in [`psg.py` (Line 4-5)](https://github.com/Jingkang50/OpenPSG/blob/d66dfa70429001ad80c2a8984be9d86a9da703bc/configs/_base_/datasets/psg.py#L4) to a correct filename for training or submission.**

**For the PSG competition, we provide `psg_train_val.json` (45697 training data + 1000 validation data with GT). Participant should use `psg_val_test.json` (1000 validation data with GT + 1177 test data without GT) to submit. Example submit script is [here](https://github.com/Jingkang50/OpenPSG/blob/main/scripts/imp/submit_panoptic_fpn_r50_sgdet.sh). You can use [`grade.sh`](https://github.com/Jingkang50/OpenPSG/blob/main/scripts/grade.sh) to simulate the competition's grading mechanism locally.**

Our codebase accesses the datasets from `./data/` and pretrained models from `./work_dirs/checkpoints/` by default.

If you want to play with VG, please download the VG dataset [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EiBEV1Z3ueBJqJVO4j7z0YwBt_Jvj2AqYTRsiIs-8pZowg?e=C2O5yg), and put it into `./data` dir.
We have pipeline [here](https://github.com/Jingkang50/OpenPSG/blob/main/openpsg/datasets/sg.py) to process the dataset.

# Panoptic Scene Graph Dataset

Panoptic Scene Graph (PSG) extends scene graph annotations from object boxes to panoptic segments. Each image is paired with COCO-style RGB images, panoptic segmentation masks, object and stuff categories, and subject-predicate-object relations between panoptic segments.

Compared with bounding-box scene graph datasets, PSG reduces coarse localization, duplicate object groundings, and trivial relations by grounding objects and backgrounds with COCO panoptic segments. The dataset contains about 49k images overlapping COCO and Visual Genome.

| ![PSG example](https://live.staticflickr.com/65535/52231743087_2bda038ee2_b.jpg) |
|:--:|
| Comparison between classic VG-150 and PSG. |

## Download

Download the official data from the OpenPSG release:

- OpenPSG repository: https://github.com/Jingkang50/OpenPSG
- PSG COCO version: [dataset link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh)
- PSG VG version: [dataset link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EiBEV1Z3ueBJqJVO4j7z0YwBt_Jvj2AqYTRsiIs-8pZowg?e=C2O5yg)

This project expects the PSG COCO-format data under:

```text
data/PSG/
```

## Expected File Structure

After extraction, the directory should look like this:

```text
data/PSG/
|-- README.md
|-- psg.json
|-- psg_train_val.json
|-- psg_val_test.json
|-- tiny_psg.json
`-- coco/
    |-- annotations/
    |   |-- captions_train2017.json
    |   |-- captions_val2017.json
    |   |-- instances_train2017.json
    |   |-- instances_val2017.json
    |   |-- panoptic_train2017.json
    |   |-- panoptic_val2017.json
    |   |-- person_keypoints_train2017.json
    |   |-- person_keypoints_val2017.json
    |   |-- stuff_train2017.json
    |   `-- stuff_val2017.json
    |-- detectron/
    |   |-- attribute_categories.json
    |   |-- stuff_categories.json
    |   |-- thing_categories.json
    |   |-- train_data.json
    |   `-- val_data.json
    |-- train2017/
    |-- val2017/
    |-- panoptic_train2017/
    `-- panoptic_val2017/
```


## Predicate Taxonomy

PSG defines 56 predicate classes. The semantic groups are:

| Group | Predicates |
| --- | --- |
| Positional relations | `over`, `in front of`, `beside`, `on`, `in`, `attached to` |
| Common object-object relations | `hanging from`, `on back of`, `falling off`, `going down`, `painted on` |
| Common actions | `walking on`, `running on`, `crossing`, `standing on`, `lying on`, `sitting on`, `leaning on`, `flying over`, `jumping over`, `jumping from`, `wearing`, `holding`, `carrying`, `looking at`, `guiding`, `kissing`, `eating`, `drinking`, `feeding`, `biting`, `catching`, `picking`, `playing with`, `chasing`, `climbing`, `cleaning`, `playing`, `touching`, `pushing`, `pulling`, `opening` |
| Human actions | `cooking`, `talking to`, `throwing`, `slicing` |
| Traffic scene actions | `driving`, `riding`, `parked on`, `driving on` |
| Sports scene actions | `about to hit`, `kicking`, `swinging` |
| Background interactions | `entering`, `exiting`, `enclosing` |

For training or evaluation, always use the exact order stored in `predicate_classes`.

# Multimodal Scene Graph Pair Recipes

This directory contains sample recipe JSON files for cross-modal scene graph pairs. 
A recipe records which samples are paired and, when needed, which frames or views should be used. 
It does not store raw images, videos, masks, or point clouds.

The raw data is loaded on demand from the source datasets:

- `data/PSG/` for image scene graphs.
- `data/PVSG/` for video scene graphs.
- `data/3DSG/` for 3D scene graphs and 3RScan point clouds.


## Directory Layout

```text
data/multimodal/
|-- I-D/
|   |-- id_pairs_train.json
|   `-- id_pairs_val.json
|-- I-V/
|   |-- iv_pairs_train.json
|   `-- iv_pairs_val.json
|-- S-D/
|   |-- sd_pairs_train.json
|   `-- sd_pairs_val.json
|-- S-I/
|   |-- si_pairs_train.json
|   `-- si_pairs_test.json
`-- README.md
```

## Available Pair Sets

| Pair | Files | Source data | Association target | Count |
| --- | --- | --- | --- | ---: |
| S-I | `S-I/si_pairs_train.json`, `S-I/si_pairs_test.json` | PSG + COCO captions | Text entities are matched to image objects by aligned category labels. | 46,563 train / 2,186 test |
| I-V | `I-V/iv_pairs_train.json`, `I-V/iv_pairs_val.json` | PVSG | Image and video objects are matched by shared PVSG `object_id`. | 338 train / 62 val |
| I-D | `I-D/id_pairs_train.json`, `I-D/id_pairs_val.json` | 3DSSG / 3RScan | 3D objects are projected into RGB-D frames; visible objects are matched by `object_id`. | 3,158 train / 345 val |
| S-D | `S-D/sd_pairs_train.json`, `S-D/sd_pairs_val.json` | 3DSSG / 3RScan + scene captions | Text entities are matched to 3D objects by aligned category labels. | 1,061 train / 117 val |


## Recipe Schemas

### Text-Image (S-I)

```json
{
  "split": "train",
  "pairs": [
    {
      "image_id": "107902",
      "coco_image_id": "417720",
      "caption": "A pretty young lady holding a dark colored umbrella."
    }
  ]
}
```

The dataset loader reads the PSG image annotation by `image_id` and parses the caption into a text scene graph at load time.

### Image-Video (I-V)

```json
{
  "split": "train",
  "num_frames": 8,
  "pairs": [
    {
      "video_id": "0001_4164158586",
      "source": "VidOR",
      "image_frame": 0,
      "video_frames": [90, 102, 115, 128, 140, 153, 166, 179],
      "num_frames_total": 180
    }
  ]
}
```

The image side uses one frame from the video, and the video side uses a temporally separated clip. Object association is computed from stable PVSG object IDs.

### Image-3D (I-D)

```json
{
  "split": "train",
  "frames_per_scan": 3,
  "pairs": [
    {
      "scan_id": "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
      "frame": "frame-000000",
      "num_grounded": 11
    }
  ]
}
```

Each recipe selects RGB-D frames whose visible pixels can be associated with 3D objects in the corresponding 3RScan scene.

### Text-3D (S-D)

```json
{
  "split": "train",
  "pairs": [
    {
      "scan_id": "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
      "caption": "The room is architecturally designed with walls attached to the wooden floor..."
    }
  ]
}
```

## Loading the Pairs

Dataset implementations live in `usg_par/datasets/`:

| Pair | Dataset class | Collate function |
| --- | --- | --- |
| S-I | `SIPairDataset` in `usg_par/datasets/si_pairs.py` | `si_collate` |
| I-V | `IVPairDataset` in `usg_par/datasets/iv_pairs.py` | `iv_collate` |
| I-D | `IDPairDataset` in `usg_par/datasets/id_pairs.py` | `id_collate` |
| S-D | `SDPairDataset` in `usg_par/datasets/sd_pairs.py` | `sd_collate` |

The loaders derive cross-modal association targets on the fly:

- Shared `object_id` for I-V.
- 3D-to-2D visibility projection for I-D.
- Category-level text-to-visual matching for S-I and S-D.


## Text Parsing Dependency

S-I and S-D parse captions into text scene graphs. Install the parser dependencies before using these loaders:

```bash
pip install SceneGraphParser
python -m spacy download en_core_web_sm
```

Use `SceneGraphParser`; the `sng_parser` package on PyPI is not a drop-in replacement for this code.

## Notes

- 3RScan `sequence.zip` files are read lazily by the I-D loader; they do not need to be pre-extracted.

## Dataset

The FACTUAL Scene Graph dataset includes 40,369 instances with lemmatized predicates/relations.

### FACTUAL Scene Graph dataset:

- **Storage**: `data/factual_sg/factual_sg.csv`
- **From Huggingface**: `load_dataset('lizhuang144/FACTUAL_Scene_Graph')`

**Splits**:
- Random Split:
  - Train: `data/factual_sg/random/train.csv`
  - Test: `data/factual_sg/random/test.csv`
  - Dev: `data/factual_sg/random/dev.csv`
- Length Split:
  - Train: `data/factual_sg/length/train.csv`
  - Test: `data/factual_sg/length/test.csv`
  - Dev: `data/factual_sg/length/dev.csv`

**Data Fields**:

- `image_id`: The ID of the image in Visual Genome.
- `region_id`: The ID of the region in Visual Genome.
- `caption`: The caption of the image region.
- `scene_graph`: The scene graph of the image region and caption.

**Related Resources**: Please find the details of images and regions from [Visual Genome](https://huggingface.co/datasets/visual_genome) given their corresponding IDs.

### FACTUAL-MR dataset:

- `data/factual_mr/factual_mr.csv`
- `data/factual_mr/meta.json`: the metadata for mapping the abbreviations of quantifiers in `factual_mr.csv` to their complete names.

### VG Scene Graph dataset:

- **From Huggingface**: `load_dataset('lizhuang144/VG_scene_graph_clean')`
- **Details**: Cleaned to exclude empty instances; includes 2.9 million instances.

### FACTUAL Scene Graph dataset with identifiers:

- **From Huggingface**: `load_dataset('lizhuang144/FACTUAL_Scene_Graph_ID')`
- **Enhancements**: Contains verb identifiers, passive voice indicators, and node indexes.

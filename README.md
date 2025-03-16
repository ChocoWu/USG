## <div align="center"> Universal Scene Graph Generation<div>


##### <div align="center"> <a href="https://chocowu.github.io/">Shengqiong Wu</a>, <a href="http://haofei.vip/">Hao Fei</a>*, and <a href="https://www.chuatatseng.com/">Tat-Seng Chua</a>. <div>
##### <div align="center"> (*Correspondence ) <div>






## Abstract

Scene graph (SG) representations can neatly and efficiently describe scene semantics, which has driven sustained intensive research in SG generation.
In the real world, multiple modalities often coexist, with different types, such as images, text, video, and 3D data, expressing distinct characteristics.
Unfortunately, current SG research is largely confined to single-modality scene modeling, preventing the full utilization of the complementary strengths of different modality SG representations in depicting holistic scene semantics.
To this end, we introduce Universal SG (USG), a novel representation capable of fully characterizing comprehensive semantic scenes from any given combination of modality inputs, encompassing modality-invariant and modality-specific scenes.
Further, we tailor a niche-targeting USG parser, USG-Par, which effectively addresses two key bottlenecks of cross-modal object alignment and out-of-domain challenges.
We design the USG-Par with modular architecture for end-to-end USG generation, in which we devise an object associator to relieve the modality gap for cross-modal object alignment.
Further, we propose a text-centric scene contrasting learning mechanism to mitigate domain imbalances by aligning multimodal objects and relations with textual SGs. 
Through extensive experiments, we demonstrate that USG offers a stronger capability for expressing scene semantics than standalone SGs, and also that our USG-Par achieves higher efficacy and performance.

<!-- <img src="./assets/full-usg3-crop.png" align="center" width="100%"> -->


 ![framework](./static/images/full-usg3-crop.png)




## Mehotd
Our model consists of five main modules.
***First***, we extract the modality-specific features with a modality-specific backbone. 
***Second***, we employ a shared mask decoder to extract object queries for various modalities. 
These object queries are then fed into the modality-specific object detection head to obtain the category label and tracked positions of the corresponding objects. 
***Third***, the object queries are input into the object associator, which determines the association relationships between objects across modalities. 
***Fourth***, a relation proposal constructor is utilized to retrieve the most confidential subject-object pairs. 
***Finally***, a relation decoder is employed to decode the final predicate prediction between the subjects and objects.

  ![framework](./static/images/frame4-cropped.png)



## Data 

To evaluate the efficacy of USG-Par, which supports both single-modality and multi-modality scene parsing, we utilize existing single-modality datasets and a manually constructed multimodal dataset.

* Single-modal Dataset

  - Image:
    1) [Visual Genome (VG)](data/VG/README.md)
    2) [Panoptic Scene Graph (PSG)](data/PSG/README.md)
  - Video:
    1) [Action Genome (AG)](data/AG/README.md)
    2) [Panoptic Video Scene Graph (PVSG)](data/PVSG/README.md)
  - Text:
    1) [FACUTAL](data/FACTUAL/README.md)
  - 3DSG
    1) [3D Scene Graph (3DSG)](data/3DSG/README.md)
  
  please refer to the corresponding instructions for dataset preparation.

* Multi-modal Dataset
  - Text-Image
  
    Inspired by [LLM4SGG](https://github.com/rlqja1107/torch-LLM4SGG), we leverage the three image caption datasets: [COCO caption](https://cocodataset.org/#download), [Conceptual (CC) caption](https://ai.google.com/research/ConceptualCaptions/) , and [VG](data/VG/README.md) caption to build the Text-Image pair-wise SG. 

  - Text-Video 

    To construct the text-video pairwise USG dataset, we select 400 videos from [ActivityNet](http://activity-net.org/), which includes dense caption annotations. 

  - Text-3D 

    To construct the text-3D pairwise USG dataset, we use the [ScanRefer](https://github.com/daveredrum/ScanRefer) dataset, which contains 46,173 descriptions of 724 object types across 800 [ScanNet](http://www.scan-net.org/) scenes. 

  - Image-Video

    To construct the image-video pairwise USG dataset, we utilize the existing [PVSG](data/PVSG/README.md) video dataset.

  - Image-3D
    To construct the Image-3D USG dataset, we leverage the existing [3DSG](data/3DSG/README.md) dataset. 


## Code
Coming soon.




## Citation

If you use USG in your project, please kindly cite:
```
@inproceedings{wu2025usg,
    title={Universal Scene Graph Generation},
    author={Wu, Shengqiong and Fei, Hao and Chua, Tat-Seng},
    booktitle={CVPR},
    year={2025}
}
```


# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

# DIPR-Net: A Dual-Branch Integrated Prototype Representation Network for Colonoscopy Polyp Segmentation

This repository contains the official PyTorch implementation for the paper "DIPR-Net: A Dual-Branch Integrated Prototype Representation Network for Colonoscopy Polyp Segmentation". A manuscript detailing the method and experimental results has been submitted to **Signal, Image and Video Processing (SIViP)**.

DIPR-Net is a unified framework designed for accurate and robust segmentation of polyps in colonoscopy images. It addresses the challenges of blurred boundaries and high morphological variability by synergistically integrating a hybrid feature encoder with a novel prototype-guided, dual-branch decoder. 

The core idea is to leverage a collaborative learning paradigm: a main segmentation branch focuses on precise spatial delineation, while a parallel prototype-guided branch enforces feature consistency and discrimination between polyps and surrounding tissue. This dual-path strategy, governed by a multi-objective loss function, compels the network to learn powerful and distinctive representations that excel at identifying polyps of varying shapes and sizes.

## Code Release Status

We are committed to making our research fully reproducible. The code will be released in two stages:

### Currently Available
The core implementation of the DIPR-Net architecture is now publicly available in this repository. This includes:
- **`model.py`:** The main network class that assembles all components.
- **`modules/` directory:** Detailed implementations of all sub-modules for the encoder, decoder, bottleneck, and prototype-guidance mechanism.
- **Dependencies:** The code relies on the `dynamic-network-architectures` library, which is included in this repository for convenience.

### To Be Released
Upon acceptance and publication of our manuscript, we will release the complete experimental framework, including:
- **Full Training and Inference Scripts:** Scripts to replicate our training process and run inference on new images.
- **Evaluation Code:** Code for calculating all performance metrics reported in the paper (mDice, mIo- **Data Preprocessing Scripts:** Scripts detailing the data preparation and augmentation steps.
- **Visualization Tools:** Code to generate qualitative results and heatmap visualizations.
- **Pre-trained Model Weights:** The model weights used to produce the results in our paper.

We believe that providing the complete codebase will be of great value to the community and facilitate future research in this area.

## Datasets

Our experiments were conducted on five publicly available polyp segmentation datasets. The training and testing splits follow the standard established in PraNet.

### Training Datasets
The training set comprises a total of 1,450 images:
- **Kvasir-SEG** [1]: 900 images from the Kvasir-SEG dataset. It contains images of varying resolutions, and each image has a corresponding ground truth mask.
- **CVC-ClinicDB** [2]: 550 images from the CVC-ClinicDB (also known as CVC-612) dataset. These are frames extracted from colonoscopy videos.

### Testing Datasets
The testing set contains a total of 798 images from five datasets, including both seen and unseen domains:
- **Kvasir-SEG** [1]: 100 images.
- **CVC-ClinicDB** [2]: 62 images.
- **CVC-ColonDB** [3]: 380 images from 15 short colonoscopy video sequences.
- **ETIS-LaribPolypDB** [3]: 196 images containing polyps with significant variations in shape and size.
- **EndoScene** [4]: 60 images from the CVC-300 dataset, also part of the Endocv2021 challenge dataset.

---
### References and Links

[1] **Kvasir-SEG**
    - **Paper:** Jha, D., Smedsrud, P. H., Riegler, M. A., et al. "Kvasir-seg: A segmented polyp dataset." *International conference on multimedia modeling.* (2020).
    - **Link:** [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)

[2] **CVC-ClinicDB**
    - **Paper:** Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., et al. "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians." *Computerized Medical Imaging and Graphics* 43 (2015).
    - **Link:** [https://www.kaggle.com/datasets/balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)

[3] **CVC-ColonDB & ETIS-LaribPolypDB**
    - **Paper:** Silva, J., Histace, A., Romain, O., et al. "Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer." *International journal of computer assisted radiology and surgery* 9.2 (2014).
    - **Links:**
        - CVC-ColonDB: [https://www.kaggle.com/datasets/longvil/cvc-colondb](https://www.kaggle.com/datasets/longvil/cvc-colondb)
        - ETIS-LaribPolypDB: [https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb)

[4] **EndoScene**
    - **Paper:** Vázquez, D., et al. "A benchmark for an endoluminal scene segmentation of colonoscopy images." *Journal of healthcare engineering* (2017).
    - **Link:** [https://service.tib.eu/ldmservice/dataset/endoscene](https://service.tib.eu/ldmservice/dataset/endoscene)

## Citation
If you find our work useful in your research, please consider citing our paper (citation details will be provided upon publication). You can also cite the datasets using the BibTeX entries below:

```bibtex
@inproceedings{jha2020kvasir,
  title={Kvasir-seg: A segmented polyp dataset},
  author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
  booktitle={International conference on multimedia modeling},
  pages={451--462},
  year={2020},
  organization={Springer}
}

@article{bernal2015wmdova,
  title={WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians},
  author={Bernal, Jorge and S{\'a}nchez, F Javier and Fern{\'a}ndez-Esparrach, Gloria and Gil, Debora and Rodr{\'\i}guez, Cristina and Vilarino, Fernando},
  journal={Computerized medical imaging and graphics},
  volume={43},
  pages={99--111},
  year={2015},
  publisher={Elsevier}
}

@article{silva2014embedded,
  title={Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer},
  author={Silva, Juan and Histace, Aymeric and Romain, Olivier and Dray, Xavier and Granado, Bertrand},
  journal={International journal of computer assisted radiology and surgery},
  volume={9},
  number={2},
  pages={283--293},
  year={2014},
  publisher={Springer}
}

@article{vazquez2017benchmark,
  title={A benchmark for endoluminal scene segmentation of colonoscopy images},
  author={V{\'a}zquez, David and Bernal, Jorge and S{\'a}nchez, F Javier and Fern{\'a}ndez-Esparrach, Gloria and L{\'o}pez, Antonio M and Romero, Adriana and Drozdzal, Michal and Courville, Aaron},
  journal={Journal of healthcare engineering},
  volume={2017},
  year={2017},
  publisher={Hindawi}
}

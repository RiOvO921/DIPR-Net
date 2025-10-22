# DIPR-Net: A Dual-Branch Integrated Prototype Representation Network for Colonoscopy Polyp Segmentation

This repository contains the official PyTorch implementation for the paper "DIPR-Net: A Dual-Branch Integrated Prototype Representation Network for Colonoscopy Polyp Segmentation". A manuscript detailing the method and experimental results has been submitted to **Signal, Image and Video Processing (SIViP)**.

DIPR-Net is a unified framework designed for accurate colonoscopy polyp segmentation. It synergistically integrates a hybrid encoder with a prototype-guided, dual-branch decoder to effectively balance fine-grained boundary details with global contextual understanding. The network architecture is designed with several key innovations:
- A novel **hybrid encoder** that uniquely allocates channels between parallel convolutional and transformer paths to capture both detailed local textures and long-range dependencies.
- A specialized **bottleneck module** that goes beyond simple multi-scale fusion by actively enhancing boundary-aware cues.
- A **dual-branch decoding strategy** where one branch performs selective spatial refinement of polyp regions, while a concurrent prototype-guided branch enforces feature discrimination and consistency. This collaborative design compels the shared encoder to learn more robust and distinctive features.

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
- **Evaluation Code:** Code for calculating all performance metrics reported in the paper (mDice, mIoU, etc.).
- **Data Preprocessing Scripts:** Scripts detailing the data preparation and augmentation steps.
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
**References and Links:**

[1] **Kvasir-SEG:** Jha, D., Smedsrud, P. H., Riegler, M. A., et al. "Kvasir-seg: A segmented polyp dataset." *International conference on multimedia modeling.* (2020). [Dataset Link](https://datasets.simula.no/kvasir-seg/)

[2] **CVC-ClinicDB:** Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., et al. "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians." *Computerized Medical Imaging and Graphics* 43 (2015). [Dataset Page](https://polyp.grand-challenge.org/CVCClinicDB/)

[3] **CVC-ColonDB & ETIS-LaribPolypDB:** Silva, J., Histace, A., Romain, O., et al. "Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer." *International journal of computer assisted radiology and surgery* 9.2 (2014).

[4] **EndoScene:** Vázquez, D., et al. "A benchmark for an endoluminal scene segmentation of colonoscopy images." *Journal of healthcare engineering* (2017). Part of the Endocv2021 Challenge. [Challenge Link](https://endocv2021.grand-challenge.org/)

## Citation
If you find our work useful in your research, please consider citing our paper (citation details will be provided upon publication).

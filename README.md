# CorMulT: Correlation-aware Multimodal Transformer for Unaligned Multimodal Language Sequences

This project implements a PyTorch version of the Correlation-aware Multimodal Transformer (CorMulT). Built as an improvement over previous transformer-based multimodal frameworks, CorMulT incorporates explicit modality correlation learning in a two-stage training process to boost performance in tasks like Multimodal Sentiment Analysis.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Paper & Citation](#paper--citation)
3. [CorMulT Framework](#cormult-framework)
4. [Usage](#usage)
5. [Datasets](#datasets)
6. [Running the Code](#running-the-code)
7. [Acknowledgement](#acknowledgement)

---

## Introduction

Multimodal Sentiment Analysis combines information from different modalities—such as text, video, and audio—to recognize and analyze emotions. In many real-world scenarios, the correlation among these modalities is neither strong nor perfectly aligned. To address this, our approach builds upon existing transformer-based models by introducing an explicit modality correlation learning mechanism.

CorMulT is designed with a two-stage training process:
- **Stage 1 – Modality Correlation Contrastive Learning:**  
  The model encodes features from different modalities and uses contrastive learning to construct positive and negative pairs. This stage learns modality correlation coefficients that quantify the similarity between any two modalities, enabling robust performance even when the data alignment is imperfect.
  
- **Stage 2 – Correlation-aware Transformer Training:**  
  The learned correlation coefficients are then used to weight and fuse the multimodal features. In this stage, a crossmodal transformer integrates the weighted features for sequential modeling, followed by a downstream classifier for sentiment prediction.

Experiments on popular datasets (such as CMU-MOSEI) demonstrate that CorMulT maintains high accuracy even in scenarios with weak modality correlations, significantly enhancing the overall performance in multimodal sentiment analysis tasks.

---

## Paper & Citation

For details on the model and experiments, please refer to our paper available on arXiv:

```
@misc{li2024cormultsemisupervisedmodalitycorrelationaware,
  title={CorMulT: A Semi-supervised Modality Correlation-aware Multimodal Transformer for Sentiment Analysis},
  author={Yangmin Li and Ruiqi Zhu and Wengen Li},
  year={2024},
  eprint={2407.07046},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2407.07046},
}
```

If you use CorMulT in your research or projects, please consider citing the paper.

---

## CorMulT Framework

The overall workflow of CorMulT is illustrated below:

1. **Stage 1 – Modality Correlation Contrastive Learning:**  
   - Encode initial features from each modality (video, audio, text).
   - Employ contrastive learning to generate positive and negative sample pairs.
   - Learn correlation coefficients to effectively capture the degree of similarity between modalities.

2. **Stage 2 – Correlation-aware Transformer Training:**  
   - Utilize the learned correlation coefficients to guide feature fusion.
   - Fuse the weighted features using a crossmodal transformer.
   - Predict sentiment through a downstream classifier.

This two-stage process allows the model to exploit explicit inter-modality dependencies, leading to better performance on unaligned multimodal sequences.

---

## Usage

The project setup is largely similar to that of existing multimodal transformer implementations. Below is a summary of the required environment and installation steps:

- **Environment Requirements:**
  - Python 3.10 (tested with Python 3.10.15)
  - PyTorch (>=1.0.0) and TorchVision
  - CUDA 12.0 or later (if GPU usage is needed)

- **Installation (example):**

  ```bash
  conda create -n cormult_env python=3.10
  conda activate cormult_env
  pip install -r requirements.txt
  ```

Adjust the environment settings as needed.

---

## Datasets

We use datasets such as CMU-MOSEI for multimodal sentiment analysis. Preprocessed versions of MOSI, MOSEI, and IEMOCAP can be downloaded from the CMU-MultimodalSDK repository. Please refer to https://github.com/ecfm/CMU-MultimodalSDK for more details.

---

## Running the Code

The project is organized with the following structure:

```
.
└── CorMulT
    ├── LICENSE
    ├── README.md
    ├── imgs
    ├── main.py
    ├── modality_correlation
    │   ├── main_correlation.py
    │   └── ...
    ├── modules
    │   └── ...
    ├── pre_trained_models
    │   └── ...
    └── src
        └── ...
```

The training process is divided into two stages:

1. **Stage 1 – Pretraining for Modality Correlation:**  
   Run the following command to start the modality correlation pretraining:

   ```bash
   python modality_correlation/main_correlation.py
   ```

2. **Stage 2 – Full Training:**  
   After Stage 1, execute the following to run the complete training with the learned correlation coefficients:

   ```bash
   python main.py [--FLAGS]
   ```

   The `[--FLAGS]` can include various hyperparameters such as dataset selection, number of training epochs, batch size, etc. The default settings are tailored for unaligned MOSEI data; adjust the parameters as needed for other datasets.

During Stage 1, the model saves the learned correlation evaluation weights to the `pre_trained_models/` directory. These weights are then used in Stage 2 to guide the training of the full multimodal transformer.

---

## Acknowledgement

We thank all the contributors of the open-source projects that served as inspiration for this work. If you use CorMulT in your research or projects, please cite the corresponding paper and acknowledge the original ideas.
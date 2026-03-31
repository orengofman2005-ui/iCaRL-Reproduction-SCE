# iCaRL: Incremental Classifier and Representation Learning - Reproduction Project

## Project Overview

This repository contains a full reproduction of the paper ["iCaRL: Incremental Classifier and Representation Learning" (Rebuffi et al., CVPR 2017)](https://arxiv.org/pdf/1611.07725). The focus of this implementation is to evaluate the model's ability to handle Class-Incremental Learning (CIL) and mitigate the effects of Catastrophic Forgetting.

## Technical Methodology

*   **Dataset**: CIFAR-100, consisting of 100 distinct object classes.
*   **Incremental Protocol**: The dataset is partitioned into 10 sequential tasks, with 10 new classes introduced in each phase.
*   **Architecture**: Implementation based on the PyCIL framework (Python Class-Incremental Learning) using a ResNet32 backbone.
*   **Optimization**: Training is performed on a local NVIDIA GPU with specific CUDA device configurations.

## Execution Guide

1.  Install the required environment:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the training process:
    ```bash
    python main.py --config=./exps/icarl.json
    ```

## Experimental Results

Quantitative results, including Top-1 Accuracy across incremental steps and comparison graphs with the original paper, will be populated here upon execution completion.

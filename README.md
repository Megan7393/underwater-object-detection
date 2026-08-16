# Underwater Object Detection Project

# Overview

This project investigates the use of transfer learning and pre-trained deep learning models for underwater marine organism detection. Three object detection architectures representing different methodological approaches were fine-tuned on underwater imagery and compared using standard object detection performance metrics.This repository contains the code, model configurations, evaluation scripts and supporting files developed as part of the research project.

The study compares:
YOLOv8n (one stage detector)
Faster R-CNN ResNet50-FPN ( two-stage object detector)
DETR ResNet-50  (transformer based object detection)
All three models were trained and evaluated using the DUO underwater object detection dataset. Faster R-CNN was also independently fine-tuned and evaluated on the RUOD dataset to assess its performance on a larger and more diverse underwater detection dataset.

# Aim

The aim of this project is to evaluate and compare different deep learning object detection architectures for underwater marine organism detection, with particular emphasis on detection accuracy, class-specific performance and computational efficiency. The study also considers the practical trade-offs between model accuracy and inference speed to assess the suitability of each architecture for different marine monitoring applications.

# Objectives
-Prepare and explore publicly available underwater object detection datasets.
-Fine-tune YOLOv8n, Faster R-CNN and DETR using transfer learning.
-Evaluate the models using standard object detection metrics.
-Compare overall and per-class detection performance.
-Compare inference speed across architectures.
-Investigate the relationship between class imbalance and model performance.
-Evaluate Faster R-CNN on a second underwater dataset, RUOD.
-Assess the practical suitability of each architecture for marine monitoring applications.

# Dataset descriptions:

# DUO
The DUO dataset was used as the primary dataset for architecture comparison. 
The dataset contains four object classes:
Holothurian
Echinus
Scallop
Starfish

# RUOD
The RUOD dataset was used as a secondary evaluation dataset for Faster R-CNN.
It contains ten underwater object classes:
Holothurian
Echinus
Scallop
Starfish
Fish
Corals
Diver
Cuttlefish
Turtle
Jellyfish

# Transfer Learning

All models were initialised using pretrained weights and subsequently fine-tuned on the underwater datasets. Transfer learning reduced the computational requirements associated with training deep neural networks from scratch and allowed the models to reuse general visual features learned from large-scale image datasets.

# Evaluation Metrics

The models were evaluated using the following metrics:
mAP@0.5 - Mean Average Precision calculated using an Intersection over Union threshold of 0.5.
mAP@0.5:0.95 - Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 in increments of 0.05.
Per-class AP - Per-Class Average Precision
FPS = Frames per Second

# Libraries used

Python
PyTorch
Torchvision
Ultralytics
Hugging Face Transformers
pandas
NumPy
Matplotlib
COCO evaluation utilities

# Hardware

The experiments were conducted using CPU-based hardware.This imposed limitations on: training duration, model size, batch size, hyperparameter optimisation etc.
(hese constraints should be considered when interpreting the results, particularly the performance of DETR).

# Folder structure

underwater-object-detection/
  data/
    raw/
    processed/
    test_images/
  figures/
  models/
  notebooks/
  results/
  runs/
  src/
  venv/
  README.md

# Environment setup and how to run project

The project was developed using Python 3.13.7 in a CPU-based environment. To reproduce the project, the repository should be cloned and a Python virtual environment created before installing the required dependencies.The DUO and RUOD datasets are not included within the repository and should be downloaded separately from their original sources before being placed within the appropriate data/raw/ directories. The project workflow consists of preprocessing the datasets, creating the required training and validation splits, training the selected object detection architecture, selecting the best-performing validation checkpoint, and evaluating the final model on the corresponding official test set. Separate scripts are provided for YOLOv8n, Faster R-CNN ResNet50-FPN and DETR ResNet-50, with additional scripts for the independent Faster R-CNN evaluation on RUOD. Generated model checkpoints, training histories, evaluation metrics and figures are stored within the relevant results/ directories. Exact script names, paths and model configurations are documented within the repository and should be followed when reproducing individual experiments.


# Results summary

Faster R-CNN ResNet50-FPN achieved the highest overall DUO detection accuracy, with mAP@0.5 = 0.6949 and mAP@0.5:0.95 = 0.4721.
YOLOv8n achieved slightly lower accuracy, with mAP@0.5 = 0.6542 and mAP@0.5:0.95 = 0.4456, but was substantially faster at 54.60 FPS.
DETR ResNet-50 achieved the lowest overall DUO performance, with mAP@0.5 = 0.4659, mAP@0.5:0.95 = 0.2359, and 6.84 FPS.
Faster R-CNN operated at 1.88 FPS, highlighting the trade-off between detection accuracy and inference speed.
Echinus was the strongest-performing DUO class across all three models, while scallop consistently achieved the lowest Average Precision.
Faster R-CNN was independently fine-tuned and evaluated on RUOD, achieving mAP@0.5 = 0.7546 and mAP@0.5:0.95 = 0.4571.
Overall, Faster R-CNN provided the strongest detection accuracy, while YOLOv8n offered the best balance between accuracy and computational efficiency.
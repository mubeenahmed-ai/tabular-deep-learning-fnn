# Deep Learning for Tabular Data: FNN Architecture & Generalization Analysis

This repository contains a structured implementation of a Feedforward Neural Network (FNN) optimized for multi-class classification on high-dimensional tabular feature sets. The project highlights foundational deep learning concepts, feature engineering analysis, and rigorous verification of model generalization capabilities.

## Key Features
* **Custom FNN Pipeline: ** Architecture designed specifically to process heterogeneous tabular data, applying dense layers, activation functions, and regularization techniques.
* **Multi-Class Classification:** Network optimized to predict categorical outcomes across 4 distinct target classes using cross-entropy evaluation.
* **Generalization Monitoring:** Comprehensive tracking of training vs. testing performance metrics over extended epochs to map convergence behaviors and catch latent overfitting patterns.
* **Feature Distribution Auditing:** Built-in verification steps to analyze and print out statistical markers (e.g., means, feature counts) across train/test splits, ensuring consistent data pipelines.

## Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
* **Methodologies:** Multi-Layer Perceptrons (MLPs), Feature Alignment, Evaluation Metrics, Generalization Mapping.

## Analytical Insights & Performance
* **Pipeline Soundness:** Validated feature stability across target distributions (e.g., monitoring key metric footprints like `amenity_score` variance) to guarantee model inputs match across deployment boundaries.
* **Overfitting Diagnostics:** Generated custom generalization analysis visualizations (`partD_generalisation.png`) to measure exact testing delta paths against training loss stabilization points.
* **Optimization Takeaways:** Evaluated how layer structural depth configurations and scaling behavior affect convergence reliability when operating on densely packed continuous variables.

## Project Structure
* `fnn-tabular-classification-analysis.ipynb`: Complete self-contained notebook containing data loading workflows, neural network topologies, validation metrics, and analytical plots.

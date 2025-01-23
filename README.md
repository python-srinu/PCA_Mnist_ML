# MNIST Dimensionality Reduction with PCA

This project demonstrates how to use Principal Component Analysis (PCA) for dimensionality reduction on the MNIST dataset, followed by evaluating a machine learning model's performance before and after applying PCA.

## Project Overview

The goal of this project is to:
1. Load the MNIST dataset, a collection of handwritten digits represented as 28x28 grayscale images.
2. Preprocess the data by normalizing pixel values and standardizing the dataset.
3. Train a Random Forest Classifier on the original high-dimensional data.
4. Apply PCA to reduce the dimensionality of the dataset while retaining 95% of the variance.
5. Train the Random Forest Classifier on the reduced-dimensional data and compare its performance to the original model.

## Steps Performed

1. **Dataset Loading**: 
   - The MNIST dataset is loaded using TensorFlow/Keras. 
   - The dataset is split into training, validation, and test sets.

2. **Data Preprocessing**:
   - The images are reshaped from 28x28 matrices to 784-dimensional vectors.
   - Pixel values are normalized to the range [0, 1].
   - StandardScaler is applied to standardize the data for better performance in PCA and classification.

3. **Model Training (Before PCA)**:
   - A Random Forest Classifier is trained on the full 784-dimensional dataset.
   - The model's performance is evaluated using the test set.

4. **Dimensionality Reduction with PCA**:
   - PCA is applied to reduce the dataset's dimensionality while retaining 95% of the variance.
   - This step significantly reduces the number of features, improving computational efficiency.

5. **Model Training (After PCA)**:
   - The Random Forest Classifier is retrained on the reduced-dimensional dataset.
   - The model's performance is evaluated and compared with the original high-dimensional model.

6. **Visualization**:
   - A plot of the cumulative explained variance is generated to show how much variance is retained with the selected number of components.

## Requirements

The project uses the following libraries:
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For PCA, classification, and evaluation.
- **TensorFlow/Keras**: To load the MNIST dataset.

## Files in the Repository

- **`PCA_ML.ipynb`**: The main Python script containing the entire implementation.

## Key Features

- **Dimensionality Reduction**: Demonstrates the use of PCA for reducing high-dimensional datasets while retaining significant variance.
- **Model Performance Comparison**: Compares the performance of a Random Forest Classifier before and after applying PCA.
- **Explained Variance Visualization**: Provides visual insights into the cumulative explained variance retained by PCA, helping to understand the effectiveness of the reduction.


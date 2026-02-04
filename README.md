# DDoS-Attack-Detection

Applications of Machine Learning algorithms for DDoS Attack Detection and Prevention

## Overview

This project implements and compares multiple machine learning algorithms to detect and classify Distributed Denial of Service (DDoS) attacks. The project uses network traffic data to train and evaluate various ML models including neural networks, decision trees, KNN, logistic regression, naive bayes, and CNNs.

## Dataset

The project utilizes network traffic flow data from the CSE 543 dataset, which includes both normal network traffic and various types of DDoS attacks. The dataset contains features extracted from network packet captures including:
- Flow statistics (duration, inter-arrival times, etc.)
- Packet flags and counts
- Protocol information
- Byte and packet rates
- Header information

## Machine Learning Models Implemented

1. **Artificial Neural Networks (ANN)** - [ann.ipynb](ann.ipynb)
2. **Convolutional Neural Networks (CNN)** - [cnn.ipynb](cnn.ipynb)
3. **Decision Trees** - [decision_tree.ipynb](decision_tree.ipynb)
4. **K-Nearest Neighbors (KNN)** - [knn.ipynb](knn.ipynb)
5. **Logistic Regression** - [logistic_regression.ipynb](logistic_regression.ipynb)
6. **Naive Bayes** - [naive_bayes.ipynb](naive_bayes.ipynb)

## Methodology

### Data Preprocessing
The preprocessing pipeline includes:
- Data cleaning and normalization
- Feature selection and engineering
- Handling missing values
- Scaling and standardization
- Train-test split for model validation

See [preprocessing.ipynb](preprocessing.ipynb) for detailed preprocessing steps.

### Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Structure

```
DDoS-Attack-Detection/
├── preprocessing.ipynb          # Data preprocessing and feature engineering
├── ann.ipynb                     # Artificial Neural Network implementation
├── cnn.ipynb                     # Convolutional Neural Network implementation
├── decision_tree.ipynb           # Decision Tree classifier
├── knn.ipynb                     # K-Nearest Neighbors classifier
├── logistic_regression.ipynb     # Logistic Regression classifier
├── naive_bayes.ipynb             # Naive Bayes classifier
├── README.md                     # This file
└── Project Report/
    └── CSE543_Paper.pdf          # Detailed research paper
```

## Results and Findings

The comparative analysis of different ML algorithms provides insights into:
- Model accuracy and performance metrics
- Trade-offs between different algorithms
- Computational efficiency
- Real-time detection capabilities

Detailed results and discussions are available in the [Project Report](Project%20Report/CSE543_Paper.pdf).

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- TensorFlow/Keras (for neural networks)
- matplotlib
- seaborn

## Usage

1. Ensure all dependencies are installed
2. Run preprocessing.ipynb first to prepare the dataset
3. Execute individual model notebooks to train and evaluate each algorithm
4. Compare results across different models

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided for educational purposes.

## References

For detailed methodology, results, and analysis, refer to the [CSE543_Paper.pdf](Project%20Report/CSE543_Paper.pdf) in the Project Report folder.

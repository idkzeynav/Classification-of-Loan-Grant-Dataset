# Loan Grant Classification - Machine Learning Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Overview

This project implements a comprehensive machine learning classification system to predict loan grant approvals using a dataset of 111,000+ loan applications. Six different supervised learning algorithms are trained, evaluated, and compared to identify the most effective model for loan grant classification.

## 🎯 Project Objective

To develop and compare multiple machine learning models that can accurately predict whether a loan grant application will be approved or rejected, helping financial institutions make data-driven lending decisions.

## 📊 Dataset

- **File**: `loangrant.csv`
- **Size**: 111,000+ records
- **Features**: Multiple demographic, financial, and application-related attributes
- **Target Variable**: Loan grant approval status (Binary classification)
- **Documentation**: See `LoanGrant_Data_Dictionary.docx` for detailed feature descriptions

## 🔧 Methodology

### Data Preprocessing
- ✅ Missing value imputation
- ✅ Outlier detection and treatment using IQR method
- ✅ Duplicate record removal
- ✅ Categorical variable encoding
- ✅ Feature scaling and normalization
- ✅ 80-20 stratified train-test split

### Machine Learning Models Implemented

1. **Decision Tree Classifier**
2. **Bagging Classifier**
3. **AdaBoost (Boosting)**
4. **Random Forest Classifier**
5. **K-Nearest Neighbors (KNN)**
6. **Support Vector Machine (SVM - Linear Kernel)**

### Performance Optimization

All models were optimized for large-scale data processing:
- Parallel processing enabled (`n_jobs=-1`)
- Reduced ensemble sizes for faster training
- Linear SVM kernel for computational efficiency
- Vectorized operations for outlier detection
- Strategic hyperparameter tuning

**Total Training Time**: ~44 seconds for all 6 models

## 📈 Results

### Best Performing Model: Random Forest 🏆

**Testing Set (20%) Performance:**
- **Accuracy**: 74.18%
- **ROC AUC**: 0.7855
- **MCC**: 0.2732
- **Sensitivity**: 92.28%
- **Specificity**: 28.51%

### Complete Performance Comparison

#### Training Set (80%)
| Model | Accuracy | Sensitivity | Specificity | MCC | ROC AUC |
|-------|----------|-------------|-------------|-----|---------|
| Decision Tree | 75.37% | 89.10% | 40.74% | 0.3405 | 0.8094 |
| **Bagging** | **100%** | **100%** | **100%** | **1.0000** | **1.0000** |
| AdaBoost | 73.22% | 95.54% | 16.90% | 0.2069 | 0.7815 |
| Random Forest | 75.67% | 93.00% | 31.93% | 0.3232 | 0.8182 |
| KNN | 84.40% | 90.75% | 68.37% | 0.6074 | 0.9081 |
| SVM | 52.13% | 57.44% | 38.73% | -0.0350 | 0.4682 |

#### Testing Set (20%)
| Model | Accuracy | Sensitivity | Specificity | MCC | ROC AUC |
|-------|----------|-------------|-------------|-----|---------|
| Decision Tree | 72.81% | 87.31% | 36.24% | 0.2687 | 0.7728 |
| Bagging | 72.08% | 84.98% | 39.52% | 0.2669 | 0.7724 |
| AdaBoost | 73.60% | 95.68% | 17.88% | 0.2234 | 0.7746 |
| **Random Forest** | **74.18%** | 92.28% | 28.51% | **0.2732** | **0.7855** |
| KNN | 70.52% | 81.18% | 43.63% | 0.2555 | 0.7112 |
| SVM | 52.07% | 58.46% | 35.94% | -0.0516 | 0.4610 |

### Key Insights

- **Random Forest** achieved the best generalization with highest test accuracy and ROC AUC
- **Bagging** showed perfect training performance but moderate test performance (overfitting indicator)
- **AdaBoost** achieved highest sensitivity (95.68%) - best for minimizing false negatives
- **SVM** underperformed, suggesting linear kernel limitations for this dataset
- Most models showed high sensitivity but lower specificity, indicating bias toward approval predictions

## 📁 Generated Outputs

| File | Description |
|------|-------------|
| `confusion_matrices_training_80.png` | Visual confusion matrices for all models (training data) |
| `confusion_matrices_testing_20.png` | Visual confusion matrices for all models (testing data) |
| `roc_curves_comparison.png` | ROC curve comparison across all models |
| `training_80_performance.csv` | Detailed performance metrics (training set) |
| `testing_20_performance.csv` | Detailed performance metrics (testing set) |

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Analysis
```python
# Load and run the main script
python loan_classification.py
```

## 📋 Project Structure
```
loan-grant-classification/
│
├── loangrant.csv                          # Dataset
├── LoanGrant_Data_Dictionary.docx         # Feature documentation
├── Tasks.docx                             # Project requirements
├── loan_classification.py                 # Main analysis script
├── README.md                              # This file
│
└── outputs/                               # Generated results
    ├── confusion_matrices_training_80.png
    ├── confusion_matrices_testing_20.png
    ├── roc_curves_comparison.png
    ├── training_80_performance.csv
    └── testing_20_performance.csv
```

## 📊 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Sensitivity (Recall)**: True positive rate - correctly identified approvals
- **Specificity**: True negative rate - correctly identified rejections
- **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets
- **ROC AUC**: Area under the receiver operating characteristic curve

## 🔍 Future Improvements

- [ ] Feature engineering and selection
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Address class imbalance with SMOTE or class weights
- [ ] Deep learning approaches (Neural Networks)
- [ ] Ensemble stacking methods
- [ ] Cost-sensitive learning implementation
- [ ] Cross-validation analysis

## 📝 Requirements Completed

✅ Dataset loading and analysis  
✅ Complete data preprocessing pipeline  
✅ Implementation of 6 ML algorithms  
✅ 80-20 stratified train-test split  
✅ Confusion matrix generation  
✅ Comprehensive performance metrics  
✅ ROC curve analysis  
✅ Performance optimization for large datasets  

## 👤 Author

*idkzeynav*

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project was developed as part of a machine learning classification assignment focused on real-world financial data analysis.

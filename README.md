# 🛒 Predicting Online Shopper Conversion
**A Comparative Analysis of Logistic Regression and Gaussian Naive Bayes for E-commerce Optimization**

---

## 🌟 Project Overview
This project focuses on predicting whether an online shopper will complete a purchase based on their browsing behavior. By analyzing the **Online Shoppers Purchasing Intention Dataset**, we compared the performance of two models: **Logistic Regression (LR)** and **Gaussian Naive Bayes (GNB)**. The goal is to optimize predictions, enhance e-commerce decision-making, and improve customer targeting.

---

## 📊 Dataset Overview
- **Dataset**: Online Shoppers Purchasing Intention Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset).
- **Size**: 12,330 rows, 17 features.
- **Target Variable**: `Revenue` (1 = Purchase, 0 = No Purchase).
- **Features**:
  - **Numerical**: Administrative duration, Bounce rates, Exit rates, Product-related views, Page values, etc.
  - **Categorical**: Visitor type, Month, Weekend.
- **Class Imbalance**: Only ~15% of shoppers make a purchase, requiring special handling.

---

## ⚙️ Methods and Preprocessing
### Preprocessing Steps
1. **Encoding**: One-hot encoding for categorical features like `Month` and `VisitorType`.
2. **Normalization**: All numerical features were normalized to have zero mean and unit variance.
3. **Outlier Removal**: Top 2% of extreme values were removed.
4. **Class Imbalance Handling**:
   - Oversampling (SMOTE and random sampling).
   - Undersampling.
   - Weighted Logistic Regression.

### Models
1. **Logistic Regression (LR)**:
   - Interpretable linear model with coefficients that reflect feature importance.
   - Optimized with hyperparameter tuning (C, penalty, solver).
2. **Gaussian Naive Bayes (GNB)**:
   - Probabilistic model assuming feature independence.
   - Simple, efficient, and handles imbalanced datasets naturally.

---

## 🔍 Results
### Performance Metrics
| Metric          | Logistic Regression (Optimized) | Gaussian Naive Bayes (Optimized) |
|-----------------|---------------------------------|----------------------------------|
| Accuracy        | 0.87537                         | 0.8194                           |
| Precision       | 0.56803                         | 0.4413                           |
| Recall          | 0.76241                         | 0.6933                           |
| F1-Score        | 0.65102                         | 0.5393                           |
| AUC             | 0.89776                         | 0.8415                           |

### Visualizations
- **ROC Curve Comparison**: [Add your graph here].
- **Confusion Matrices**: [Add your matrices here].
- **Feature Importance (LR Coefficients)**: [Add your graph here].
- **Box Plots**: Show distributions of numerical features.

---

## 🧠 Key Insights
- **Logistic Regression** achieved higher AUC and interpretability, making it ideal for understanding feature importance.
- **Naive Bayes** excelled in recall, making it suitable for identifying minority classes (purchases).
- **Feature Engineering**: Removing redundant features (e.g., `Browser`, `OperatingSystems`) and normalizing data significantly improved model performance.

---

## 🚀 Future Improvements
1. Explore non-linear models like Random Forest, XGBoost, or Neural Networks.
2. Add external features (e.g., promotions, holidays).
3. Use advanced oversampling techniques like **SMOTE** for better class balancing.
4. Combine models into an ensemble for improved predictions.

---

##🙏 Acknowledgments
Dataset from the UCI Machine Learning Repository.
Libraries: Scikit-learn, NumPy, Matplotlib, Pandas

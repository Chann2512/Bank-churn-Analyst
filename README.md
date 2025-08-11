# Bank-churn-Analyst
Built a predictive model for customer credit card churn
## 📌 Credit Card Customer Churn Prediction

**Objective:** Predict the likelihood of a customer leaving a credit card service, enabling proactive retention strategies and operational integration.

**Workflow:**
1. **Data Preparation**  
   - Dataset: 10,000 customers with 23 demographic & behavioral features.  
   - Preprocessing: outlier removal, categorical encoding (Ordinal & One-Hot), class balancing (SMOTE + Tomek Links), normalization.  
   - Dimensionality reduction & feature selection with PCA and RFE.
   
2. **Model Training & Evaluation**  
   - Trained Logistic Regression, KNN, Random Forest, and XGBoost models with K-Fold Cross-Validation.  
   - Selected XGBoost with RFE + SMOTE as the optimal model.

3. **Model Packaging & Deployment**  
   - Built a **scikit-learn Pipeline** encapsulating preprocessing, feature selection, and the trained classifier.  
   - Saved the packaged model as a `.pkl` file using `joblib`, enabling consistent predictions in production without repeating manual preprocessing steps.  
   - The packaged model can be integrated into applications or APIs to automatically generate churn probabilities for new customers.

**Results:**
- **Accuracy:** 95.18%  
- **Precision:** 95.20%  
- **Recall:** 95.18%  
- **F1-score:** 95.17%  
- High ROC-AUC and stable learning curve with no overfitting.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE, Tomek Links), PCA, RFE, Matplotlib, Seaborn, Joblib.

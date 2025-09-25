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

## 💡 Actionable Business Recommendations

This project goes beyond a predictive model by providing a clear, data-driven framework for business strategy. The model's output—the churn probability of each customer—can be directly translated into targeted business actions:

**Proactive Customer Retention:**

- **High-Risk Customers (Churn Probability > 70%)**: Flag these customers for immediate, personalized outreach from a dedicated customer relationship manager. Offer high-value incentives like annual fee waivers or credit limit increases to address specific pain points.

- **Medium-Risk Customers (30% - 70%)**: Launch automated, targeted marketing campaigns. Use channels like email or in-app notifications to highlight the benefits of their card (e.g., reward points, exclusive offers) and re-engage them.

- **Low-Risk Customers (< 30%)**: Maintain strong relationships through regular communication and loyalty programs. Focus on cross-selling new products and services to increase their lifetime value.

# Product and Policy Optimization:

- Analyze the most influential features identified by the model (e.g., Avg_Utilization_Ratio, Credit_Limit). If certain features are highly predictive of churn, it signals a need to re-evaluate product policies, such as credit limit offerings or fee structures.

**Example**: If customers with a high Avg_Utilization_Ratio tend to churn, the bank might consider offering a limited-time credit limit increase to incentivize them to stay.

# Strategic Resource Allocation:

- By focusing retention efforts on the most at-risk customers, the company can optimize marketing spend and human resources. Instead of applying costly, one-size-fits-all campaigns, resources are strategically directed where they have the highest potential return on investment.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE, Tomek Links), PCA, RFE, Matplotlib, Seaborn, Joblib.


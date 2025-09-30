# Data Analysis of the 2020 WUSTL EHMS Dataset for IoMT Cybersecurity

## Abstract 📄

[span_1](start_span)This project focuses on the analysis of the **2020 WUSTL EHMS dataset**, a critical resource for research in the cybersecurity of the Internet of Medical Things (IoMT)[span_1](end_span). [span_2](start_span)The data is sourced from an Enhanced Health Monitoring System (EHMS) that collects patient information in real-time[span_2](end_span). [span_3](start_span)Our primary objective is to identify the most effective methods for processing and classifying this data[span_3](end_span).

**[span_4](start_span)Keywords**: 2020 WUSTL EHMS, IoMT Cybersecurity, Enhanced Health Monitoring System, Data Classification[span_4](end_span)

---

## Table of Contents 📋

* [Introduction](#introduction-)
* [Comparison of Prediction Methods](#comparison-of-prediction-methods-)
* [Proposed Methodology](#proposed-methodology-)
* [Best Algorithm Analysis](#best-algorithm-analysis-)
* [Conclusion & Future Work](#conclusion--future-work-)
* [References](#references-)

---

## Introduction 💡

[span_5](start_span)Data analysis is a fundamental tool for extracting valuable insights from data[span_5](end_span). [span_6](start_span)This project delves into a dataset containing information about attacks and the medical status of patients[span_6](end_span). [span_7](start_span)The goal is to leverage machine learning models to predict and identify these attacks within healthcare data[span_7](end_span). [span_8](start_span)[span_9](start_span)The timely detection of such threats is paramount in medical and hospital environments, as it can be a life-saving intervention[span_8](end_span)[span_9](end_span).

### Project Objectives

[span_10](start_span)The core aim is to design and implement machine learning models capable of predicting attacks and medical emergencies from health data[span_10](end_span). Key steps in this process include:
* **[span_11](start_span)Data Preprocessing**[span_11](end_span)
* **[span_12](start_span)Anomaly Detection**[span_12](end_span)
* **[span_13](start_span)Model Evaluation**[span_13](end_span)

[span_14](start_span)We will utilize tools like **LazyPredict** for an initial comparison and evaluate the final models based on metrics such as **Accuracy** and **F1-Score**[span_14](end_span).

### Importance and Application

[span_15](start_span)The ability to predict medical attacks is of immense importance, particularly in critical care and hospital settings[span_15](end_span). [span_16](start_span)This project aims to provide a framework that assists researchers and medical professionals in detecting threats more rapidly and accurately, thereby enabling prompt and effective treatment[span_16](end_span).

---

## Comparison of Prediction Methods 📊

[span_17](start_span)Predictive analytics plays a cornerstone role in modern healthcare, aiding in everything from disease prevention to resource management[span_17](end_span). [span_18](start_span)The accuracy of these predictions can significantly influence clinical decisions[span_18](end_span). Here's a comparison of several common machine learning algorithms:

* [span_19](start_span)Logistic Regression[span_19](end_span)
* [span_20](start_span)Decision Tree[span_20](end_span)
* [span_21](start_span)Random Forest[span_21](end_span)
* [span_22](start_span)Support Vector Machines (SVM)[span_22](end_span)
* [span_23](start_span)Artificial Neural Networks (ANN)[span_23](end_span)
* [span_24](start_span)Gradient Boosting[span_24](end_span)
* [span_25](start_span)K-Nearest Neighbors (KNN)[span_25](end_span)

| Criteria | Logistic Regression | Decision Tree | Random Forest | SVM | ANN | Gradient Boosting | KNN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Implementation Simplicity**| [span_26](start_span)Relatively Simple[span_26](end_span) | [span_27](start_span)Very Simple[span_27](end_span) | [span_28](start_span)More Complex[span_28](end_span) | [span_29](start_span)More Complex[span_29](end_span) | [span_30](start_span)Complex[span_30](end_span) | [span_31](start_span)More Complex[span_31](end_span) | [span_32](start_span)Simple[span_32](end_span) |
| **Interpretability** | [span_33](start_span)High[span_33](end_span) | [span_34](start_span)High[span_34](end_span) | [span_35](start_span)Medium[span_35](end_span) | [span_36](start_span)Low[span_36](end_span) | [span_37](start_span)Low[span_37](end_span) | [span_38](start_span)Medium[span_38](end_span) | [span_39](start_span)Low[span_39](end_span) |
| **Handling Large Data** | [span_40](start_span)Limited[span_40](end_span) | [span_41](start_span)Medium[span_41](end_span) | [span_42](start_span)Good[span_42](end_span) | [span_43](start_span)Good[span_43](end_span) | [span_44](start_span)Very Good[span_44](end_span) | [span_45](start_span)Very Good[span_45](end_span) | [span_46](start_span)Limited[span_46](end_span) |
| **Resistance to Overfitting**| [span_47](start_span)Low[span_47](end_span) | [span_48](start_span)Low[span_48](end_span) | [span_49](start_span)High[span_49](end_span) | [span_50](start_span)Medium[span_50](end_span) | [span_51](start_span)Low[span_51](end_span) | [span_52](start_span)Very High[span_52](end_span) | [span_53](start_span)Low[span_53](end_span) |
| **Prediction Accuracy** | [span_54](start_span)Medium[span_54](end_span) | [span_55](start_span)Lower[span_55](end_span) | [span_56](start_span)High[span_56](end_span) | [span_57](start_span)High[span_57](end_span) | [span_58](start_span)Very High[span_58](end_span) | [span_59](start_span)Very High[span_59](end_span) | [span_60](start_span)Medium[span_60](end_span) |
| **Training Speed** | [span_61](start_span)Very Fast[span_61](end_span) | [span_62](start_span)Fast[span_62](end_span) | [span_63](start_span)Faster[span_63](end_span) | [span_64](start_span)Slower[span_64](end_span) | [span_65](start_span)Slow[span_65](end_span) | [span_66](start_span)Slower[span_66](end_span) | [span_67](start_span)Very Fast[span_67](end_span) |
| **Need for Parameter Tuning**| [span_68](start_span)Low[span_68](end_span) | [span_69](start_span)Low[span_69](end_span) | [span_70](start_span)Medium[span_70](end_span) | [span_71](start_span)High[span_71](end_span) | [span_72](start_span)High[span_72](end_span) | [span_73](start_span)High[span_73](end_span) | [span_74](start_span)Low[span_74](end_span) |
| **Suitability for Non-linear Data** | [span_75](start_span)Weak[span_75](end_span) | [span_76](start_span)Medium[span_76](end_span) | [span_77](start_span)Good[span_77](end_span) | [span_78](start_span)Very Good[span_78](end_span) | [span_79](start_span)Very Good[span_79](end_span) | [span_80](start_span)Very Good[span_80](end_span) | [span_81](start_span)Weak[span_81](end_span) |


---

## Proposed Methodology 🛠️

### Data Preprocessing

This is a vital first step. The dataset contains missing values and anomalies that must be addressed:
* **[span_82](start_span)Handling Missing Values**: We use the **mean** for numerical columns and the **mode** for non-numerical ones[span_82](end_span).
* **[span_83](start_span)Replacing Zeros**: To avoid computational issues, zero values are replaced with the column's mean[span_83](end_span).
* **[span_84](start_span)Data Normalization**: We apply **MinMaxScaler** to normalize the feature set[span_84](end_span).

### Feature Selection

We carefully select features relevant to identifying attacks. **[span_85](start_span)One-Hot Encoding** is used for categorical variables to improve model performance, and features with low correlation to the target variable are removed[span_85](end_span).

### Modeling and Evaluation

We employ a range of machine learning models for prediction:
* **[span_86](start_span)LazyPredict**: For a quick, high-level comparison[span_86](end_span).
* **[span_87](start_span)Random Forest**[span_87](end_span)
* **[span_88](start_span)Logistic Regression**[span_88](end_span)
* **[span_89](start_span)SVM**[span_89](end_span)

[span_90](start_span)The performance of these models is rigorously evaluated using metrics like **Accuracy**, **F1-Score**, **Precision**, and **Recall**[span_90](end_span).

### Anomaly Detection

[span_91](start_span)To enhance model performance, we use **Isolation Forest** and **One-Class SVM** to identify and remove outliers from the training data[span_91](end_span).

---

## Best Algorithm Analysis 🏆

Our analysis revealed that several models performed exceptionally well.

1.  **Random Forest Classifier**:
    * **[span_92](start_span)Accuracy**: Achieved an impressive **0.9963**[span_92](end_span).
    * **[span_93](start_span)Performance**: Showed high precision, recall, and f1-scores for all classes[span_93](end_span). [span_94](start_span)The confusion matrix confirmed a low error rate[span_94](end_span).
    * **[span_95](start_span)ROC Curve**: The Area Under the Curve (AUC) was close to 1, demonstrating excellent class separation capability[span_95](end_span).

2.  **AdaBoost Classifier**:
    * **[span_96](start_span)Accuracy**: Reached a perfect **1.0**[span_96](end_span).
    * **[span_97](start_span)Performance**: The confusion matrix and ROC curve reflected this perfect score[span_97](end_span).

3.  **Gradient Boosting Classifier**:
    * **[span_98](start_span)Accuracy**: Also achieved a perfect **1.0**, performing on par with AdaBoost[span_98](end_span).

### Overall Analysis

* [span_99](start_span)All three top models—Random Forest, AdaBoost, and Gradient Boosting—are excellent candidates for this task[span_99](end_span).
* [span_100](start_span)For real-world applications, further tests are necessary to evaluate their robustness against noise or unseen data[span_100](end_span).

---

## Conclusion & Future Work 🏁

### Conclusion

[span_101](start_span)Our analysis demonstrates that models like **Random Forest**, **AdaBoost**, and **Gradient Boosting** are highly effective for classifying data in this project[span_101](end_span). [span_102](start_span)The high accuracy and strong performance metrics indicate their potential for effective prediction on the test data[span_102](end_span).

### Future Suggestions

To build upon this work, we recommend the following:
* **[span_103](start_span)Enhanced Generalizability Testing**: Evaluate the models on different datasets and use **k-fold cross-validation**[span_103](end_span).
* **[span_104](start_span)Advanced Optimization**: Employ techniques like **Bayesian Optimization** or **Random Search** for hyperparameter tuning and explore more powerful models like **XGBoost** or **LightGBM**[span_104](end_span).
* **[span_105](start_span)In-depth Feature Analysis**: Use tools like **SHAP** or **Permutation Importance** to investigate feature importance and simplify the model by removing less impactful features[span_105](end_span).
* **[span_106](start_span)Managing New and Noisy Data**: Simulate real-world scenarios with noisy data to check model performance and stability[span_106](end_span).
* **[span_107](start_span)Deployment**: Develop a **real-time prediction system** and implement the model in a REST service or an application for operational use[span_107](end_span).

---

## References 📚

* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. [span_108](start_span)Springer.[span_108](end_span)
* Breiman, L. (2001). Random Forests. *[span_109](start_span)Machine Learning*, 45(1), 5-32.[span_109](end_span)
* Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *[span_110](start_span)Machine Learning*, 20(3), 273-297.[span_110](end_span)
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. [span_111](start_span)MIT Press.[span_111](end_span)
* Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *[span_112](start_span)Annals of Statistics*, 29(5), 1189-1232.[span_112](end_span)

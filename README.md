# Data Analysis of the 2020 WUSTL EHMS Dataset for IoMT Cybersecurity

## Abstract 📄

<span id="span_1">This project focuses on the analysis of the **2020 WUSTL EHMS dataset**, a critical resource for research in the cybersecurity of the Internet of Medical Things (IoMT).</span> <span id="span_2">The data is sourced from an Enhanced Health Monitoring System (EHMS) that collects patient information in real-time.</span> <span id="span_3">Our primary objective is to identify the most effective methods for processing and classifying this data.</span>

**<span id="span_4">Keywords</span>**: 2020 WUSTL EHMS, IoMT Cybersecurity, Enhanced Health Monitoring System, Data Classification

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

<span id="span_5">Data analysis is a fundamental tool for extracting valuable insights from data.</span> <span id="span_6">This project delves into a dataset containing information about attacks and the medical status of patients.</span> <span id="span_7">The goal is to leverage machine learning models to predict and identify these attacks within healthcare data.</span> <span id="span_8"><span id="span_9">The timely detection of such threats is paramount in medical and hospital environments, as it can be a life-saving intervention.</span></span>

### Project Objectives

<span id="span_10">The core aim is to design and implement machine learning models capable of predicting attacks and medical emergencies from health data.</span> Key steps in this process include:

* **<span id="span_11">Data Preprocessing</span>**
* **<span id="span_12">Anomaly Detection</span>**
* **<span id="span_13">Model Evaluation</span>**

<span id="span_14">We will utilize tools like **LazyPredict** for an initial comparison and evaluate the final models based on metrics such as **Accuracy** and **F1-Score**.</span>

### Importance and Application

<span id="span_15">The ability to predict medical attacks is of immense importance, particularly in critical care and hospital settings.</span> <span id="span_16">This project aims to provide a framework that assists researchers and medical professionals in detecting threats more rapidly and accurately, thereby enabling prompt and effective treatment.</span>

---

## Comparison of Prediction Methods 📊

<span id="span_17">Predictive analytics plays a cornerstone role in modern healthcare, aiding in everything from disease prevention to resource management.</span> <span id="span_18">The accuracy of these predictions can significantly influence clinical decisions.</span> Here's a comparison of several common machine learning algorithms:

* <span id="span_19">Logistic Regression</span>
* <span id="span_20">Decision Tree</span>
* <span id="span_21">Random Forest</span>
* <span id="span_22">Support Vector Machines (SVM)</span>
* <span id="span_23">Artificial Neural Networks (ANN)</span>
* <span id="span_24">Gradient Boosting</span>
* <span id="span_25">K-Nearest Neighbors (KNN)</span>

| Criteria                            | Logistic Regression                         | Decision Tree                         | Random Forest                          | SVM                                    | ANN                                 | Gradient Boosting                      | KNN                                 |
| :---------------------------------- | :------------------------------------------ | :------------------------------------ | :------------------------------------- | :------------------------------------- | :---------------------------------- | :------------------------------------- | :---------------------------------- |
| **Implementation Simplicity**       | <span id="span_26">Relatively Simple</span> | <span id="span_27">Very Simple</span> | <span id="span_28">More Complex</span> | <span id="span_29">More Complex</span> | <span id="span_30">Complex</span>   | <span id="span_31">More Complex</span> | <span id="span_32">Simple</span>    |
| **Interpretability**                | <span id="span_33">High</span>              | <span id="span_34">High</span>        | <span id="span_35">Medium</span>       | <span id="span_36">Low</span>          | <span id="span_37">Low</span>       | <span id="span_38">Medium</span>       | <span id="span_39">Low</span>       |
| **Handling Large Data**             | <span id="span_40">Limited</span>           | <span id="span_41">Medium</span>      | <span id="span_42">Good</span>         | <span id="span_43">Good</span>         | <span id="span_44">Very Good</span> | <span id="span_45">Very Good</span>    | <span id="span_46">Limited</span>   |
| **Resistance to Overfitting**       | <span id="span_47">Low</span>               | <span id="span_48">Low</span>         | <span id="span_49">High</span>         | <span id="span_50">Medium</span>       | <span id="span_51">Low</span>       | <span id="span_52">Very High</span>    | <span id="span_53">Low</span>       |
| **Prediction Accuracy**             | <span id="span_54">Medium</span>            | <span id="span_55">Lower</span>       | <span id="span_56">High</span>         | <span id="span_57">High</span>         | <span id="span_58">Very High</span> | <span id="span_59">Very High</span>    | <span id="span_60">Medium</span>    |
| **Training Speed**                  | <span id="span_61">Very Fast</span>         | <span id="span_62">Fast</span>        | <span id="span_63">Faster</span>       | <span id="span_64">Slower</span>       | <span id="span_65">Slow</span>      | <span id="span_66">Slower</span>       | <span id="span_67">Very Fast</span> |
| **Need for Parameter Tuning**       | <span id="span_68">Low</span>               | <span id="span_69">Low</span>         | <span id="span_70">Medium</span>       | <span id="span_71">High</span>         | <span id="span_72">High</span>      | <span id="span_73">High</span>         | <span id="span_74">Low</span>       |
| **Suitability for Non-linear Data** | <span id="span_75">Weak</span>              | <span id="span_76">Medium</span>      | <span id="span_77">Good</span>         | <span id="span_78">Very Good</span>    | <span id="span_79">Very Good</span> | <span id="span_80">Very Good</span>    | <span id="span_81">Weak</span>      |

---

## Proposed Methodology 🛠️

### Data Preprocessing

This is a vital first step. The dataset contains missing values and anomalies that must be addressed:

* **<span id="span_82">Handling Missing Values</span>**: We use the **mean** for numerical columns and the **mode** for non-numerical ones.
* **<span id="span_83">Replacing Zeros</span>**: To avoid computational issues, zero values are replaced with the column's mean.
* **<span id="span_84">Data Normalization</span>**: We apply **MinMaxScaler** to normalize the feature set.

### Feature Selection

We carefully select features relevant to identifying attacks. **<span id="span_85">One-Hot Encoding</span>** is used for categorical variables to improve model performance, and features with low correlation to the target variable are removed.

### Modeling and Evaluation

We employ a range of machine learning models for prediction:

* **<span id="span_86">LazyPredict</span>**: For a quick, high-level comparison.
* **<span id="span_87">Random Forest</span>**
* **<span id="span_88">Logistic Regression</span>**
* **<span id="span_89">SVM</span>**

<span id="span_90">The performance of these models is rigorously evaluated using metrics like **Accuracy**, **F1-Score**, **Precision**, and **Recall**.</span>

### Anomaly Detection

<span id="span_91">To enhance model performance, we use **Isolation Forest** and **One-Class SVM** to identify and remove outliers from the training data.</span>

---

## Best Algorithm Analysis 🏆

Our analysis revealed that several models performed exceptionally well.

1. **Random Forest Classifier**:

   * **<span id="span_92">Accuracy</span>**: Achieved an impressive **0.9963**.
   * **<span id="span_93">Performance</span>**: Showed high precision, recall, and f1-scores for all classes. <span id="span_94">The confusion matrix confirmed a low error rate.</span>
   * **<span id="span_95">ROC Curve</span>**: The Area Under the Curve (AUC) was close to 1, demonstrating excellent class separation capability.

2. **AdaBoost Classifier**:

   * **<span id="span_96">Accuracy</span>**: Reached a perfect **1.0**.
   * **<span id="span_97">Performance</span>**: The confusion matrix and ROC curve reflected this perfect score.

3. **Gradient Boosting Classifier**:

   * **<span id="span_98">Accuracy</span>**: Also achieved a perfect **1.0**, performing on par with AdaBoost.

### Overall Analysis

* <span id="span_99">All three top models—Random Forest, AdaBoost, and Gradient Boosting—are excellent candidates for this task.</span>
* <span id="span_100">For real-world applications, further tests are necessary to evaluate their robustness against noise or unseen data.</span>

---

## Conclusion & Future Work 🏁

### Conclusion

<span id="span_101">Our analysis demonstrates that models like **Random Forest**, **AdaBoost**, and **Gradient Boosting** are highly effective for classifying data in this project.</span> <span id="span_102">The high accuracy and strong performance metrics indicate their potential for effective prediction on the test data.</span>

### Future Suggestions

To build upon this work, we recommend the following:

* **<span id="span_103">Enhanced Generalizability Testing</span>**: Evaluate the models on different datasets and use **k-fold cross-validation**.
* **<span id="span_104">Advanced Optimization</span>**: Employ techniques like **Bayesian Optimization** or **Random Search** for hyperparameter tuning and explore more powerful models like **XGBoost** or **LightGBM**.
* **<span id="span_105">In-depth Feature Analysis</span>**: Use tools like **SHAP** or **Permutation Importance** to investigate feature importance and simplify the model by removing less impactful features.
* **<span id="span_106">Managing New and Noisy Data</span>**: Simulate real-world scenarios with noisy data to check model performance and stability.
* **<span id="span_107">Deployment</span>**: Develop a **real-time prediction system** and implement the model in a REST service or an application for operational use.

---

## References 📚

* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. <span id="span_108">Springer.</span>
* Breiman, L. (2001). Random Forests. *<span id="span_109">Machine Learning</span>*, 45(1), 5-32.
* Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *<span id="span_110">Machine Learning</span>*, 20(3), 273-297.
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. <span id="span_111">MIT Press.</span>
* Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *<span id="span_112">Annals of Statistics</span>*, 29(5), 1189-1232.

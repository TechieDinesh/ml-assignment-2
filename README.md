📌 Problem Statement

The goal of this project is to build multiple machine learning classification models to predict whether a patient has heart disease based on medical attributes. The project also demonstrates deployment of these models using a Streamlit web application.

Dataset Description

The dataset contains medical attributes such as age, cholesterol level, resting blood pressure, heart rate, and other health indicators.
The target variable represents whether the patient has heart disease (1) or not (0).

The dataset contains:
1. 800 records
2. 13 input features
3. Binary classification target

Models Used & Evaluation Metrics :
<img width="335" height="106" alt="image" src="https://github.com/user-attachments/assets/ab148f67-6886-49e0-b917-17236766c8a9" />

Model	            Accuracy	AUC	    Precision	Recall	        F1	        MCC
Logistic Regression	0.95	0.967342342	0.960526316	0.986486486	0.973333333	0.587929801
Decision Tree	        1	      1	        1	        1	        1	        1
KNN	                0.94375	0.787725225	0.942675159 	1	    0.970491803	    0.4854573
Naive Bayes	        0.94375	0.951013514	0.948387097	0.993243243	0.97029703	    0.494374045
Random Forest	    0.95	0.995213964	0.948717949	     1	    0.973684211	    0.562351595
XGBoost	            0.99375	0.999436937	0.993288591	     1	    0.996632997	    0.954208856

Model Performance Observations

Observations: 

ML Model Name  

Observation about model performance 

Logistic Regression 

Achieved high accuracy (95%) with strong precision and recall, indicating that the dataset has meaningful linear relationships. However, its MCC score is moderate, suggesting some limitations in handling complex patterns. 

Decision Tree 

Obtained perfect accuracy and MCC on the test set, indicating that it learned the dataset very well. However, such perfect scores may indicate overfitting, meaning the model might not generalize well to unseen data 

KNN 

Showed good recall (100%) but comparatively lower MCC and AUC, suggesting that although it correctly identifies most positive cases, its overall predictive reliability is weaker. 

Naive Bayes 

Performed consistently with high recall and F1 score, showing it handles class separation well. However, its moderate MCC indicates that the independence assumption may reduce performance on correlated features. 

Random Forest 

Achieved strong accuracy and very high AUC, indicating good predictive power and stability. The ensemble approach reduces overfitting compared to a single decision tree. 

XGBoost 

Delivered the best overall performance with the highest accuracy, AUC, F1 score, and MCC. This shows that boosting effectively captured complex relationships and produced the most reliable model for this dataset. 


Streamlit App Features are below : 
1. Upload CSV dataset
2. Select ML model
3. Display confusion matrix
4. Display classification report


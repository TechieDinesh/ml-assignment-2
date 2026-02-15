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

Model	            Accuracy	AUC	    Precision	Recall	        F1	        MCC
Logistic Regression	0.95	0.967342342	0.960526316	0.986486486	0.973333333	0.587929801
Decision Tree	        1	      1	        1	        1	        1	        1
KNN	                0.94375	0.787725225	0.942675159 	1	    0.970491803	    0.4854573
Naive Bayes	        0.94375	0.951013514	0.948387097	0.993243243	0.97029703	    0.494374045
Random Forest	    0.95	0.995213964	0.948717949	     1	    0.973684211	    0.562351595
XGBoost	            0.99375	0.999436937	0.993288591	     1	    0.996632997	    0.954208856

Model Performance Observations

Logistic Regression:
Performs well on linear relationships but may miss complex patterns.

Decision Tree:
Captures nonlinear patterns but may overfit on training data.

KNN:
Works well for local patterns but sensitive to feature scaling and noise.

Naive Bayes:
Fast and efficient but assumes feature independence.

Random Forest:
Improves accuracy by combining multiple decision trees and reduces overfitting.

XGBoost:
Usually provides the best performance due to boosting and handling complex relationships.

Streamlit App Features are below : 
1. Upload CSV dataset
2. Select ML model
3. Display confusion matrix
4. Display classification report
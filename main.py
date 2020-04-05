import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib

from pprint import pprint

def preprocessing(data):
	X = data.drop('Class', axis=1)
	y = data['Class']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	robust_scaler = RobustScaler().fit(X_train)

	X_test = pd.DataFrame(robust_scaler.transform(X_test), columns=X.columns)

	return X_test, y_test

def load_and_predict(model_name, X_test, y_test):
	model = joblib.load(model_name)

	predictions = model.predict(X_test)

	print(classification_report(y_test, predictions, target_names=['Not Fraud', 'Fraud']))

def test_single_tuple(model_name, X_test, y_test):
	model = joblib.load(model_name)	
	
	input_tuple = X_test.sample(1)

	print("Selected tuple is:\n")
	pprint(input_tuple)

	print("\nPrediction made for given input was", 'Not Fraud' if model.predict(input_tuple)[0] == 0 else 'Fraud')

	print("True Result for given input was", 'Not Fraud' if y_test.iloc[input_tuple.index[0]] == 0 else 'Fraud')

if __name__ == '__main__':
	data = pd.read_csv('creditcard.csv')
	print("\n1. Dataset Read Complete")

	X_test, y_test = preprocessing(data)
	print("\n2. Dataset Preprocessing Complete\n")

	print("Classification Report for Logistic Regression:\n")
	load_and_predict('logistic_regression_model.pkl', X_test, y_test)

	print("Classification Report for Random Forest:\n")
	load_and_predict('random_forest_model.pkl', X_test, y_test)

	print("Classification Report for Naive Bayes:\n")
	load_and_predict('naive_bayes_model.pkl', X_test, y_test)

	print("\nSelecting a Random Tuple each time and Running the Models on it\n")

	print("\nFor Logistic Regression:")
	test_single_tuple('logistic_regression_model.pkl', X_test, y_test)

	print("\nFor Random Forest:")
	test_single_tuple('random_forest_model.pkl', X_test, y_test)

	print("\nFor Naive Bayes:")
	test_single_tuple('naive_bayes_model.pkl', X_test, y_test)

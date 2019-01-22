import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#set seaborn default style
sns.set()


def main():
	loans = pd.read_csv("./loan_data.csv")
	#histogram of two FICO distributions on top of each other, one for each credit.policy outcome
	plt.figure(figsize=(10,6))
	loans[loans['credit.policy'] == 1]['fico'].hist(bins=35,color='blue',alpha=0.6,label='credit policy=1')
	loans[loans['credit.policy'] == 0]['fico'].hist(bins=35,color='red',alpha=0.6,label='credit policy=0')
	plt.legend()
	plt.title('FICO distributions by credit.policy column')
	plt.xlabel('FICO score')
	plt.show()

	#histogram of two FICO distributions on top of each other, one for each not.fully.paid column
	plt.figure(figsize=(10,6))
	loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=35,color='blue',alpha=0.6,label='not fully paid=1')
	loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=35,color='red',alpha=0.6,label='not fully paid=0')
	plt.legend()
	plt.title('FICO distributions by not.fully.paid column')
	plt.xlabel('FICO score')
	plt.show()

	#countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
	plt.figure(figsize=(10,8))
	sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1',)
	plt.title('Loans count by purpose')
	plt.show()

	#create a fixed larger dataframe that has new feature columns with dummy variables for purpose category column
	cat_feats=['purpose']
	final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
	#split training data and test data
	X=final_data.drop('not.fully.paid',axis=1)
	y=final_data['not.fully.paid']
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
	#train decision tree classifier
	dtree = DecisionTreeClassifier()
	dtree.fit(X_train,y_train)
	#predict results
	predictions = dtree.predict(X_test)
	print("----- Evaluation report using Deicision Tree Classifier -----")
	print_evaluation_report(y_test,predictions)

	#train random forest classifier
	rfClass = RandomForestClassifier()
	rfClass.fit(X_train,y_train)
	pred_rfc = rfClass.predict(X_test)
	print("----- Evaluation report using Random Forest Classifier -----")
	print_evaluation_report(y_test,pred_rfc)


def print_evaluation_report(y_test,pred):
	print(classification_report(y_test,pred))
	print('\n')
	print(confusion_matrix(y_test,pred))


if __name__ == '__main__':
	main()
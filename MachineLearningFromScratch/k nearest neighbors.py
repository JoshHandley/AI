import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
#recognised as outlier
df.replace('?', -9999, inplace=True)
#remmove un related data
df.drop(['id'], 1, inplace=True)
#features
x= np.array(df.drop(['class'], 1))
#labels
y= np.array(df['class'])
#shuffle data and seperate into train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,4,1], [4,2,1,2,2,2,3,4,1]])
#predict on anynumber of list objexts
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
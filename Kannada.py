# Import
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Train and Test
x_train = np.load('Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_train.npz')['arr_0']
x_test = np.load('Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_test.npz')['arr_0']
y_train = np.load('Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_train.npz')['arr_0']
y_test = np.load('Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_test.npz')['arr_0']
x_train = x_train.reshape(x_train.shape[0], (28*28))
x_test = x_test.reshape(x_test.shape[0], (28*28))
pca_no = int(input("Enter the PCA n_components :"))
pca = PCA(n_components=pca_no)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
# Model Processing Funtion
def model_process(model,x_train, y_train, x_test, y_test):
  model.fit(x_train, y_train)
  print(model)
  expected_y  = y_test
  predicted_y = model.predict(x_test)
  print(metrics.classification_report(expected_y, predicted_y))
  print(metrics.confusion_matrix(expected_y, predicted_y))
  print("\n")
# Dicision Tree
model = tree.DecisionTreeClassifier()
model_process(model,x_train, y_train, x_test, y_test)
# Random Forest
model = ExtraTreesClassifier()
model_process(model,x_train, y_train, x_test, y_test)
# Naive Bayes Model
model = GaussianNB()
model_process(model,x_train, y_train, x_test, y_test)
#KNN Classifier
model = KNeighborsClassifier()
model_process(model,x_train,y_train,x_test,y_test)
#SVM
model = svm.SVC()
model_process(model,x_train, y_train, x_test, y_test)
#SVM OvO
model = svm.SVC(decision_function_shape='ovo')
model_process(model,x_train, y_train, x_test, y_test)
#Linear SVM
model = svm.LinearSVC(class_weight='balanced')
model_process(model,x_train, y_train, x_test, y_test)
#SVM RBF
model = svm.SVC(kernel='rbf')
model_process(model,x_train, y_train, x_test, y_test)
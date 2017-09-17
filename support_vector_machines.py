# -*- coding: utf-8 -*-
#SVM-> Support Vector Machines
# Finds a separeting line (hyperplane) between data of two classes
# Margin: Distance between hyperplane and the nearest point of either of the two classes.
# If we maximize distance, we maximaze robustness of the result.
#SVM puts emphaty in first make a correct classification, and then maximazing the margin.
#Some timed there are ouliers (parte aislada, un punt barrejat en la classificació que no li toca).
#If we consider z=x2+y2 (el·lipse, all the red crosses are close to the x line respect z, and all the blue circles are far, so there is a svm that can separete them.
#Kernel trick: using functions that transform low dimensional spaces to more dimensional spaces:
# x,y ---> x1,x2,x3,x4,x5 (not linear separable to separable) (kernel)
# Non linear separation <--- solution
from sklearn import svm
X = [[0,0],[1,1]]
Y =[0,1]
clf = svm.SVC(kernel="linear", gamma=1.0)  #SVC: support vector classification
clf.fit(X,Y)
print clf.predict([[2.,2.]])
#Parameters passed when creating the classifier: arguments before fitting the classifier
#	-> kernel (linear, rbf)
#	-> C: Controls the tradeoff between a smooth decision boundary and classifies all training point correctly
#	-> gamma: defines how far the influence of a single training example reaches. Low values: far, hight values: close reach
#Over fitting: thing to avoid in machine learning
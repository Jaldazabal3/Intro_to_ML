# -*- coding: utf-8 -*-
#Example of supervised prediction

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB 								#import external modules, in this case GaussianNB
clf = GaussianNB()														#create a classifier
clf.fit(X,Y)															#fit (or train) classifier with our data, we are giving our training data X->features and Y->labels to learn the patterns
#Ask the classifier that we have trained for two predictions
#print(clf.predict([[-0.8,-1]]))


# Dado un estimador para una variable estadistica discreta se definen 2 valores asociados:
# Sensibilidad: capacidad de estimador para dar como casos positivos los casos realmente enfermos. Capacidad de prueba para detectar la enfermedad en sujetos enfermos
# Especificidad: capacidad de estimado para dar con casos negativos los casos realmente sanos.
#Bayes Rule-> incorporates somo evidence from the test into your prior probability to arrive at posterior priority,
#prior: probabilidad antes de ejecutar el test * test evidence = posterior probability
# P(h|d)=(P(d|h)·P(h))/P(d)
# prior: P(C) =0.01     P(Pos | C) = 0.9    P(Neg | !C) =0.9
#joint (first posterior)   P(C | Pos ) = P(C)·Pos(Pos|C)=0.009         P(!C|Pos)=P(!C)·P(Pos|!C)=0.099

#Normalizer: probability of a positive test result (sum of other two) : 0.108
#posterior: divide the joint/normalizer = 0.08333 and 0.91666   -> el resultat dona 1


#prior: P(C)
#sensitivity: P(Pos | C)
#specitivity: P(Neg | !C)

#Naive bayes (example Chris Sarah):
#P(Chris) = P(Sarah)=0.5   Joint-->  P(Chris | "Life Deal") = 0.04 P(Sarah | "Life Deal")=0.03  Normalizer = 0.07
#Posterior --> Chris: 0.5714 Sarah: 0.4285

#Chris joint: 0.1*0.8*0.5
chris = 0.1*0.8*0.5
sarah= 0.5*0.2*0.5
total=sarah+chris
print "Chris = "+str(chris/total)
print "Sarah ="+str(sarah/total)
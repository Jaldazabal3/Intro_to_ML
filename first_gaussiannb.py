#Example of supervised prediction

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB 								#import external modules, in this case GaussianNB
clf = GaussianNB()														#create a classifier
clf.fit(X,Y)															#fit (or train) classifier with our data, we are giving our training data X->features and Y->labels to learn the patterns
#Ask the classifier that we have trained for two predictions
print(clf.predict([[-0.8,-1]]))


# Dado un estimador para una variable estadistica discreta se definen 2 valores asociados:
# Sensibilidad: capacidad de estimador para dar como casos positivos los casos realmente enfermos. Capacidad de prueba para detectar la enfermedad en sujetos enfermos
# Especificidad: capacidad de estimado para dar con casos negativos los casos realmente sanos.
#Bayes Rule-> incorporates somo evidence from the test into your prior probability to arrive at posterior priority,
# P(h|d)=(P(d|h)·P(h))/P(d)
# prior: P(C) =0.01     P(Pos | C) = 0.9    P(Neg | !C) =0.9
#posterior   P(C | Pos ) = P(C)·Pos(Pos|C)=0.009         P(!C|Pos)=P(!C)·P(Pos|!C)=0.099

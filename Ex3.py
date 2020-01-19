import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#definition of the class starts here
class indexWithScore:
    indexOne = 0
    indexTwo = 0
    score = 0

    #defining constructor
    def __init__(self, indexOne, indexTwo , score):
        self.indexOne = indexOne
        self.indexTwo = indexTwo
        self.score = score

read = pd.read_excel('BreastTissue.xlsx', sheet_name='Data')
read = shuffle(read)

# Exclude X matrix
X = read.iloc[:, 2:].values
y = read.iloc[:, 1].values
y = pd.factorize(y)[0]

norm_x = X.T
for i in range(0, np.size(norm_x, 0)):
    norm_x[i] = (np.subtract(norm_x[i], norm_x[i].mean()) / norm_x[i].std())
norm_x = norm_x.T

"""
normalize data then Split the data.
training set while holding out 30% of the data for testing (evaluating) our classifier
"""
X_train, X_test, y_train, y_test = train_test_split(norm_x,y, test_size=0.3, random_state=0)


"""
To find max score
"""
def hamdan(X_train, X_test, y_train, y_test):
    firstScores = []
    for i in range(X_train.shape[1]):
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train[:,[i]], y_train)
        firstScores.append(logisticRegr.score(X_test[:,[i]], y_test))
    firstX = firstScores.index(max(firstScores))
    secondScores = []
    for i in range(X_train.shape[1] - 1):
        if i != firstX:
            temp_X_train = np.append(X_train[:, [firstX]], X_train[:, [i]], axis=1)
            temp_X_test = np.append(X_test[:, [firstX]], X_test[:, [i]], axis=1)
            logisticRegr = LogisticRegression()
            logisticRegr.fit(temp_X_train, y_train)
            secondScores.append(logisticRegr.score(temp_X_test, y_test))
    secondX = secondScores.index(max(secondScores))
    return firstX , secondX , firstScores , secondScores

def two(X_train, X_test, y_train, y_test):
    scores = []
    tempScores = []
    for i in range(X_train.shape[1]):
        for j in range(i+1,X_train.shape[1]):
            temp_X_train = np.append(X_train[:, [i]], X_train[:, [j]], axis=1)
            temp_X_test = np.append(X_test[:, [i]], X_test[:, [j]], axis=1)
            logisticRegr = LogisticRegression()
            logisticRegr.fit(temp_X_train, y_train)
            scores.append(indexWithScore(i,j,logisticRegr.score(temp_X_test, y_test)))
    for i in range(len(scores)):
        tempScores.append(scores[i].score)
    index = tempScores.index(max(tempScores))
    return scores[index].indexOne , scores[index].indexTwo , tempScores

firstX , secondX , firstScores , secondScores = hamdan(X_train, X_test, y_train, y_test)
print("Hamdan : firstX is",firstX,"secondX is" , secondX)
one , two , tempScores = two(X_train, X_test, y_train, y_test)
print("Algorithm Two : firstX is",one,"secondX is" , two)
plt.suptitle('Hamdan')
for i in range(len(firstScores)):
    plt.plot(firstScores)
for i in range(len(secondScores)):
    plt.plot(secondScores)
plt.xlabel('Attribute')
plt.ylabel('Scores')
plt.show()
plt.suptitle('Algorithm Two')
for i in range(len(firstScores)):
    plt.plot(tempScores)
plt.xlabel('Attribute')
plt.ylabel('Scores')
plt.show()
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import sklearn as sk



#definition of the class starts here
class IndexWithProbability:
    indexClass = 0
    probability = 0

    #defining constructor
    def __init__(self, indexClass, probability):
        self.indexClass = indexClass
        self.probability = probability


epsilon = 0.005
delta = 0.001

################################################################################
read = pd.read_excel('BreastTissue.xlsx', sheet_name='Data')
"""
0- car
1-fad
2-mas
3-gla
4-con
5-adi
"""
# Exclude X matrix
x = read.iloc[:, 2:].values
y = read.iloc[:, 1].values
y = pd.factorize(y)[0].tolist()

###############################################################################
"""
vector for normalize
Normalize target data
-----------------------
Normalize x matrix of Features
------------------------------
"""
norm_x = x.T
for i in range(0,np.size(norm_x,0)):
    norm_x[i] = (np.subtract(norm_x[i], norm_x[i].mean()) / norm_x[i].std())
norm_x = norm_x.T
"""
Create table start with a column of one's rows and add normalize the table
"""
# addOnes = np.ones((len(y),1))
# norm_xWithOnes = np.append(addOnes, norm_x, axis=1)

numOfClass = np.unique(y).shape[0]

# for help
cols = np.size(norm_x, 1)
rows = np.size(norm_x, 0)

# Construct matrix for K binary training sets
classesMatrix = np.zeros((rows, numOfClass))

for i in range(rows):
    classesMatrix[i][y[i]] = 1

dataWithClasses = np.append(norm_x , classesMatrix , axis=1)
np.random.shuffle(dataWithClasses)
rowTrain_data = int(dataWithClasses.shape[0]*0.7)
rowTest_data = dataWithClasses.shape[0] - rowTrain_data
train_data_with_classes = dataWithClasses[:rowTrain_data , :]
test_data_with_classes = dataWithClasses[rowTrain_data:,:]

def onVsAll(train_data_with_classes,test_data_with_classes):

    LG = []
    for JClass in range(numOfClass):
        diabetesCheck = LogisticRegression()
        LG.append(diabetesCheck.fit(train_data_with_classes[:,:-numOfClass], train_data_with_classes[:,-(numOfClass - JClass)]))

    sortBestProbability = []
    indexWithMaxProbability = []

    for row in range(test_data_with_classes.shape[0]):
        for JClass in range(numOfClass):
            sortBestProbability.append(LG[JClass].predict_proba([test_data_with_classes[row,:-numOfClass]])[0][1])
        temp = IndexWithProbability(sortBestProbability.index(max(sortBestProbability)) ,max(sortBestProbability))
        indexWithMaxProbability.append(temp)
        sortBestProbability = []

    counterForClass = np.zeros(numOfClass)

    for run in range(test_data_with_classes.shape[0]):
        numC = indexWithMaxProbability[run].indexClass
        if(test_data_with_classes[run , -(numOfClass - numC)] == 1):
            counterForClass[numC] = counterForClass[numC] + 1

    correctAnswer = counterForClass.sum()

    return indexWithMaxProbability , correctAnswer


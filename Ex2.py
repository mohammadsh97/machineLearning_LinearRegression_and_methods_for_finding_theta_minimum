import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



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
y = pd.factorize(y)[0]

###############################################################################
"""
vector for normalize
Normalize target data
-----------------------
Normalize x matrix of Features
------------------------------
"""
norm_x = x.T
for i in range(0, np.size(norm_x, 0)):
    norm_x[i] = (np.subtract(norm_x[i], norm_x[i].mean()) / norm_x[i].std())
norm_x = norm_x.T

# for help
norm_xAfterThatY = np.append(norm_x, y.reshape(y.shape[0],1), axis=1)
numOfClass = np.unique(y).shape[0]
cols = np.size(norm_x, 1)
rows = np.size(norm_x, 0)

# Construct matrix for K binary training sets
classesMatrix = np.zeros((rows, numOfClass))

for i in range(rows):
    classesMatrix[i][y[i]] = 1

dataWithClasses = np.append(norm_x, classesMatrix, axis=1)
np.random.shuffle(dataWithClasses)
rowTrain_data = int(dataWithClasses.shape[0] * 0.7)
rowTest_data = dataWithClasses.shape[0] - rowTrain_data
train_data_with_classes = dataWithClasses[:rowTrain_data, :]
test_data_with_classes = dataWithClasses[rowTrain_data:, :]


def oneVsAll(train_data_with_classes,test_data_with_classes,numOfClass):

    LG = []#build algorthem from training
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


def oneVsOne(dataWithClasses,norm_xAfterThatY, numOfClass):
    arrForAllTableWithOutY = []
    counter = 0
    startIndex = 0

    # to separate classes
    ##########################################################################################################################################
    for i in range(norm_xAfterThatY.shape[0]):
        if(norm_xAfterThatY[i, norm_xAfterThatY.shape[1]-1] != counter or (counter == numOfClass - 1 and i == norm_xAfterThatY.shape[0] - 1)):
            if(counter == numOfClass - 1 and i == norm_xAfterThatY.shape[0] - 1):
                #array for all table with out col y "arrForAllTable"
                arrForAllTableWithOutY.append(norm_xAfterThatY[startIndex:i+1, :-1])
                startIndex = i
                counter += 1
            else:
                arrForAllTableWithOutY.append(norm_xAfterThatY[startIndex:i ,:-1])
                startIndex = i
                counter += 1
    ##########################################################################################################################################


    seventyPercentFromAllTable_train_data = []
    thirtyPercentFromAllTable_test_data = []

    #to take 70% from each class and make 30% for test data
    for run in range(numOfClass):
        end = int(arrForAllTableWithOutY[run].shape[0] * 0.7)
        #data train
        seventyPercentFromAllTable_train_data.append(arrForAllTableWithOutY[run][ :end, :])
        #data test
        thirtyPercentFromAllTable_test_data.append(arrForAllTableWithOutY[run][end:, :])

    temp_data_train = []
    temp_data_test = []

    #to collect one Vs one class
    for i in range(numOfClass):
        for j in range(i, numOfClass):
            temp_data_train = np.append(
                #add col ones for first class
                #############################################################################
                np.append(
                seventyPercentFromAllTable_train_data[i] ,
                np.ones((seventyPercentFromAllTable_train_data[i].shape[0], 1)), axis = 1)
                #############################################################################
                ,
                # add col zeros for secand class
                #############################################################################
                np.append(
                seventyPercentFromAllTable_train_data[j] ,
                np.zeros((seventyPercentFromAllTable_train_data[j].shape[0], 1)), axis = 1)
                #############################################################################
                ,
                axis=0)
            temp_data_test = np.append()

    oneVsOneClass = []
    LG = [] #build algorthem from training
    #shuffle row for all class and build algorthem
    for i in range(15):
        np.random.shuffle(oneVsOneClass[i])
        np.random.shuffle(thirtyPercentFromAllTable_test_data[i])
        diabetesCheck = LogisticRegression()
        #diabetesCheck.fit( all of data X , data Y)
        LG.append(diabetesCheck.fit(temp_data_train[i][:, :-1],temp_data_train[i][:,-1]))


    # prediction = diabetesLoadedModel.predict(sampleDataFeatures)



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



    return arrForAllTableWithOutY


# arrForAllTable = oneVsOne(train_data_with_classes,test_data_with_classes,norm_xAfterThatY,numOfClass)
#
# arrForAllTable = np.array(arrForAllTable)
# print(arrForAllTable[0][ :10, :])

print(np.ones((10,1)).shape)
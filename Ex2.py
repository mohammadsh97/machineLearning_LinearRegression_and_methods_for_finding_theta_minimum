import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sns as sns
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


def NumberOfTable(numOfClass):
    temp = 0
    for run in range(0,numOfClass):
        temp += run
    return temp


def oneVsAll(train_data_with_classes,test_data_with_classes,numOfClass):

    LG = []#build algorthem from training
    for JClass in range(numOfClass):
        diabetesCheck = LogisticRegression()
        LG.append(diabetesCheck.fit(train_data_with_classes[:,:-numOfClass], train_data_with_classes[:,-(numOfClass - JClass)]))

    sortBestProbability = []
    indexWithMaxProbabilityForOne = []
    tempProbability = []
    probability = []
    for row in range(test_data_with_classes.shape[0]):
        for JClass in range(numOfClass):
            tempProbability.append(LG[JClass].predict_proba([test_data_with_classes[row,:-numOfClass]])[0])
            sortBestProbability.append(tempProbability[JClass][1])
        temp = IndexWithProbability(sortBestProbability.index(max(sortBestProbability)) ,max(sortBestProbability))
        indexWithMaxProbabilityForOne.append(temp)
        probability.append(tempProbability)
        sortBestProbability = []
        tempProbability = []
    # print(probability[0])
    # print(indexWithMaxProbabilityForOne[0].probability , "->" ,indexWithMaxProbabilityForOne[0].indexClass)
    # print(probability[1])
    # print(indexWithMaxProbabilityForOne[1].probability, "->" ,indexWithMaxProbabilityForOne[1].indexClass)
    # print(probability[2])
    # print(indexWithMaxProbabilityForOne[2].probability, "->" ,indexWithMaxProbabilityForOne[1].indexClass)
    counterForClass = np.zeros(numOfClass)

    for run in range(test_data_with_classes.shape[0]):
        numC = indexWithMaxProbabilityForOne[run].indexClass
        if(test_data_with_classes[run , -(numOfClass - numC)] == 1):
            counterForClass[numC] = counterForClass[numC] + 1

    correctAnswer = counterForClass.sum()
    indexWithMaxProbabilityForOne = np.array(indexWithMaxProbabilityForOne)
    probability = np.array(probability)

    return probability , indexWithMaxProbabilityForOne , correctAnswer



def oneVsOne(norm_xAfterThatY, numOfClass):
    numberOfTable = NumberOfTable(numOfClass)
    arrForAllTableWithY = []
    counter = 0
    startIndex = 0

    # to separate classes
    ##########################################################################################################################################
    for i in range(norm_xAfterThatY.shape[0]):
        if(norm_xAfterThatY[i, norm_xAfterThatY.shape[1]-1] != counter or (counter == numOfClass - 1 and i == norm_xAfterThatY.shape[0] - 1)):
            if(counter == numOfClass - 1 and i == norm_xAfterThatY.shape[0] - 1):
                #array for all table with col y "arrForAllTable"
                arrForAllTableWithY.append(norm_xAfterThatY[startIndex:i+1, :])
                startIndex = i
                counter += 1
            else:
                arrForAllTableWithY.append(norm_xAfterThatY[startIndex:i ,:])
                startIndex = i
                counter += 1
    ##########################################################################################################################################
    arrForAllTableWithY = np.array(arrForAllTableWithY)
    seventyPercentFromAllTable_train_data = []
    thirtyPercentFromAllTable_test_data = []

    #to take 70% for data train and 30% for test data from each class
    for run in range(numOfClass):
        end = int(arrForAllTableWithY[run].shape[0] * 0.7)
        #data train with out Y
        seventyPercentFromAllTable_train_data.append(arrForAllTableWithY[run][ :end, :-1])
        #data test with Y 0-car 1-fad 2-mas 3-gla 4-con 5-adi
        thirtyPercentFromAllTable_test_data.append(arrForAllTableWithY[run][end:, :])
    # For help
    ##########################################
    temp_data_train = []

    LG = []  # build algorthem from training
    counter = 0

    ##########################################
    #to collect one Vs one class
    for i in range(numOfClass - 1):
        for j in range(i+1, numOfClass):
            temp_data_train.append(np.append(
                # add col ones for first class
                #############################################################################
                np.append(np.array(seventyPercentFromAllTable_train_data[i]),
                          np.ones((seventyPercentFromAllTable_train_data[i].shape[0], 1)), axis=1)
                #############################################################################
                ,
                # add col zeros for secand class
                #############################################################################
                np.append(np.array(seventyPercentFromAllTable_train_data[j]),
                          np.zeros((seventyPercentFromAllTable_train_data[j].shape[0], 1)), axis=1)
                #############################################################################
                ,
                axis=0))
            if counter < numberOfTable:  # range (numberOfTable -> 16)
                # the counter start from point 0 -> end to point 15
                # the lable data training =  temp_data_train[counter][:, -1]
                # the data training = temp_data_train[counter][:, :-1]

                # shuffle row for all class and build algorthem
                np.random.shuffle(temp_data_train[counter])
                np.random.shuffle(thirtyPercentFromAllTable_test_data[counter])
                #######################################################################################################
                # LogisticRegression()
                # for training
                #######################################################################################################
                diabetesCheck = LogisticRegression()
                # diabetesCheck.fit( all of data X , data Y)
                LG.append(diabetesCheck.fit(temp_data_train[counter][:, :-1], temp_data_train[counter][:, -1]))
                #######################################################################################################
            #######################################################################################################
        counter += 1
    thirtyPercentFromAllTable_test_data = np.array(thirtyPercentFromAllTable_test_data)
    LG = np.array(LG)
    Probability = []
    arrCounterToHelp = [0]*numOfClass
    counter = 0
    rowNumY = []
    tableRunY = []
    for tableRun in range (numOfClass):
        for rowRun in range(thirtyPercentFromAllTable_test_data[tableRun].shape[0]):
            counter = 0
            for i in range(numOfClass - 1):
                for j in range(i + 1, numOfClass):
                    # for testing
                    #######################################################################################################
                    if counter < numberOfTable :
                        Probability.append(LG[counter].predict_proba([thirtyPercentFromAllTable_test_data[tableRun][rowRun, :-1]])[0])
                        prediction = (LG[counter].predict([thirtyPercentFromAllTable_test_data[tableRun][rowRun, :-1]]))
                        if (prediction == 1):
                            arrCounterToHelp[i] += 1
                        else:
                            arrCounterToHelp[j] += 1
                        counter += 1
            index = arrCounterToHelp.index(max(arrCounterToHelp))
            rowNumY.append(index)
            arrCounterToHelp = [0] * numOfClass
        tableRunY.append(rowNumY)
        rowNumY = []
    tableRunY = np.array(tableRunY)
    counter = 0
    correctAnswer = 0
    allOfRow = 0
    for i in range(tableRunY.shape[0]):
        allOfRow += len(tableRunY[i])
        for j in range(len(tableRunY[i])):
            if(tableRunY[i][j] == counter):
                correctAnswer += 1
        counter +=1
    return arrForAllTableWithY , tableRunY , correctAnswer , allOfRow ,Probability

arrForAllTableWithY , tableRunY , correctAnswer2 , allOfRow ,Probability = oneVsOne(norm_xAfterThatY, numOfClass)
probability , indexWithMaxProbabilityForOne , correctAnswer1 = oneVsAll(train_data_with_classes,test_data_with_classes,numOfClass)


for i in range(probability.shape[0]):
    for j in range(len(probability[i])):
        plt.plot(probability[i][j])
plt.suptitle('probability for one vs all')
print("one vs all: The correct Answer is: " ,correctAnswer1 , " Form :",test_data_with_classes.shape[0])
plt.show()
for i in range(tableRunY.shape[0]):
    plt.plot(tableRunY[i])
plt.suptitle('prediction for one vs one \n X: 0->car 1->fad 2->mas 3->gla 4->con 5->adi')
print("one vs one: The correct Answer is: " , correctAnswer2 , " Form :",allOfRow)
plt.show()
plt.plot(Probability)
plt.suptitle('Probability for one vs one')
plt.xlabel('30% * 15 class = 525')
plt.ylabel('probability to give one or zero')
plt.show()
plt.plot(arrForAllTableWithY[0], marker="+")
plt.suptitle('The class is: car')
plt.xlabel('row')
plt.ylabel('Value')
plt.show()
plt.plot(arrForAllTableWithY[1], marker="x")
plt.suptitle('The class is: fad')
plt.xlabel('row')
plt.ylabel('Value')
plt.show()
plt.plot(arrForAllTableWithY[2], marker="+")
plt.suptitle('The class is: mas')
plt.xlabel('row')
plt.ylabel('Value')
plt.show()
plt.plot(arrForAllTableWithY[3], marker="+")
plt.suptitle('The class is: gla')
plt.xlabel('row')
plt.ylabel('Value')
plt.show()
plt.plot(arrForAllTableWithY[4], marker="+")
plt.suptitle('The class is: con')
plt.xlabel('row')
plt.ylabel('Value')
plt.show()
# plt.plot(arrForAllTableWithY[5], marker="+")
# plt.subplot('The class is: adi')
# plt.show()


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# -> θj     : Weights of the hypothesis.
# -> hθ(xi) : predicted y value for ith input.
# -> j     : Feature index number (can be 0, 1, 2, ......, n).
# -> α     : Learning Rate of Gradient Descent.

################################################################################
read = pd.read_csv('./cancer_data.csv')
x = read.iloc[:,:-1].values
y = read.iloc[:,-1].values
###############################################################################
epsilon = 0.005
delta = 0.001

"""
vector for normalize
"""
def vec_normal(vec):
    return math.sqrt(sum([v ** 2 for v in vec]))
"""
Q1)A)
------
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
Normalize y matrix of Features
------------------------------
"""
norm_y = np.subtract(y, y.mean()) / y.std()
"""
Create table start with a column of one's rows and add normalize the table
"""
addOnes = np.ones((len(y),1))
norm_xWithOnes = np.append(addOnes, norm_x, axis=1)
"""
The end of normalize and add column of one's
"""
################################################################################
#Q1)B)
#------
def regressionLineaire (x, theta):
    res = 0
    for i in range(len(theta)):
        res += theta[i] * x[i]
    return res
################################################################################
#Q1)C)
def jOfTheta(x ,y,theta):
    m = len(y)
    j = 0
    for i in range(m):
        j += (regressionLineaire(x[i],theta) - y[i]) ** 2
    return j / (2 * m)
################################################################################
#Q1)D)
def gradentJOfTheta(startIndex ,x ,y , theta):
    m = len(y)
    partOfSolve = np.zeros((len(theta),))
    for j in range(len(theta)):
        for run in range(startIndex ,startIndex + m):
            i = run % m
            partOfSolve[j] += ((regressionLineaire(x[i], theta) - y[i]) * x[i][j])/m
    return partOfSolve
################################################################################
#Q1)E)
def gradientDescent(x ,y ,learning_rate ,max_itreations):
    theta = np.random.randn(10)
    resultsOfJ = []
    for k in range(max_itreations):
        newtheta = theta - (learning_rate * gradentJOfTheta(0,x, y, theta))
        resultsOfJ.append(jOfTheta(x, y, theta))

        if vec_normal(newtheta - theta) < epsilon or abs(resultsOfJ[-1] - resultsOfJ[-2]) < delta if len(resultsOfJ) > 1 else False:
            return newtheta, resultsOfJ
        else:
            theta = newtheta
    return theta, resultsOfJ

################################################################################
#Q1)F)
def miniBatch(number ,x ,y ,learning_rate ,max_itreations):
    theta = [0] * (len(x[0]))
    resultsOfJ = []
    for k in range(max_itreations):
        newtheta = theta - learning_rate * gradentJOfTheta(number ,x , y, theta)
        resultsOfJ.append(jOfTheta(x, y, theta))
        if vec_normal(newtheta - theta) < epsilon or abs(resultsOfJ[-1] - resultsOfJ[-2]) < delta if len(resultsOfJ) > 1 else False:
            return newtheta, resultsOfJ
        else:
            theta = newtheta
    return theta, resultsOfJ
#Q1)J)
def momentum(x ,y,learning_rate,max_itreations):
    v, theta, resultsOfJ = np.zeros((len(x[0],))), np.zeros((len(x[0],))), []
    for i in range(max_itreations):
        new_v = ((1 - learning_rate) * v) + (learning_rate * (gradentJOfTheta(0,x, y, theta)))
        new_theta = theta - new_v
        resultsOfJ.append(jOfTheta(x,y,new_theta))
        if vec_normal(new_theta - theta) < epsilon or abs(resultsOfJ[-1] - resultsOfJ[-2]) < delta if len(resultsOfJ) > 1 else False:
            return new_theta, resultsOfJ
        else:
            v = new_v
            theta = new_theta
    return theta, resultsOfJ
"""
my tester
"""
temp0, gradientDescentAlphaZeroDotOne = gradientDescent(norm_xWithOnes ,norm_y ,0.1 ,100)
temp1, gradientDescentAlphaZeroDotZeroOne = gradientDescent(norm_xWithOnes ,norm_y ,0.01 ,100)
temp2, gradientDescentAlphaZeroDotZeroZeroOne = gradientDescent(norm_xWithOnes ,norm_y ,0.001 ,100)
temp3, momentumAlphaZeroDotOne = momentum(norm_xWithOnes ,norm_y,0.1,100)
temp4, miniBatch = miniBatch(20 ,norm_xWithOnes ,norm_y ,0.1 ,100)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(gradientDescentAlphaZeroDotOne, 'r--')
axes[0].plot(gradientDescentAlphaZeroDotZeroOne, 'g--')
axes[0].plot(gradientDescentAlphaZeroDotZeroZeroOne, 'b*-')
axes[0].set_ylim([0, 20])
axes[0].set_xlim([0, 130])
axes[0].axis('tight')
axes[0].set_title("gradientDescent: alpha = 0.1, 0.01, 0.001")
axes[1].plot(miniBatch , 'b*--')
axes[1].set_ylim([0, 60])
axes[1].set_xlim([0, 60])
axes[1].axis('tight')
axes[1].set_title("MiniBatch")
axes[2].plot(momentumAlphaZeroDotOne, 'b*--')
axes[2].set_ylim([0, 3])
axes[2].set_xlim([0, 15])
axes[2].axis('tight')
axes[2].set_title("Momentum")
plt.show()
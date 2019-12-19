import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

dfOriginal = pd.read_excel('BreastTissue.xlsx', sheet_name = 'Data')

df = dfOriginal.drop(['Case #'], axis=1)

#shuffle rows
df1 = df.sample(frac = 1).reset_index(drop=True)

df = df1
df_train = df[:80]
df_test = df[80:]
#df[df == 0].count(axis=0) #check for missing elements

x_train = np.asarray(df_train.drop('Class',1))
y_train = np.asarray(df_train['Class'])

x_test = np.asarray(df_test.drop('Class',1))
y_test = np.asarray(df_test['Class'])

# normalize x_train and x_test matrix
means = np.mean(x_train, axis=0)
stds = np.std(x_train, axis=0)
x_train = (x_train - means) / stds
x_test = (x_test - means) / stds

# the class names list
classNames = list(set(np.asarray(df['Class'])))


#####################################################################################################################

#add h vector to a given model
def calcH(model, xTest):
    theta_dot_x = [np.dot(model['model'].coef_[0], row) for row in xTest]
    model['h'] = [1 / (1 + np.math.e ** (-1 * tx)) for tx in theta_dot_x]


###########################################################One vs All################################################

def Obj(className, training_classes_results, xTrain):
    yData = [1 if name == className else 0 for name in training_classes_results]
    model = LogisticRegression().fit(xTrain, yData)
    return {"className": className,
            "model": model
            }


def oneVsAll(oneVsAllModel, x_test):
    predictions = []
    if not ('h' in oneVsAllModel[0].keys()):##checks where are the models 'h' and calculates each model using the calc function
        for model in oneVsAllModel:
            calcH(model, x_test)
    for i in range(len(oneVsAllModel[0]['h'])):
        max = 0
        maxClass = oneVsAllModel[0]['className']
        for model in oneVsAllModel:
            if model['h'][i] > max:
                max = model['h'][i]
                maxClass = model['className']
        predictions.append(maxClass)
    return predictions


oneVsAllModel = [Obj(className, df_train['Class'], x_train) for className in classNames]

oneVsAllPredict = oneVsAll(oneVsAllModel, x_test)
cm_one_vs_all_test = confusion_matrix(df_test['Class'], oneVsAllPredict, labels = classNames)


f1_one_vs_all = f1_score(df_test['Class'], oneVsAllPredict, average='micro')

plt.figure(figsize=(6, 6))
sns.heatmap(cm_one_vs_all_test,xticklabels=classNames, yticklabels=classNames, annot=True, fmt="d", linewidths=1, square=True, cmap='hot')

plt.ylabel('real value ')
plt.xlabel('predicted value ')
plt.title("One vs All \nf1_score : {}".format(f1_one_vs_all))
plt.show()

########################################################### End of One vs All################################################

########################################################### Start of One vs One##############################################

# returns an object containing a pair and its model, training only the rows of the pair
def createPairsObj(name1, name2, training_classes_results, x_train):
    x_train = [(xRow, 1 if class_res == name1 else 0) for xRow, class_res in zip(x_train, training_classes_results) if
               class_res == name1 or class_res == name2]
    yData = [class_res for xRow, class_res in x_train]
    x_train = [xRow for xRow, class_res in x_train]
    model = LogisticRegression().fit(x_train, yData)
    return {"pair":(name1, name2),
            "model": model
           }


def getPredictions_oneVsOne(oneVsOnePairsModels, x_test):
    oneVsOneH = [] # a list of dictionaries with 'className' and 'h' as keys
    for model in oneVsOnePairsModels:
        calcH(model, x_test)
    for className in classNames:
        h = [0] * len(oneVsOnePairsModels[0]['h'])
        for i in range(len(oneVsOnePairsModels[0]['h'])):
            for model_pair in oneVsOnePairsModels:
                name1, name2 = model_pair["pair"]
                if name1 == className:
                    h[i] += model_pair['h'][i]
                if name2 == className:
                    h[i] += 1 - model_pair['h'][i]
        oneVsOneH.append({"className": className, "h": h})

    predictions = [] # list of max h for each x_test
    for i in range(len(oneVsOneH[0]['h'])):
        max = 0
        maxClass = ''
        for x in oneVsOneH:
            if x['h'][i] > max:
                max = x['h'][i]
                maxClass = x['className']
        predictions.append(maxClass)
    return predictions



# pairs combination
all_pairs = [(name1, name2) for name1 in classNames for name2 in classNames if name1 < name2]
# list containing objects of pairs and their model
oneVsOnePairsModels = [createPairsObj(name1, name2, df_train['Class'], x_train) for (name1, name2) in all_pairs]


oneVsOnePredict = getPredictions_oneVsOne(oneVsOnePairsModels, x_test)

cm_one_vs_one_test = confusion_matrix(df_test['Class'], oneVsOnePredict, labels = classNames)
f1_one_vs_one = f1_score(df_test['Class'], oneVsOnePredict, average='micro')

plt.figure(figsize=(6, 6))
sns.heatmap(cm_one_vs_one_test,xticklabels=classNames, yticklabels=classNames , annot=True, fmt="d", linewidths=.5, square = True, cmap='Blues_r')
plt.ylabel('real value ')
plt.xlabel('predicted value ')
plt.title('One vs One Confusion matrix \nf1_score : {}'.format(f1_one_vs_one))

plt.show()

########################################################### End of One vs One################################################


'''

#tester

print("one vs all:")
print(list(y_test))
print(list(LogisticRegression().fit(x_train, y_train).predict(x_test)))
print(oneVsAllPredict)

print("----------------------------------------------------------------------------------------------")

print("one vs one:")
print(list(y_test))
print(list(LogisticRegression(multi_class='multinomial').fit(x_train, y_train).predict(x_test)))
print(oneVsOnePredict)

'''


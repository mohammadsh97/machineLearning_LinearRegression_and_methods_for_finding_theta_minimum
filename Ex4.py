import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
tempX = read.iloc[:, 2:].values
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

# ================================================================================================
# ==== Helper functions ==========================================================================
# ================================================================================================
def init_centroids(k, data):
    '''
        This function will be used to initialize the centroids once in the beginning.
        Centroids will be randomly chosen as points(features) from the dataset, as this provides faster
        convergence. Another implementation
        can be assigning completely random numbers as centroids, but this is dangerous.
        :param k: (int) number of centroids
        :param data: (np-array) containing the features of the dataset
        :return: (list) 'k' number of randomly selected centroids from the dataset
        '''
    c = []
    # Generate a list of k random numbers between 0 and the # of features of dataset
    s = np.random.randint(low=1, high=len(data), size=k)
    # Check if all members of 's' are unique, if not, generate again
    while (len(s) != len(set(s))):
        s = np.random.randint(low=1, high=len(data), size=k)
    # For every i in s, get ith feature from dataset, and add it to list
    for i in s:
        c.append(data[i])
    return c

def euc_dist(a, b):
    '''
    This function calculates and returns the euclidean distance between two input vectors.
    This is a helper function for cal_dist() to calculate distance of any given point in
    data w.r.t. the centroids.
    :param a: (list) vector a
    :param b: (list) vector b
    :return: (float) euclidean distance between two input vectors
    '''
    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return np.math.sqrt(sum)

def cal_dist(centroids, data):
    '''
    This function will be used to generate the distance table.
    For each point in the given dataset, this function will calculate its euclidean distance
    with respect to each centroid, and a distance table will be generated, which will then be
    used to update the centroid positions.
    :param centroids: (list) containing position of centroids
    :param data: (np-array) containing the features of the dataset
    :return: (list) containing the distances of each point w.r.t. each centroid
    '''
    c_dist = []
    # For each centroid c, iterate through all points in data to calculate its distance from c
    for i in centroids:
        temp = []
        for j in data:
            temp.append(euc_dist(i, j))
        c_dist.append(temp)
    return c_dist

def perf_clustering(k, dist_table):
    '''
    This function will perform clustering on the basis of distance table w.r.t. to all centroids.
    With reference to the distance table, for each point in the table, this function will compare
    its distance from all the centroids, and then cluster the point with the nearest centroid.
    :param k: (int) number of centroids
    :param dist_table: (list) containing the distances of each point w.r.t. each centroid
    :return: (list) containing clusters and indexes of respective members
    '''
    # create empty cluster list of size k
    clusters = []
    for i in range(k):
        clusters.append([])
    # start clustering data points, such that each point is clustered to nearest centroid
    for i in range(len(dist_table[0])):
        d = []
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)
    return clusters

def update_centroids(centroids, cluster_table, data):
    '''
    This function will update the centroids locations after each iteration
    After performing the clustering, some elements might have migrated from one cluster
    to the another cluster, so w.r.t. the new cluster table, this function will calculate and
    update the new centroid locations
    :param centroids: (list) containing position of centroids
    :param cluster_table: (list) containing clusters and indexes of respective members
    :param data: (np-array) containing the features of the dataset
    :return: (list) containing updated position of centroids
    '''
    for i in range(len(centroids)):
        # Update the centroid if there are some flowers within this centroid
        if (len(cluster_table[i]) > 0):
            temp = []
            # Copy features of cluster members to temp list
            for j in cluster_table[i]:
                temp.append(list(data[j]))
            # Take mean of features of all members of cluster to get new centroid location
            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]
            centroids[i] = [p / len(temp) for p in sum]

    return centroids


def check_n_stop(dist_mem, cluster_mem):
    '''
    This function will check if the stopping criteria has been met
    This function will help to check if there is a convergence to a solution, with the help of
    specified stopping criterias:
    In case of Iris-Dataset, we can use the following stopping criterias:
    --[1] The euclidean distances/distance table stays unchanged for 2 or more iterations.
    --[2] The cluster table stays unchanged for two or more iterations, i.e. no flower migrations
    :param dist_table: (list) containing the distances of each point w.r.t. each centroid
    :param cluster_table: (list) containing clusters and indexes of respective members
    :return: True, if stopping criteria is met, else False.
    '''

    # Check if distance table has not changed over past iterations
    c1 = all(x == dist_mem[0] for x in dist_mem)
    # Check if cluster table has not changed over past iterations
    c2 = all(y == cluster_mem[0] for y in cluster_mem)

    if c1:
        print("Stopping... Distance table has not changed from few iterations")
    elif c2:
        print("Stopping... Cluster table has not changed from few iterations")
    return c1 or c2

# ================================================================================================
# ==== K-Means ===================================================================================
# ================================================================================================

def kMeans(k, data, max_iterations):
    '''
    Simple implementation of K-Means algorithm
    This function is a very naive implementation of the K-Means algorithm.
    Steps:
    --[1] Initialize random centroids
    --[2] Calculate distances for each point w.r.t. centroid, and store them in distance table.
    --[3] Perform clustering with the available distance table
    --[4] Update the centroid positions
    --[5] Stop if stopping criteria is met, else repeat again from --[2]
    :param k: (int) number of centroids
    :param data: (np-array) containing the features of the dataset
    :param max_iterations: (int) number of maximum iterations allowed.
    :return: none
    '''

    # These lists will maintain memory to check if stopping criteria is met
    dist_mem = []
    cluster_mem = []

    # Initialize centroids
    centroids = init_centroids(k, data)
    # Calculate distance table
    distance_table = cal_dist(centroids, data)
    # Perform clustering based on above generated distance table
    cluster_table = perf_clustering(k, distance_table)
    # Update centroid location based on above generated cluster table
    newCentroids = update_centroids(centroids, cluster_table, data)

    # Add distance and cluster table to memory list
    dist_mem.append(distance_table)
    cluster_mem.append(cluster_table)

    # Repeat from step 2 till stopping criteria is met
    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)

        # Check for stopping criteria
        # Maintain memory for past distance table and cluster table
        dist_mem.append(distance_table)
        cluster_mem.append(cluster_table)
        # If distance/cluster has not changed over last 10 iterations, stop, else continue
        if len(dist_mem) > 10:
            dist_mem.pop(0)
            cluster_mem.pop(0)
            if check_n_stop(dist_mem, cluster_mem):
                print("Stopped at iteration #", i)
                break

    # Display the final results
    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        plt.scatter(original_data[cluster_table[i]][:,1], original_data[cluster_table[i]][:,2])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
            print(original_data[cluster_table[i][j]])
    plt.show()

    # ================================================================================================
    # ==== Main K-Means  =============================================================================
    # ================================================================================================

one , two , tempScores = two(X_train, X_test, y_train, y_test)
f1 = tempX[:, [one]]
f2 = tempX[:, [two]]
data = np.array(list(zip(f1, f2)))
mergeData = []
mergeData = np.append( f1, f2,axis=1)
original_data = np.array(list(zip(y ,f1, f2)))
# Run the K-Means algorithm on the Iris-Dataset with k = 3, and max-iterations limited to 100
kMeans(6, data, 100)

# ================================================================================================
# ==== mixture of gaussians ======================================================================
# ================================================================================================
col = [read.columns[2+one] , read.columns[2+two]]
gmm = GaussianMixture(n_components=6)
gmm.fit(mergeData)
labels = gmm.predict(mergeData)
for k in range(0,6):
    newData = mergeData[labels==k]
    plt.scatter(newData[: , 0],newData[: , 1])
plt.show()
# ================================================================================================
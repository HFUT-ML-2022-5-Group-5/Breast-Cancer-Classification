import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#任务三:kmeans_pca
plt.rcParams['font.family'] = 'Microsoft YaHei'

from sklearn.datasets import load_breast_cancer
from numpy.linalg import svd

# import data
data = load_breast_cancer()
df = pd.DataFrame(data = data.data, columns=data.feature_names)
df.head()

print('Size of the matrix: ({0}, {1})'.format(len(df.index), len(df.columns)))

from sklearn.preprocessing import StandardScaler
def PCA(input_data,num_comp):
  # standarize
  data_stand = StandardScaler().fit_transform(input_data.data)
  # calculate eigenvectors
  u,s,vh = np.linalg.svd(data_stand.T)
  return u.T[0:num_comp], data_stand
eigenVectors, data_std = PCA(data,2)
# multiply the original dataset with eigenvector
product = np.dot(data_std, eigenVectors.T)

labels = set(data['target'])
label_dict={
    0:'malignant',
    1:'benign'
}

plt.figure(figsize=(10,6))
for i in labels:
  plt.plot(product[data.target==i,0], product[data.target==i,1], 'o', label=label_dict[i])

plt.legend(['恶性', '良性'])
plt.show()

# This function returns the distance between two points
def distance(point1, point2):
  point1 = np.array(point1)
  point2 = np.array(point2)
  dist = np.sqrt(np.sum(np.power(point1-point2, 2)))
  return dist

def k_means(input_data, k, max_iter):
  # generate k centroid as starting points
  centroid = []
  for i in range(k):
    # centroid.append(np.random.uniform(np.min(input_data),np.max(input_data),len(input_data[0])))
    centroid.append(input_data[i])
  for _ in range(max_iter):  # iteration
    sum_dist = 0
    # calculate distance of each point to each centroid and assign class
    point_class = np.zeros(len(input_data),dtype=int)
    for i in range(len(input_data)):
      min_dist = np.Inf
      for j in range(k):
        point_dist = distance(centroid[j],input_data[i])
        if point_dist < min_dist:
          min_dist = point_dist
          point_min = j
      sum_dist = sum_dist + min_dist**2  # calculate distortion
      point_class[i] = point_min
    # update centroid positions
    for i in range(len(centroid)):
      centroid[i] = np.mean(input_data[point_class==i], axis=0)
  print('-----finished ', k)
  print(sum_dist)

  return centroid, point_class, sum_dist

data_stand = StandardScaler().fit_transform(data.data)
a,b,c = k_means(data_stand,2,1000)
print('Final Centroid:', a)

# calculate distortion
dist = []
for i in range(2,8):
  a,b,c = k_means(data_stand,i,1000)
  dist.append(c)

# plot distortion
plt.figure(figsize=(10,6))
plt.plot(np.arange(2,8),dist)
plt.xlabel('簇的数量————2022217511卢冠辰')
plt.ylabel('失真')
plt.show()

from sklearn.preprocessing import StandardScaler
# standardize
data_stand = StandardScaler().fit_transform(data.data)
a,b,c = k_means(data_stand,3,1000)
print('Final Centroid:', a)
def PCA(input_data,num_comp):
  # standarize
  data_stand = StandardScaler().fit_transform(input_data)
  # calculate eigenvectors
  u,s,vh = np.linalg.svd(data_stand.T)
  return u.T[0:num_comp], data_stand
eigenVectors_cl, data_std_cl = PCA(data.data,2)
product_cl = np.dot(data_std_cl, eigenVectors.T)
eigenVectors_cen, data_std_cen = PCA(a,2)
product_cen = np.dot(data_std_cen, eigenVectors.T)

labels = set(b)
label_dict = {
    0: 'cluster 0',
    1: 'cluster 1',
    2: 'cluster 2'
}

plt.figure(figsize=(15, 9))
for i in labels:
    plt.title('簇的数量 k=3  2022217511卢冠辰')
    fig1 = plt.plot(product_cl[b == i, 0], product_cl[b == i, 1], 'o', label=label_dict[i])
    fig2 = plt.plot(product_cen[:, 0], product_cen[:, 1], '^', markersize=15, markerfacecolor='red')

plt.show()

from sklearn.model_selection import train_test_split
# use standardized input
X, X_test, Y, Y_test = train_test_split(StandardScaler().fit_transform(data.data), data.target, test_size=0.3, random_state=1)

w = np.dot(np.linalg.inv((np.dot(np.transpose(X),X))),np.dot(np.transpose(X),Y))
w


def predict_lin(X,Y):
  # predict
  g = np.dot(X,w)
  # sigmoid
  g_sig = 1/(1 + np.exp(-g))
  # condition check
  result = []
  for i in range(len(g_sig)):
    if g_sig[i] >= 0.5:
      result.append(1)
    else:
      result.append(0)
  # report accuracy
  correct = 0
  for i in range(len(Y)):
    if Y[i] == result[i]:
      correct = correct + 1
  acc = correct/len(Y)
  return acc,result

print('Train Set Accuracy:', predict_lin(X,Y)[0])
print('Test Set Accuracy:', predict_lin(X_test,Y_test)[0])

# make prediction on the entire dataset
data_stand = StandardScaler().fit_transform(data.data)
pred = predict_lin(data_stand,data.target)[1]

# plot
labels = set(data['target'])
label_dict={
    0:'malignant',
    1:'benign'
}

plt.figure(figsize=(10,6))
for i in labels:
  plt.plot(product[pred==i,0], product[pred==i,1], 'o', label=label_dict[i])

plt.legend(['恶性', '良性'])
plt.show()

# the following function returns the bias. True label Y and predicted label 'prediction' are passed in to the function.
def get_bias(Y,prediction):
  bias = abs(np.mean(prediction) - np.mean(Y))
  return bias

# the following function returns the variance. The predicted label 'prediction' is passed in to the function
def get_var(prediction):
  var = np.mean((prediction - np.mean(prediction))**2)
  return var

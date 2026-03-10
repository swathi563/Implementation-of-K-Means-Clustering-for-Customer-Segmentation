<img width="837" height="542" alt="560756806-655e8f69-bcf6-4aa2-8c60-eb06b448679b" src="https://github.com/user-attachments/assets/3ada0042-1982-4bbf-b4b4-d9c21c0c9db3" /># Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the customer dataset and select the relevant features such as Annual Income and Spending Score.

2.Choose the number of clusters K and initialize K centroids randomly.

3.Assign each data point to the nearest centroid using Euclidean distance and update the centroids by calculating the mean of each cluster.

4.Repeat Step 3 until the centroids no longer change and display the final clusters for customer segmentation.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SWATHI S
RegisterNumber: 212225040449 
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
​
data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())
​
X = data.iloc[:, [3, 4]].values
​
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
​
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
​
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
​
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=300, c='yellow', label='Centroids')
​
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```
## Output:

<img width="721" height="146" alt="560756727-d3696d44-fa50-46bc-a4c3-283f680d50e6" src="https://github.com/user-attachments/assets/0dad40e7-1870-4222-a69e-dd9dee7188dc" />

<img width="837" height="542" alt="560756806-655e8f69-bcf6-4aa2-8c60-eb06b448679b" src="https://github.com/user-attachments/assets/28cd1b69-efea-4b21-abdc-36590cad1e99" />

<img width="847" height="628" alt="560756865-70e5b4a2-3da8-421e-80fb-697e33e5d7a4" src="https://github.com/user-attachments/assets/5eb437ab-3980-415e-858d-76481f57ea49" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

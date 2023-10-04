#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# In[53]:


#This will fetch the olivetti_faces dataset
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, Y = olivetti_faces.data, olivetti_faces.target


# In[54]:


# Split the data into training and temporary sets (80% training, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    olivetti_faces.data, olivetti_faces.target, test_size=0.2, random_state=42, stratify=olivetti_faces.target
)

# Split the temporary set into validation and test sets (50% validation, 50% test)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# In[55]:


# This will define the support vector machine classifier
classifier = SVC(kernel='linear', C=1)

# The below will perform the k-fold cross-validation on the training set
k_fold = 5  
cv_scores = cross_val_score(classifier, X_train, y_train, cv=k_fold)

classifier.fit(X_train, y_train)

# The below will predict the classifier on the validation set
y_pred = classifier.predict(X_valid)

# Calculate accuracy on the validation set
validation_accuracy = accuracy_score(y_valid, y_pred)

# Print the cross-validation scores and validation accuracy
print(f"Cross-validation scores: {cv_scores}")
print(f"Validation accuracy: {validation_accuracy}")


# In[56]:


#The below will define the range for number of cluster
k_values = range(2, 15) 

best_k = None
best_silhouette_score = -1

# Perform K-Means clustering for different values of K
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)
    
    # Calculate silhouette score for this K
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    
     #This will find the number of clusters and silhouette score
    print(f'Number of Clusters: {k}, Silhouette Score: {silhouette_avg:.2f}')
    
    # Update the best K and silhouette score if necessary
    if silhouette_avg > best_silhouette_score:
        best_k = k
        best_silhouette_score = silhouette_avg

# Fit K-Means with the chosen K
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = best_kmeans.fit_predict(X_train)

print("Best Number of clusters:", best_k)
print("Best Silhouette Score:", best_silhouette_score)

# Use the cluster labels as reduced features
reduced_features = np.column_stack((X_train, cluster_labels))

# Transform the data using the best K-Means model to reduce dimensionality
X_train_reduced = best_kmeans.transform(X_train)
X_valid_reduced = best_kmeans.transform(X_valid)


# In[57]:


#The below will define the support vector machine classifier
classifier = SVC(kernel='linear', C=1)

# Train the classifier on the training set
classifier.fit(X_train_reduced, y_train)

# Predict on the validation set
y_pred = classifier.predict(X_valid_reduced)

# Calculate accuracy on the validation set
validation_accuracy = accuracy_score(y_valid, y_pred)

# Print the validation accuracy
print(f"Validation accuracy: {validation_accuracy}")


# In[58]:


# Reshape the images into feature vectors
X = olivetti_faces.images.reshape((len(olivetti_faces.images), -1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# The below will apply the PCA for reduction
n_components = 100 
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# The below will compute the pairwise cosine distances
cosine_distances = pairwise_distances(X_pca, metric='cosine')

# Apply DBSCAN clustering
eps = 0.3  
min_samples = 5  
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
labels = dbscan.fit_predict(cosine_distances)

# The below will find the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Number of Clusters: {n_clusters}')


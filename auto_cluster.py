import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

class Cluster():
    def __init__(self, min_cluster:int = 2, max_cluster:int = 10, random_state:int = 42):
        self.scaler = None
        self.k = None
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.kmeans_model = None
        self.random_state = random_state
        
    def __fit_scaler(self, X):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        
    def __find_best_k(self, X):
        self.__fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        silhouette_scores = []
        for k in range(self.min_cluster, self.max_cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X_scaled)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X=X_scaled, labels=labels, random_state=self.random_state))
        self.k = self.min_cluster + np.argmax(silhouette_scores)
        
    def fit(self, X):
        self.__find_best_k(X)
        self.kmeans_model = KMeans(n_clusters=self.k, random_state=self.random_state)
        self.kmeans_model.fit(X)
        
    def predict(self, X):
        prediction = self.kmeans_model.predict(X)
        return prediction
    
    def fit_predict(self, X):
        self.fit(X)
        self.predict(X)

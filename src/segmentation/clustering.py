from __future__ import annotations
from typing import Dict,List,Tuple
import pandas as pd

import numpy as np
from sklearn.cluster import DBSCAN,KMeans
from sklearn.metrics import devices_bouldin_score,silhouette_score
RANDOM_SEED = 42
def  find_optimal_k(
        X:np.ndarray,k_range:range = range(2,11)
)->Tuple[int,pd.DataFrame]:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters = k,random_state = 42,n_init = 10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X,labels) if len(set(labels))>1 else float("nan")
        rows.append({"k":k,"inertia":km.inertia_,"silhouette_score":sil})
    diagnostics = pd.DataFrame(rows)
    best_k =int(diagnostics.loc[diagnostics["silhouette_score"].idxmax(),"k"])
    return best_k,diagnostics
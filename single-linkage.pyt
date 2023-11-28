import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

import pandas as pd

df = pd.read_csv('dataFix.csv')
df = df[['Age(Years)','Study time (Hours)','Time spent on social media (Hours)']]
df = df.dropna(axis=0)

clusters = hierarchy.linkage(df ,method="single")

plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(150, color='red', linestyle='--');
plt.axhline(100, color='crimson');
plt.show()

clustering_model = AgglomerativeClustering(n_clusters=3, linkage="single")
clustering_model.fit(df)
labels = clustering_model.labels_
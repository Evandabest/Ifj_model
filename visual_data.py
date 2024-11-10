import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

vectors = [
    ['1', 7, 10, 1],
    ['2', 4, 7, 8],
    ['3', 3, 1, 7],
    ['4', 9, 1, 4],
    ['5', 3, 2, 7],
    ['6', 4, 4, 5],
    ['7', 4, 3, 7],
    ['8', 3, 10, 8],
    ['9', 9, 3, 8],
    ['10', 9, 7, 4],
    ['11', 5, 3, 9],
    ['12', 3, 2, 9],
    ['13', 5, 4, 5],
    ['14', 2, 5, 9],
    ['15', 6, 8, 10],
    ['16', 6, 4, 9],
    ['17', 7, 8, 5],
    ['18', 10, 5, 8],
    ['19', 7, 9, 6],
    ['20', 6, 10, 8],
    ['21', 7, 7, 8],
    ['22', 1, 10, 10],
    ['23', 8, 10, 1],
    ['24', 6, 8, 5],
    ['25', 3, 4, 8],
    ['26', 4, 6, 7],
    ['27', 1, 4, 5],
    ['28', 2, 6, 5],
    ['29', 1, 7, 7],
    ['30', 1, 10, 8],
    ['31', 6, 7, 1],
    ['32', 10, 3, 10],
    ['33', 4, 5, 5],
    ['34', 4, 7, 7],
    ['35', 8, 1, 6],
    ['36', 3, 7, 4],
    ['37', 1, 3, 8],
    ['38', 2, 2, 3],
    ['39', 4, 4, 3],
    ['40', 3, 5, 10],
    ['41', 1, 9, 6],
    ['42', 6, 9, 6],
    ['43', 9, 1, 10],
    ['44', 8, 5, 2],
    ['45', 8, 4, 2],
    ['46', 7, 6, 7],
    ['47', 5, 3, 10],
    ['48', 6, 10, 5],
    ['49', 6, 5, 1],
    ['50', 2, 1, 4],
    ['51', 6, 6, 2],
    ['52', 7, 9, 6],
    ['53', 7, 5, 6],
    ['54', 10, 10, 4],
    ['55', 6, 5, 1],
    ['56', 7, 1, 6],
    ['57', 7, 3, 10],
    ['58', 9, 10, 5],
    ['59', 3, 7, 2],
    ['60', 7, 1, 2],
    ['61', 10, 8, 6],
    ['62', 6, 5, 1],
    ['63', 10, 3, 10],
    ['64', 8, 2, 1],
    ['65', 4, 4, 9],
    ['66', 3, 3, 4],
    ['67', 7, 6, 2],
    ['68', 5, 8, 3],
    ['69', 8, 9, 9],
    ['70', 6, 5, 2],
    ['71', 9, 7, 2],
    ['72', 5, 10, 6],
    ['73', 10, 5, 3],
    ['74', 2, 9, 5],
    ['75', 2, 7, 1],
    ['76', 1, 3, 10],
    ['77', 5, 1, 3],
    ['78', 7, 10, 7],
    ['79', 1, 8, 7],
    ['80', 10, 5, 3],
    ['81', 3, 8, 3],
    ['82', 5, 3, 10],
    ['83', 2, 6, 3],
    ['84', 6, 5, 3],
    ['85', 7, 5, 6],
    ['86', 10, 10, 7],
    ['87', 7, 1, 1],
    ['88', 10, 3, 3],
    ['89', 7, 5, 5],
    ['90', 10, 9, 7],
    ['91', 8, 1, 3],
    ['92', 9, 7, 5],
    ['93', 9, 6, 9],
    ['94', 6, 6, 9],
    ['95', 8, 1, 1],
    ['96', 10, 8, 9],
    ['97', 2, 5, 9],
    ['98', 1, 5, 9],
    ['99', 8, 6, 7],
    ['100', 10, 3, 7],
    ['101', 3, 7, 3],
    ['102', 6, 2, 2],
    ['103', 6, 1, 6],
    ['104', 8, 1, 6],
    ['105', 6, 7, 6],
    ['106', 2, 7, 3],
    ['107', 9, 6, 4],
    ['108', 4, 6, 4],
    ['109', 2, 10, 3],
    ['110', 6, 8, 4],
    ['111', 8, 1, 2],
    ['112', 3, 3, 9],
    ['113', 2, 7, 4],
    ['114', 3, 2, 7],
    ['115', 9, 10, 10],
    ['116', 1, 8, 9],
    ['117', 8, 7, 8],
    ['118', 4, 5, 7],
    ['119', 8, 2, 4],
    ['120', 7, 1, 9]
]

# Example: 100 candidates with their evaluation vectors (candidate_id, technical_skill, experience, education)
candidates_with_id = vectors

# Step 1: Create a list of vectors without the candidate ID for clustering
candidates = np.array([vector[1:] for vector in candidates_with_id])

candidates_scaled = candidates

# Step 3: Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(candidates_scaled)

# Step 4: Get the cluster labels and centroids
group_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 5: Plotting the clusters in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define colors for each cluster
colors = ['r', 'g', 'b']

# Plot each candidate in the scatter plot with cluster-based coloring
for i, candidate in enumerate(candidates_scaled):
    ax.scatter(candidate[0], candidate[1], candidate[2], color=colors[group_labels[i]], s=50)

# Plot centroids for each cluster with a new label
#ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='yellow', label='Cluster Centers', marker='X')

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           s=100,  # Size of the marker (you can adjust this value)
           c='orange', 
           label='Cluster Centers', 
           marker='o')  # Circle marker


# Labeling
ax.set_title("K-Means Clustering of Candidates")
ax.set_xlabel("Technical Skill")
ax.set_ylabel("Experience")
ax.set_zlabel("Education")
ax.legend(loc="upper right")

# Display the plot
plt.show()

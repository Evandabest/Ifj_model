import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

vectors = [
        ['1', 5, 1, 3],
    ['2', 1, 2, 2],
    ['3', 4, 6, 4],
    ['4', 1, 2, 4],
    ['5', 9, 4, 6],
    ['6', 2, 4, 4],
    ['7', 6, 2, 6],
    ['8', 8, 6, 2],
    ['9', 10, 6, 2],
    ['10', 10, 5, 8],
    ['11', 9, 7, 5],
    ['12', 1, 4, 8],
    ['13', 4, 6, 9],
    ['14', 3, 4, 3],
    ['15', 6, 5, 9],
    ['16', 7, 6, 3],
    ['17', 7, 9, 7],
    ['18', 8, 5, 9],
    ['19', 7, 9, 8],
    ['20', 2, 4, 1],
    ['21', 9, 6, 6],
    ['22', 8, 9, 6],
    ['23', 2, 4, 9],
    ['24', 9, 6, 4],
    ['25', 3, 1, 1],
    ['26', 4, 1, 7],
    ['27', 2, 10, 7],
    ['28', 10, 1, 1],
    ['29', 3, 4, 1],
    ['30', 8, 6, 1],
    ['31', 1, 10, 5],
    ['32', 5, 1, 10],
    ['33', 1, 8, 9],
    ['34', 1, 7, 3],
    ['35', 3, 7, 5],
    ['36', 9, 9, 2],
    ['37', 7, 2, 7],
    ['38', 6, 1, 2],
    ['39', 6, 9, 6],
    ['40', 9, 4, 10],
    ['41', 3, 6, 1],
    ['42', 9, 9, 8],
    ['43', 10, 3, 1],
    ['44', 10, 6, 9],
    ['45', 4, 2, 3],
    ['46', 8, 5, 5],
    ['47', 5, 5, 8],
    ['48', 8, 8, 4],
    ['49', 6, 1, 8],
    ['50', 5, 1, 4],
    ['51', 2, 10, 2],
    ['52', 7, 9, 9],
    ['53', 3, 9, 1],
    ['54', 5, 7, 3],
    ['55', 6, 7, 9],
    ['56', 4, 8, 7],
    ['57', 7, 2, 4],
    ['58', 6, 5, 6],
    ['59', 5, 2, 3],
    ['60', 9, 8, 10],
    ['61', 7, 9, 2],
    ['62', 1, 8, 4],
    ['63', 9, 8, 4],
    ['64', 10, 9, 9],
    ['65', 5, 7, 3],
    ['66', 10, 1, 6],
    ['67', 4, 2, 1],
    ['68', 6, 4, 7],
    ['69', 10, 5, 6],
    ['70', 3, 2, 2],
    ['71', 1, 7, 8],
    ['72', 3, 8, 1],
    ['73', 2, 4, 10],
    ['74', 2, 6, 4],
    ['75', 1, 8, 7],
    ['76', 7, 3, 9],
    ['77', 2, 5, 6],
    ['78', 3, 8, 1],
    ['79', 1, 2, 7],
    ['80', 10, 6, 6],
    ['81', 7, 1, 8],
    ['82', 1, 2, 8],
    ['83', 10, 9, 6],
    ['84', 10, 4, 6],
    ['85', 4, 3, 1],
    ['86', 4, 7, 4],
    ['87', 5, 8, 2],
    ['88', 3, 1, 3],
    ['89', 8, 10, 5],
    ['90', 8, 7, 10],
    ['91', 9, 2, 1],
    ['92', 4, 2, 6],
    ['93', 1, 10, 6],
    ['94', 10, 3, 2],
    ['95', 4, 5, 6],
    ['96', 10, 9, 5],
    ['97', 5, 7, 1],
    ['98', 10, 6, 2],
    ['99', 5, 5, 2],
    ['100', 7, 1, 2],
    ['101', 1, 1, 2],
    ['102', 1, 7, 5],
    ['103', 5, 3, 4],
    ['104', 9, 10, 7],
    ['105', 3, 8, 2],
    ['106', 2, 6, 9],
    ['107', 10, 8, 3],
    ['108', 1, 1, 10],
    ['109', 4, 7, 4],
    ['110', 6, 10, 2],
    ['111', 8, 1, 2],
    ['112', 3, 10, 7],
    ['113', 2, 6, 9],
    ['114', 4, 9, 5],
    ['115', 4, 3, 7],
    ['116', 10, 9, 9],
    ['117', 4, 7, 9],
    ['118', 7, 2, 10],
    ['119', 10, 9, 10],
    ['120', 10, 6, 10]
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

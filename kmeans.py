import random
import numpy as np
from sklearn.cluster import KMeans

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
    ['95', 8, 2, 9],
    ['96', 10, 8, 9],
    ['97', 2, 5, 9],
    ['98', 1, 5, 9],
    ['99', 8, 6, 7],
    ['100', 10, 3, 7],
    ['101', 3, 7, 3],
    ['102', 6, 6, 6],
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

# Step 2: Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 groups
kmeans.fit(candidates)

# Step 3: Get the group labels for each candidate
group_labels = kmeans.labels_

# Step 4: Combine the original candidate data with the assigned group label
clustered_candidates = [
    (candidates_with_id[i], group_labels[i]) for i in range(len(candidates_with_id))
]

# Step 5: Assign candidates to groups
group1, group2, group3 = [], [], []

for i in range(len(candidates_with_id)):
    if group_labels[i] == 0:
        group1.append(candidates_with_id[i])
    elif group_labels[i] == 1:
        group2.append(candidates_with_id[i])
    elif group_labels[i] == 2:
        group3.append(candidates_with_id[i])

# Step 6: Determine the minimum group size
min_size = min(len(group1), len(group2), len(group3))

# Step 7: Balance groups by randomly selecting `min_size` candidates for each group
group1_balanced = random.sample(group1, min_size)
group2_balanced = random.sample(group2, min_size)
group3_balanced = random.sample(group3, min_size)

# Remaining candidates after balancing
remaining_group1 = [candidate for candidate in group1 if candidate not in group1_balanced]
remaining_group2 = [candidate for candidate in group2 if candidate not in group2_balanced]
remaining_group3 = [candidate for candidate in group3 if candidate not in group3_balanced]

# Step 8: Generate balanced teams
teams = []

while len(group1_balanced) >= 2 and len(group2_balanced) >= 2 and len(group3_balanced) >= 2:
    team = []
    # Select 2 candidates from each balanced group for the team
    team.extend(random.sample(group1_balanced, 2))
    team.extend(random.sample(group2_balanced, 2))
    team.extend(random.sample(group3_balanced, 2))

    # Remove selected candidates from the balanced groups
    group1_balanced = [candidate for candidate in group1_balanced if candidate not in team]
    group2_balanced = [candidate for candidate in group2_balanced if candidate not in team]
    group3_balanced = [candidate for candidate in group3_balanced if candidate not in team]

    # Add the team to the list of teams
    teams.append(team)

remaining_candidates = remaining_group1 + remaining_group2 + remaining_group3
while remaining_candidates:
    team = remaining_candidates[:6]  # Take up to 6 candidates
    remaining_candidates = remaining_candidates[6:]  # Update remaining candidates
    teams.append(team)


# Step 10: Display the teams
for idx, team in enumerate(teams):
    print(f"Team {idx + 1}:")
    for candidate in team:
        print(f"Candidate ID: {candidate[0]}")
    print("-" * 30)
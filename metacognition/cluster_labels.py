import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

input_path = "metacognition/outputs/bug_labels.csv"
output_path = "metacognition/outputs/clusters.json"

df = pd.read_csv(input_path)

print(df.head())
# print(len(df["label"].unique())) # 4329 unique, some duplicates

df['label'] = df['label'].str.replace('_', ' ')

# we can potentially try using OpenAI embeddings to cluster as well, could be better
# it does look pretty good right now though with Tfidf

# we can also have a scheme where the input function source code gets embedded and then
# we choose the appropriate bugs to insert (for example if there is no list we can't
# ask it to do list-based bugs)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['label'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# print(tfidf_df)

seed = 42
k_values = range(16, 513, 16)

wcss = []

for k in tqdm(k_values):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    df['cluster'] = kmeans.fit_predict(tfidf_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show() # around 160 bug categories

print("How many clusters do you want? ", end="")
N = int(input())

kmeans = KMeans(n_clusters=N, random_state=seed, n_init='auto')
df['cluster'] = kmeans.fit_predict(tfidf_df)

df['label'] = df['label'].str.replace(' ', '_')

print(df.head())

cluster_dict = df.groupby('cluster')['label'].apply(list).to_dict()
with open(output_path, "w") as fp:
    json.dump(cluster_dict, fp)

# kmeans = KMeans(n_clusters=3, random_state=42)  # Number of clusters set to 3
# kmeans.fit(tfidf_df)

# # Add the cluster labels to the original DataFrame
# df['cluster'] = kmeans.labels_

# from openai_utils import system_prompt, user_prompt, openai_json_response

# @system_prompt
# def cluster_labels():
#     return f'''Your job is to cluster a list of bug labels into more concise
#     list. Reduce the number of unique labels by grouping similar labels into 
#     categories and give a descriptive name to each category (in a similar format
#     to the original labels). Output a JSON where the keys are the new labels
#     and the values are arrays of the labels that fall into that cluster.'''

# @user_prompt
# def provide_labels(labels):
#     return f'''{str(labels)}'''

# # print(provide_labels(list(df["label"].values)))
# # exit()
# response = openai_json_response([
#     cluster_labels(),
#     provide_labels(list(df["label"].values))
# ], model="gpt-4o")

# with open(output_path, "w") as fp:
#     json.dump(response, fp)

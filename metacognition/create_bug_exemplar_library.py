from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import difflib
import random
import pickle

random.seed(42)

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from queue import Queue

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

load_dotenv()

# =============== OPENAI UTILITIES ===============

openai_client = OpenAI()

def system_prompt(func):
    def wrapper(*args, **kwargs):
        text = func(*args, **kwargs)
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    return wrapper

def user_prompt(func):
    def wrapper(*args, **kwargs):
        text = func(*args, **kwargs)
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    return wrapper

def get_embedding(query_text, model="text-embedding-3-large", dimensions=256):
   query_text = query_text.replace("\n", " ")
   return openai_client.embeddings.create(input = [query_text], 
                    model=model, dimensions=dimensions).data[0].embedding

def openai_json_response(messages, model="gpt-4o-mini", temp=1, max_tokens=1024):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    return json.loads(response.choices[0].message.content)


# =============== OPENAI PROMPTS ===============

@system_prompt
def label_bug_with_reason():
    return f'''You will be provided with a correct piece of code along with
    a diff that inserts a bug into the code that makes it incorrect. 
    Label this bug with an label that precisely describes the type of bug
    that got introduced. You should be able to use the label as a dictionary
    key in Python. The label should be lower case letters only. The 
    label should be very descriptive and you may use multiple words to
    describe the bug that occurred. If you do use multiple words for the label,
    then join them with an underscore.
    
    Your answer should be in JSON format where key "label" corresponds to your
    naming of the bug and a second key "reason" contains a short justification
    for why you chose the label.'''

@user_prompt
def provide_bug(code_tokens, diff_output):
    return f'''Correct code:\n{code_tokens}\n\nBug insertion:\n{diff_output}'''

@system_prompt
def combine_labels_and_describe():
    return f'''You will be provided with a list of labels for 
    different types of code bugs. Reduce the provided labels
    into a single descriptive label that is an accurate reflection
    of the overall bug category. Use lowercase letters and underscores
    instead of spaces when writing out multiple words in the same
    format as the input. Output the new label using a JSON with the
    key "label" and also provide a key "description" which outlines
    the types of logic or operations or implementation details that
    a Python program would have for this category of bugs to be
    potentially present.'''

@user_prompt
def provide_labels(labels):
    return f'''{labels}'''

# =============== THREADING UTILITY ===============

class MapReduce:
    def __init__(self):
        self.out_queue = Queue()

    def get_items(self):
        return NotImplementedError

    def mapF(self, item):
        return NotImplementedError

    def mapF_helper(self, item):
        out = self.mapF(item)
        self.out_queue.put(out)

    def reduceF(self, results):
        return NotImplementedError

    def run(self, num_workers=8):
        items = self.get_items()
        futures = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for item in items:
                future = executor.submit(self.mapF_helper, item)
                futures.append(future)

            completed_count = 0
            total_futures = len(futures)

            while completed_count < total_futures:
                completed_futures = []

                for future in tqdm(as_completed(futures), total=total_futures):
                    completed_futures.append(future)
                    completed_count += 1

                futures = [f for f in futures if f not in completed_futures]

        results = []
        while not self.out_queue.empty():
            results.append(self.out_queue.get())

        reduced = self.reduceF(results)

        return reduced


# =============== LABELING JOBS ===============

class LabelBugs(MapReduce):
    def __init__(self, data_path: str, n_samples=10000):
        super().__init__()
        self.data_path = data_path
        self.n_samples = n_samples

    def get_items(self):
        file_names = [f"{i}.json" for i in range(116)] 
        data = []

        for file_name in file_names:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, "r") as fp:
                data.extend(json.load(fp))

        def is_valid_pair(pair): # are there other types of errors? 
            # e.g. timeout; i think we only care about wrong answer right now
            return len(pair) == 2 and pair[0]["verdict"] == "Wrong Answer" and pair[1]["verdict"] == "Accepted"

        data = list(filter(lambda pair: is_valid_pair(pair), data))
        
        return random.sample(data, self.n_samples)
    
    def mapF(self, item):
        incorrect_submission, correct_submission = item
        differ = difflib.Differ() # reversed order to seem as if bug is being introduced
        diff = differ.compare(correct_submission['code_tokens'].splitlines(), 
                              incorrect_submission['code_tokens'].splitlines())
        diff_output = '\n'.join(diff)

        response = openai_json_response([
            label_bug_with_reason(),
            provide_bug(incorrect_submission['code_tokens'], diff_output)
        ], model="gpt-4o")

        return {
            "label": response["label"],
            "reason": response["reason"],
            "correct_program": correct_submission['code_tokens'],
            "incorrect_program": incorrect_submission['code_tokens'],
            "diff": diff_output
        }
    
    def reduceF(self, results):
        return results
    

# =============== TFIDF+KMEANS CLUSTERING ===============

def perform_clustering_with_elbow(labels_and_data, seed, k_values):
    df = pd.DataFrame(labels_and_data)

    df['label'] = df['label'].str.replace('_', ' ')

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['label'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    wcss = []

    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit_predict(tfidf_df)
        wcss.append(kmeans.inertia_)
        
    plt.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

    print("How many clusters do you want? ", end="")
    N = int(input())

    kmeans = KMeans(n_clusters=N, random_state=seed, n_init='auto')
    df['cluster'] = kmeans.fit_predict(tfidf_df)

    df['label'] = df['label'].str.replace(' ', '_')

    return df, vectorizer, kmeans


if __name__ == "__main__":
    # # {'Wrong Answer': 668298, 'Runtime Error': 277262, 
    # #  'WA: Presentation Error': 4474, 'Time Limit Exceeded': 208362, 
    # #  'Memory Limit Exceeded': 876, 'Output Limit Exceeded': 17, 
    # #  'Internal error': 24, 'Judge Not Available': 28, 'Judge System Error': 8, 
    # #  'Query Limit Exceeded': 2}
    n_samples = 1000

    data_path = "metacognition/data/python/jsons"
    labels_and_data_path = "metacognition/outputs/labels_and_data.json"
    cluster_df_path = "metacognition/outputs/clustered.csv"
    output_path = "metacognition/outputs/library.json"
    vectorizer_path = "metacognition/outputs/tfidf_vectorizer.pkl"
    kmeans_path = "metacognition/outputs/kmeans_model.pkl"

    job = LabelBugs(data_path, n_samples=n_samples)
    labels_and_data = job.run()
    with open(labels_and_data_path, "w") as fp:
        json.dump(labels_and_data, fp)

    seed = 42
    k_values = range(16, min(len(labels_and_data), 513), 16)

    df, vectorizer, kmeans = perform_clustering_with_elbow(labels_and_data, seed, k_values)
    df.to_csv(cluster_df_path)
    with open(vectorizer_path, "wb") as tfidf_file:
        pickle.dump(vectorizer, tfidf_file)

    with open(kmeans_path, "wb") as kmeans_file:
        pickle.dump(kmeans, kmeans_file)

    bug_exemplars = {}

    label_groupby = df.groupby('cluster')['label'].apply(list).to_dict()
    for labels in tqdm(label_groupby.values()):
        response = openai_json_response([
            combine_labels_and_describe(),
            provide_labels(labels)
        ], model="gpt-4o")

        cluster_label = response["label"]
        cluster_description = response["description"]
        embedding = get_embedding(cluster_description)
        exemplars = df[df["label"].isin(labels)].to_dict("records")

        bug_exemplars[cluster_label] = {
            "description": cluster_description,
            "embedding": embedding,
            "exemplars": exemplars
        }

    with open(output_path, "w") as fp:
        json.dump(bug_exemplars, fp)

    
    # TODO
    '''When you build skill library, save the vectorizer and 
    clustering model so that you can run diversity test by 
    classifying baseline prompt outputs'''

'''
bug_exemplars.json

{
    "accumulator_initialization_error": {
        "description": "... description of what programs could have this error",
        "embedding": [..., ..., ..., ],
        "exemplars": [
            {
                "incorrect_program": "...",
                "diff": "..."
            }
        ]
    },
    ...
}

'''
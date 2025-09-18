import ast
from time import time
import pickle
import re
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts

#from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine
from cutm import MultiOutputTM


parser = argparse.ArgumentParser()
parser.add_argument("--test_set_id", type=int, help="sample id to process")
args = parser.parse_args()

sample_id = args.test_set_id
vectorizer_name = 'count_vectorizer.pickle'
experiment_name = 'no_label_sampling_results_1_30000_2000_2500_5.0_2.0_100'
use_only_positive_importance = False

string_vars = experiment_name.split('_')
n_gram = int(string_vars[-7])
max_features = int(string_vars[-6])
n_clauses_per_class = int(string_vars[-5])

# Read the dataset file
filepath = '../../data/stortinget_dataset_clean.csv'
df = pd.read_csv(filepath)
df['emneord'] = df['emneord'].apply(ast.literal_eval)

# Create dictionary with 10 most common tags
all_emneord = np.sort(np.unique(df['emneord'].sum()))
emnedict = {str(k): v for v, k in zip(range(len(all_emneord)), all_emneord)}
tags, counts = np.unique(df['emneord'].sum(), return_counts=True)
common_tags_ind = np.argsort(-counts)[:10]

print("Most common labels\n", "-"*20)
emnedict_reduced = {}
for i, tag in enumerate(common_tags_ind):
    print(f"{tags[tag]} -- {counts[tag]}")
    emnedict_reduced[str(tags[tag])] = i
print(f"\nReduced emnedict: {emnedict_reduced}")

# Split datafram into train and test sets. Don't change the seed!
train, test = tts(df, test_size=0.2, random_state=1)
test_samples = test.index
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)



if not vectorizer_name in os.listdir():
    
    # Initialize the count-vectorizer and transform the data
    with open("stop_words.txt", 'r') as infile:
        stop_words = [word.strip() for word in infile]
    
    corpus = list(train.loc[:, 'sammendrag'].values)
    vect = CountVectorizer(
        ngram_range=(1, n_gram),
        max_features=max_features,
        binary=True,
        stop_words=stop_words,
    )

    tic = time()
    vect.fit(corpus)
    toc = time()
    print(f"Vectorizer trained in {toc - tic:.3f} seconds")

    with open(vectorizer_name, 'wb') as outfile:
        pickle.dump(vect, outfile)

else:
    corpus = list(train.loc[:, 'sammendrag'].values)
    with open(vectorizer_name, 'rb') as infile:
        vect = pickle.load(infile)

tic = time()
X_train = vect.transform(corpus).toarray()
toc = time()
print(f"Corpus transformed in {toc-tic:.3f} seconds")

tic = time()
X_test = vect.transform(list(test.loc[:, 'sammendrag'].values)).toarray()
toc = time()
print(f"Test data transformed in {toc-tic:.3f} seconds")


# Get the encoded labels from the dataframes
emneset = set(emnedict_reduced.keys())
def get_target_vect(df, emneset):
    Y = np.zeros([df.shape[0], len(emneset)], dtype=int)
    for i in range(df.shape[0]):
        labels = set(df.loc[i, 'emneord']).intersection(emneset)
        for l in labels:
            Y[i, emnedict_reduced[l]] = 1
    return Y

Y_train = get_target_vect(train.iloc[:], emneset)
Y_test = get_target_vect(test.iloc[:], emneset)

# Remove all samples without any labels
train = train[np.sum(Y_train, axis=1) > 0].reset_index(drop=True)
test = test[np.sum(Y_test, axis=1) > 0].reset_index(drop=True)
X_train = X_train[np.sum(Y_train, axis=1) > 0].astype(np.uint32)
Y_train = Y_train[np.sum(Y_train, axis=1) > 0].astype(np.uint32)
X_test = X_test[np.sum(Y_test, axis=1) > 0].astype(np.uint32)
Y_test = Y_test[np.sum(Y_test, axis=1) > 0].astype(np.uint32)

# Create reversed emne-dictionary
reverse_emne = {v:k for k, v in emnedict_reduced.items()}

# Load the trained Tsetlin Machine model
FILENAME = f"{experiment_name}/model.pickle"
with open(FILENAME, 'rb') as infile:
    tm = pickle.load(infile)

# Get literals and weights
L = tm.get_literals()    # Matrix of TM literals
W = tm.get_weights()     # Matrix of TM weights

# Useful variables
n_feat = L.shape[-1] // 2
word_dict = vect.get_feature_names_out()

# Transform the sample and make predictions
x = vect.transform(list(test.loc[sample_id:sample_id, 'sammendrag'])).toarray().astype(np.uint32)
x_transformed = tm.transform(x.astype(np.uint32))[0].reshape(W.shape[0], n_clauses_per_class)
y_pred, cs = tm.predict(x)
y_pred = y_pred[0]
predicted_emner = np.argwhere(y_pred == 1).flatten()

importances = []
included_words = []
word_importances = []
for target in predicted_emner:
    xt = x_transformed[target]
    A = L[target, xt == 1, :]   # Active literals
    WA = W[target, xt==1, target]
    positive = A[WA >= 0]    # positive polarity clauses
    WP = WA[WA >= 0 ]

    if use_only_positive_importance:
        importance = (positive * WP.reshape(-1, 1)).sum(axis=0)
    else:
        importance = (A * WA.reshape(-1, 1)).sum(axis=0)
    
    literal_indices = []
    for clause_id in range(positive.shape[0]):
        for literal in range(n_feat):
            pos_literal_value = positive[clause_id][literal]
            if pos_literal_value == 1:
                literal_indices.append(literal)
                
    unique_literals = sorted(list(set(literal_indices)))
    included_words.append([word_dict[x] for x in unique_literals])
    word_importances.append([float(importance[x]) for x in unique_literals])



output_folder = f"samples/{sample_id}"
os.makedirs(output_folder, exist_ok=True)

# Store results as csv files
pd.DataFrame(included_words, index=[reverse_emne[i] for i in predicted_emner]).to_csv(f"{output_folder}/sample_clauses_id_{sample_id}.csv")
pd.DataFrame(word_importances, index=[reverse_emne[i] for i in predicted_emner]).to_csv(f"{output_folder}/sample_importance_id_{sample_id}.csv")
test.loc[sample_id].to_csv(f"{output_folder}/sample_data_id_{sample_id}.csv")

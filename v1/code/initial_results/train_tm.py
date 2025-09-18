import argparse
import ast
from time import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts

from cutm import MultiOutputTM


def get_target_vect(df, emnedict):
    emneset = set(emnedict.keys())
    Y = np.zeros([df.shape[0], len(emneset)], dtype=int)
    for i in range(df.shape[0]):
        labels = set(df.loc[i, 'emneord']).intersection(emneset)
        for l in labels:
            Y[i, emnedict[l]] = 1
    return Y


if __name__ == '__main__':

    # =============
    # Set arguments
    # =============
    MAX_FEATURES = 30000
    N_GRAM = 1
    N_LABELS = 10
    N_CLAUSES = 2000
    T = 2500
    s = 5.0
    q = 2.0
    MAX_INC_LITERALS = 32
    EPOCHS = 100
    LABEL_SAMPLING = False
    
    RESULT_FOLDER = 'results'

    
    if LABEL_SAMPLING:
        FOLDER = f"results_{N_GRAM}_{MAX_FEATURES}_{N_CLAUSES}_{T}_{s}_{q}_{EPOCHS}"
    else:
        FOLDER = f"no_label_sampling_results_{N_GRAM}_{MAX_FEATURES}_{N_CLAUSES}_{T}_{s}_{q}_{EPOCHS}"
    
    os.makedirs(FOLDER, exist_ok=True)
    
    
    # =============
    # Read the data
    # =============
    filepath = '../../data/stortinget_dataset_clean.csv'
    df = pd.read_csv(filepath)
    df['emneord'] = df['emneord'].apply(ast.literal_eval)
    
    # Preprocessing
    all_emneord = np.sort(np.unique(df['emneord'].sum()))
    emnedict = {str(k): v for v, k in zip(range(len(all_emneord)), all_emneord)}
    print(f"Number of unique labels: {len(emnedict)}")
    
    tags, counts = np.unique(df['emneord'].sum(), return_counts=True)
    common_tags_ind = np.argsort(-counts)[:N_LABELS]
    print("Most common labels\n", "-"*20)
    
    emnedict_reduced = {}
    for i, tag in enumerate(common_tags_ind):
        print(f"{tags[tag]} -- {counts[tag]}")
        emnedict_reduced[str(tags[tag])] = i
    print(f"\nReduced emnedict: {emnedict_reduced}")
    
    
    # Prepare the data
    train, test = tts(df, test_size=0.2, random_state=1)
    test_samples = test.index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    

    # ========================
    # Fit the count vectorizer
    # ========================
    # Get the stop-words
    with open("stop_words.txt", 'r') as infile:
        stop_words = [word.strip() for word in infile]
    
    corpus = list(train.loc[:, 'sammendrag'].values)
    vect = CountVectorizer(
        ngram_range=(1, N_GRAM),
        max_features=MAX_FEATURES,
        binary=True,
        stop_words=stop_words
    )
    
    tic = time()
    vect.fit(corpus)
    toc = time()
    print(f"Vectorizer trained in {toc - tic:.3f} seconds")
    
    tic = time()
    X_train = vect.transform(corpus).toarray()
    toc = time()
    print(f"Corpus transformed in {toc-tic:.3f} seconds")
    
    tic = time()
    X_test = vect.transform(list(test.loc[:, 'sammendrag'].values)).toarray()
    toc = time()
    print(f"Test data transformed in {toc-tic:.3f} seconds")
    
    with open(f"{FOLDER}/vectorizer.pickle", 'wb') as outfile:
        pickle.dump(vect, outfile)
    
    # Get the encoded labels
    Y_train = get_target_vect(train.iloc[:], emnedict_reduced)
    Y_test = get_target_vect(test.iloc[:], emnedict_reduced)
    
    # Remove samples without any labels
    X_train = (X_train[np.sum(Y_train, axis=1) > 0]).astype(np.uint32)
    Y_train = (Y_train[np.sum(Y_train, axis=1) > 0]).astype(np.uint32)
    X_test = (X_test[np.sum(Y_test, axis=1) > 0]).astype(np.uint32)
    Y_test = (Y_test[np.sum(Y_test, axis=1) > 0]).astype(np.uint32)
    
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of test samples:     {X_test.shape[0]}")
    
    dim = (X_train.shape[1], 1, 1)
    
    # =========================
    # Train the Tsetlin machine
    # =========================

    tm = MultiOutputTM(
        number_of_clauses_per_class=N_CLAUSES,
        dim=dim,
        n_classes=N_LABELS,
        T=T,
        s=s,
        q=q,
        coalesced=False,
        max_included_literals=MAX_INC_LITERALS,
    )
    
    train_res = f"{FOLDER}/{RESULT_FOLDER}/train"
    test_res = f"{FOLDER}/{RESULT_FOLDER}/test"
    
    os.makedirs(train_res, exist_ok=True)
    os.makedirs(test_res, exist_ok=True)
    
    def get_results(Y_true, Y_pred):
        result_dict = classification_report(
            Y_true,
            Y_pred,
            target_names=emnedict_reduced.keys(),
            zero_division=np.nan,
            output_dict=True,
        )
        return pd.DataFrame(result_dict)
    
    
    for i in tqdm(range(EPOCHS)):
        tm.fit(X_train, Y_train, epochs=1, label_sampling=LABEL_SAMPLING)
        Y_pred_train, cs = tm.predict(X_train)
        train_df = get_results(Y_train, Y_pred_train)
        train_df.to_csv(f"{train_res}/results_{i}")
        Y_pred_test, cs = tm.predict(X_test)
        test_df = get_results(Y_test, Y_pred_test)
        test_df.to_csv(f"{test_res}/results_{i}")
        print(f"Epoch {i:>3}    Train: {train_df.loc['f1-score', ['weighted avg']].values[0]:.3f}    Test: {test_df.loc['f1-score', ['weighted avg']].values[0]:.3f}")

        with open(f"{FOLDER}/model.pickle", 'wb') as outfile:
            pickle.dump(tm, outfile)
    
    print("Done")
import pandas as pd
import os
import pickle
from cutm import MultiOutputTM
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts
import numpy as np
from time import time


def get_target_vect(labels, emnedict):
    emneset = set(emnedict.keys())
    Y = np.zeros([len(labels), len(emneset)], dtype=int)
    for i, v in enumerate(labels):
        lab_set = set(v).intersection(emneset)
        for l in lab_set:
            Y[i, emnedict[l]] = 1
    return Y

def get_results(Y_true, Y_pred):
    result_dict = classification_report(
        Y_true,
        Y_pred,
        target_names=emnedict_reduced.keys(),
        zero_division=np.nan,
        output_dict=True,
    )
    return pd.DataFrame(result_dict)


# Get the document names
documents = sorted([x for x in os.listdir('sammendrag')])
doc_names = [x.lower().removesuffix('.txt') for x in documents]
documents = [f"sammendrag/{x}" for x in documents]

# Get the labels
filename = "InnstillingerMedEmneord.csv"
df = pd.read_csv(filename, sep=';')
df['Filnavn'] = df['Filnavn'].apply(lambda x: x.lower())

labels = []
for doc_name in doc_names:
    labels.append([x.lower() for x in df[df['Filnavn'] == f"{doc_name}.xml"]['emneord'].values])

all_labels = []
for x in labels:
    all_labels += x



N_LABELS = 10

tags, counts = np.unique(all_labels, return_counts=True)
common_tags_ind = np.argsort(-counts)[:N_LABELS]
print("Most common labels\n", "-"*20)
    
emnedict_reduced = {}
for i, tag in enumerate(common_tags_ind):
    print(f"{tags[tag]} -- {counts[tag]}")
    emnedict_reduced[str(tags[tag])] = i
print(f"\nReduced emnedict: {emnedict_reduced}")

Y = get_target_vect(labels, emnedict_reduced)
non_empty_rows = np.arange(Y.shape[0])[Y.sum(axis=1) != 0]

# Split into train-test
train_ind, test_ind = tts(non_empty_rows, test_size=0.25, random_state=1)
train_doc = (np.array(documents)[train_ind]).tolist()
test_doc = np.array(documents)[test_ind]
Y_train = Y[train_ind].astype(np.uint32)
Y_test = Y[test_ind].astype(np.uint32)


# ========================
# Fit the count vectorizer
# ========================
vect = CountVectorizer(
    input='filename',
    ngram_range=(1, 2),
    max_features=30000,
    binary=True,
)

tic = time()
vect.fit(train_doc)
toc = time()
print(f"Vectorizer trained in {toc - tic:.3f} seconds")

tic = time()
X_train = vect.transform(train_doc).toarray().astype(np.uint32)
toc = time()
print(f"Corpus transformed in {toc-tic:.3f} seconds")

tic = time()
X_test = vect.transform(test_doc).toarray().astype(np.uint32)
toc = time()
print(f"Corpus transformed in {toc-tic:.3f} seconds")



dim = (X_train.shape[1], 1, 1)
N_CLAUSES = 3000
T = 3100
s = 5.0
q = 4.0
MAX_INC_LITERALS = 32
EPOCHS = 12
LABEL_SAMPLING = False


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

rng = np.random.default_rng(1)
order = np.arange(X_train.shape[0])

for i in range(EPOCHS):
    rng.shuffle(order)
    tm.fit(X_train[order], Y_train[order], epochs=1, label_sampling=LABEL_SAMPLING)
    Y_pred_train, cs = tm.predict(X_train)
    train_df = get_results(Y_train, Y_pred_train)
    #train_df.to_csv(f"{train_res}/results_{i}")
    Y_pred_test, cs = tm.predict(X_test)
    test_df = get_results(Y_test, Y_pred_test)
    #test_df.to_csv(f"{test_res}/results_{i}")
    print(f"Epoch {i:>3}    Train: {train_df.loc['f1-score', ['weighted avg']].values[0]:.3f}    Test: {test_df.loc['f1-score', ['weighted avg']].values[0]:.3f}")
    print(classification_report(
        Y_test,
        Y_pred_test,
        target_names=emnedict_reduced.keys(),
        zero_division=np.nan,
    ))
    with open(f"tm_model.pickle", 'wb') as outfile:
        pickle.dump(tm, outfile)

import numpy as np
import pandas as pd
from tqdm import tqdm

import ast


def check_unique(sakid):
    if sakid in unique_ids or sakid in batch_ids:
        return np.nan
    else:
        batch_ids.append(sakid)
        return 1
    
def get_unique_text_number(url):
    url_id = url.split('/')[-3]
    if url_id not in unique_urls:
        unique_urls.append(url_id)
    return len(unique_urls)


filepath = '../data/stortinget_dataset.csv'
newfile = '../data/stortinget_dataset_clean.csv'

batch_num = 0
batch_size = 1000
total_number_of_documents = 17221
s = 0
unique_ids = []

names = pd.read_csv(filepath, nrows=0, skiprows=0)
names.to_csv(newfile, index=False, header=True, mode='w')
names = names.columns

for batch_num in tqdm(range((total_number_of_documents // batch_size) + 1)):
    batch_ids = []
    df = pd.read_csv(filepath, nrows=batch_size, skiprows=(batch_size * batch_num) + 1, names=names, header=None)
    df= df.dropna(subset=['sammendrag', 'url', 'html'], axis=0)

    # Check if document is in file already
    old = df.shape[0]
    df['is_unique'] = df['sakid'].apply(check_unique)
    df = df.dropna(subset=['is_unique'], axis=0)
    df = df.drop(columns=['is_unique'])
    print(f"Removing {old - df.shape[0]} duplicated samples")
    unique_ids += list(df['sakid'])

    # Merge labels for identical texts
    old = df.shape[0]
    df['emneord'] = df['emneord'].apply(ast.literal_eval)
    unique_urls = []
    df['text_id'] = df['url'].apply(get_unique_text_number)
    
    text_emneord = []
    for i in df.groupby('text_id')['emneord']:
        text_emneord.append(list(set(i[1].sum())))

    df['emneord'] = df['text_id'].apply(lambda x: text_emneord[x-1])
    _, indices = np.unique(df['text_id'], return_index=True)
    df = df.iloc[indices, :]
    df = df.drop(columns=['text_id'])
    print(f"Merged {old - df.shape[0]} samples")

    s += df.shape[0]
    print(f"Documents written: {df.shape[0]}")
    
    df.to_csv(newfile, index=False, header=False, mode='a')


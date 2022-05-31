import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def transform_hash_to_number(df: pd.DataFrame,
                             columns: list) -> pd.DataFrame:
    """ Transform data frame columns containing hash strings to
    numeric values"""
    
    for column in columns:
        name_dictionary = dict((y,x+1) for x,y in enumerate(sorted(set(df[column]))))
        df[column] = pd.Series([name_dictionary[x] for x in df[column]]).reset_index(drop=True)

    return df

def load_data_all(path="./data/transactions_simple.csv", 
                  header=['id', 'customer_id', 'article_id'],
                  test_size=0.2, 
                  sep=","):
    """ Loads the data from 'transactions.csv' and transform it into
    usable form """

    df = transform_hash_to_number(pd.read_csv(path, sep=sep, names=header, engine='python'), \
        ['customer_id', 'article_id'])

    n_users = df.customer_id.unique().shape[0]
    n_items = df.article_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[2] - 1
        i = line[3] - 1
        train_dict[(u, i)] = 1

    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                train_rating.append(1)
            else:
                train_rating.append(0)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    neg_items = {}
    train_interaction_matrix = []
    for u in range(n_users):
        neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[2] - 1)
        test_col.append(line[3] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_interaction_matrix, test_dict, n_users, n_items


def load_data_neg(path="./data/transactions_simple.csv", 
                  header=['id', 'customer_id', 'article_id'],
                  test_size=0.2, 
                  sep=","):

    """ Loads the data from 'transactions.csv' and transform it into
    usable form (ommit rating data) """
                  
    df = transform_hash_to_number(pd.read_csv(path, sep=sep, names=header, engine='python'), \
        ['customer_id', 'article_id'])
 
    n_users = df.customer_id.unique().shape[0]
    n_items = df.article_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[2] - 1
        i = line[3] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[2] - 1)
        test_col.append(line[3] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items


def subset_data(customers: pd.DataFrame,
                articles: pd.DataFrame,
                transactions: pd.DataFrame,
                n: int,
                output_path: str):

    articles_simple = articles.iloc[1:n,]

    transactions_simple = transactions.merge(articles_simple, on="article_id").loc[:,list(transactions.columns)]

    customers_simple = customers.merge(transactions_simple, on="customer_id").loc[:,list(customers.columns)].drop_duplicates("customer_id")

    transactions_simple[['customer_id','article_id']].to_csv(output_path + "/transactions_" + str(n) + ".csv")
    customers_simple.to_csv(output_path + "/customers_" + str(n) + ".csv")
    articles_simple.to_csv(output_path + "/articles_" + str(n) + ".csv")

    return transactions_simple

def save_name_dict_to_csv(df: pd.DataFrame,
                          output_path: str):

    name_dictionary_df = pd.DataFrame(
        {'article_original': [],
         'article_new': []})

    for column in ["customer_id", "article_id"]:
        name_dictionary = dict((y,x+1) for x,y in enumerate(sorted(set(df[column]))))
        df[column] = pd.Series([name_dictionary[x] for x in df[column]]).reset_index(drop=True)

    for key, val in name_dictionary.items():
        name_dictionary_df
        name_dictionary_df.loc[len(name_dictionary_df.index)] = [key, val]

    name_dictionary_df = name_dictionary_df.astype('int')
 
    name_dictionary_df.to_csv(output_path)
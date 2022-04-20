import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import tensorflow as tf
from Models.CDAE import CDAE 


def load_data_all(path="./Data/transactions_simple.csv", header=['id', 'customer_id', 'article_id'],
                  test_size=0.2, sep=","):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    cusotmers_dict = dict((y,x+1) for x,y in enumerate(sorted(set(df.customer_id))))
    acrticles_dict = dict((y,x+1) for x,y in enumerate(sorted(set(df.article_id))))

    df.customer_id =  pd.Series([cusotmers_dict[x] for x in df.customer_id]).reset_index(drop=True)
    df.article_id = pd.Series([acrticles_dict[x] for x in df.article_id]).reset_index(drop=True)

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


def load_data_neg(path="./Data/transactions_simple.csv", header=['id', 'customer_id', 'article_id'],
                  test_size=0.2, sep=","):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')
 
    

    cusotmers_dict = dict((y,x+1) for x,y in enumerate(sorted(set(df.customer_id))))
    acrticles_dict = dict((y,x+1) for x,y in enumerate(sorted(set(df.article_id))))

    df.customer_id =  pd.Series([cusotmers_dict[x] for x in df.customer_id]).reset_index(drop=True)
    df.article_id = pd.Series([acrticles_dict[x] for x in df.article_id]).reset_index(drop=True)
    
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

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

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


def load_data_separately(path_train=None, path_test=None, path_val=None, header=['customer_id', 'article_id', 'rating'],
                         sep=" ", n_users=0, n_items=0):
    n_users = n_users
    n_items = n_items
    print("start")
    train_matrix = None
    if path_train is not None:
        train_data = pd.read_csv(path_train, sep=sep, names=header, engine='python')
        print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

        train_row = []
        train_col = []
        train_rating = []

        for line in train_data.itertuples():
            u = line[2]  # - 1
            i = line[3]  # - 1
            train_row.append(u)
            train_col.append(i)
            train_rating.append(1)

        train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    test_dict = None
    if path_test is not None:
        test_data = pd.read_csv(path_test, sep=sep, names=header, engine='python')
        test_row = []
        test_col = []
        test_rating = []
        for line in test_data.itertuples():
            test_row.append(line[2])
            i = line[3]  # - 1
            test_col.append(i)
            test_rating.append(1)

        test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

        test_dict = {}
        for u in range(n_users):
            test_dict[u] = test_matrix.getrow(u).nonzero()[1]
    all_items = set(np.arange(n_items))
    train_interaction_matrix = []
    for u in range(n_users):
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    if path_val is not None:
        val_data = pd.read_csv(path_val, sep=sep, names=header, engine='python')

    print("end")
    return train_interaction_matrix, test_dict, n_users, n_items

if __name__ == "__main__":
    """ Folder Structure
    |--WTUM
    |   |--main.py
    |   |--Data
    |   |   |--customers.csv
    |   |   |--articles.csv
    |   |   |--transactions_train.csv
    |   |--Models
    """
    # print("wczytywanie danych")
    # # Load data 
    # customers = pd.read_csv("./Data/customers.csv")

    # # Change NaN to 0 in customers table
    # customers["FN"] = np.nan_to_num(customers["FN"])
    # customers["Active"] = np.nan_to_num(customers["Active"])
    
    # articles = pd.read_csv("./Data/articles.csv")        

    # transactions = pd.read_csv("./Data/transactions_train.csv") 
    # print("Tworzenie mniejszych zbiorow danych")
    
    # n = 100
    # articles_simple = articles.iloc[1:n,]
    
    # transactions_simple = transactions.merge(articles_simple, on="article_id").loc[:,list(transactions.columns)]
    
    # customers_simple = customers.merge(transactions_simple, on="customer_id").loc[:,list(customers.columns)].drop_duplicates("customer_id")
    
    # transactions_simple[['customer_id','article_id']].to_csv("Data/transactions_simple.csv")
    # customers_simple.to_csv("Data/customers_simple.csv")
    # articles_simple.to_csv("Data/articles_simple.csv")
    
    
    # args = argpars.parse_args()
    # epochs = args.epochs
    # learning_rate = args.learning_rate
    # reg_rate = args.reg_rate
    # num_factors = args.num_factors
    # display_step = args.display_step
    # batch_size = args.batch_size
    
    print("Testowanie")
    train_data, test_data, n_user, n_item = load_data_neg(test_size=0.2, sep=",")
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
    model = None
    train_data, test_data, n_user, n_item = load_data_all(test_size=0.2, sep=",")
    model = CDAE(n_user, n_item)

    if model is not None:
        model.build_network()
        model.execute(train_data, test_data)

    
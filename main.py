import numpy as np
import pandas as pd
import sqlite3

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

    # Load data 
    customers = pd.read_csv("./Data/customers.csv")

    # Change NaN to 0 in customers table
    customers["FN"] = np.nan_to_num(customers["FN"])
    customers["Active"] = np.nan_to_num(customers["Active"])
    
    n = 10000
    articles = pd.read_csv("./Data/articles.csv")        
    articles_simple = articles.iloc[1:n,]
    
    transactions = pd.read_csv("./Data/transactions_train.csv") 
    
    transactions_simple = transactions.merge(articles_simple, on="article_id").loc[:,list(transactions.columns)]
    
    customers_simple = customers.merge(transactions_simple, on="customer_id").loc[:,list(customers.columns)].drop_duplicates("customer_id")
    
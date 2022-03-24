import numpy as np
import pandas as pd


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
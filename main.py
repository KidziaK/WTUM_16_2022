import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Load data
    customers = pd.read_csv("./Dane/customers.csv")

    # Change NaN to 0 in customers table
    customers["FN"] = np.nan_to_num(customers["FN"])
    customers["Active"] = np.nan_to_num(customers["Active"])
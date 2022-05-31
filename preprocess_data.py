from functions.utility import subset_data, save_name_dict_to_csv
import pandas as pd
import numpy as np
import json


if __name__ == "__main__":

    file =  open('configuration.json')

    config = json.load(file)

    file.close()

    print("Loading customers")
    customers = pd.read_csv("./Data/customers.csv")

    customers["FN"] = np.nan_to_num(customers["FN"])
    customers["Active"] = np.nan_to_num(customers["Active"])

    print("Loading articles")
    articles = pd.read_csv("./Data/articles.csv")        

    print("Loading transactions")
    transactions = pd.read_csv("./Data/transactions_train.csv") 


    print("Subsetting")
    transactions_simple = subset_data(customers=customers,
                articles=articles,
                transactions=transactions,
                n=config["preprocessing_n"],
                output_path='./Data')

    print("Unhashing")
    save_name_dict_to_csv(df=transactions_simple[["customer_id", "article_id"]], 
                          output_path="./Data/article_dict.csv")
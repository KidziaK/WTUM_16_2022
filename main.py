import json

import tensorflow as tf

from functions.utility import load_data_all
from models.CDAE import CDAE 

if __name__ == "__main__":
    """ Main script used to train model"""

    file =  open('configuration.json')

    config = json.load(file)

    file.close()

    try:
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_memory_growth(gpus[0], True)
    except:
        print("Couldn't use GPU.")
        pass
    
    tf.compat.v1.disable_eager_execution()

    epoch = config["epoch"]
    train_data, test_data, n_user, n_item = load_data_all(test_size=config["test_size"], 
                                                          sep=",", 
                                                          path="./data/" + config["transaction_table_name"])
    sess = tf.compat.v1.Session()
    model = CDAE(sess, n_user, n_item, epoch=epoch, path="./models/trained/")

    if model is not None:
        model.build_network()
        model.execute(train_data, test_data)
        model.save(name="CDAE_{}_epoch".format(epoch))
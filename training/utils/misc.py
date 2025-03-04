import pandas as pd
import os


def skip_save(function):
    def _wrapper(*args,**kwargs):
        try:
            function(*args,**kwargs)
        except Exception as e:
            print("DID NOT SAVE!", str(e))
    return _wrapper


@skip_save
def save_predictions(path, name, predictions):
    # Make the directory if it does not exist already
    if not os.path.exists(path):
        os.makedirs(path)

    save_name = os.path.join(path, f"{name}_predictions.csv")
    predictions.to_csv(save_name, index=False)
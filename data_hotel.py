import encodings
from numpy import unicode_
import pandas as pd 

def load_data(data):
	df = pd.read_csv(data)
	return df 

def get_list_hotel():
    df = load_data("list-hotel/list-hotels.csv")
    results_json = df.to_json(orient ='table')
    return(results_json)
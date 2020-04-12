import io

import pandas as pd
import requests

url = "https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), nrows=10, index_col=0)
df
df.shape

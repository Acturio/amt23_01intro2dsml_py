import pandas as pd
import pandas
import numpy as np
from pandas import read_csv, read_excel

import koalas as ko
#### lectura de archivo .csv ####

#ames_csv = pd.read_csv("~/Documents/Freelance/amat/amt23_01intro2dsml_py/data/ames.csv")
data_csv = pandas.read_csv("data/ames.csv")
data_csv = pd.read_csv("data/ames.csv")
data_csv = read_csv("data/ames.csv")

pd.read_csv
ko.read_csv

#del data_csv
data_csv

data_csv.info()

pd.set_option('display.max_columns', 6)
data_csv.head(5)
data_csv.tail(5)

data_csv.describe()


#### lectura de archivo .txt ####

ames_txt = pd.read_csv("data/ames.txt", delimiter = ";")
ames_txt.head(3)


#### lectura de archivo .xlsx ####

ames_xlsx = pd.read_excel("data/ames.xlsx")
ames_xlsx.head(3)


#### lectura de archivo .pkl ####

data_csv.to_pickle("data/ames.pkl")

ames_pkl = pd.read_pickle("data/ames.pkl")
ames_pkl.head(5)



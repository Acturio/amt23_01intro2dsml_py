import pandas as pd
from siuba import *
import plydata as pr
import pickle
from plydata.tidy import pivot_wider, pivot_longer

ames_housing = pd.read_csv("data/ames.csv")

(
  ames_housing >>
  mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >>
  mutate(Antique_status = if_else(_.Antique < 10, "new", "old") )
)

(
  ames_housing >>
  mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >>
  mutate(Antique_status = if_else(_.Antique < 10, "new", "old") ) >>
  pr.group_by("Antique_status") >>
  pr.tally()
)

#### pivote horizontal ####


loc_mun_cdmx = pd.read_pickle('data/loc_mun_cdmx.pkl')


loc_mun_cdmx

(
loc_mun_cdmx >>
    pivot_wider(names_from = "Ambito", values_from = "Total_localidades")
)

(
loc_mun_cdmx >>
    pivot_wider(
     names_from = "Ambito", 
     values_from = "Total_localidades", 
     values_fill = 0
    )
)



# with open('data/us_rent_income.pkl', 'rb') as f:
#     us_rent_income = pickle.load(f)

us_rent_income = pd.read_pickle('data/us_rent_income.pkl')
us_rent_income


(
us_rent_income >>
    select(-_.GEOID)
)

pd.set_option('display.max_columns', 6)
(
us_rent_income >>
    #select(-_.GEOID) >>
    pivot_wider(
     names_from = "variable", 
     values_from = ["estimate", "moe"],
     id_cols = "GEOID"
    )
)

#### Pivote vertical ####

relig_income = pd.read_pickle('data/relig_income.pkl')
relig_income

(
 relig_income >>
 pivot_longer(
  cols = ['<$10k', '$10-20k', '$20-30k', '$30-40k', '$40-50k', '$50-75k',
          '$75-100k', '$100-150k', '>150k', "Don't know/refused"], 
  names_to = "rango", 
  values_to = "casos")
)



























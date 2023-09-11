import pandas as pd
from siuba import *
import plydata as pr
import pickle
from plydata.tidy import pivot_wider, pivot_longer

ames_housing = pd.read_csv("data/ames.csv")

(
  ames_housing >>
  mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >>
  mutate(Antique_status = if_else(_.Antique < 10, "new", "old") ) >>
  pr.group_by("Antique_status") >>
  pr.tally()
)

#### pivote horizontal ####

with open('data/loc_mun_cdmx.pkl', 'rb') as f:
    loc_mun_cdmx = pickle.load(f)


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



with open('data/us_rent_income.pkl', 'rb') as f:
    us_rent_income = pickle.load(f)

us_rent_income

(
us_rent_income >>
    select(-_.GEOID) >>
    pivot_wider(
     names_from = "variable", 
     values_from = ["estimate", "moe"]
    )
)

#### Pivote vertical ####

with open('data/relig_income.pkl', 'rb') as f:
    relig_income = pickle.load(f)

relig_income


(
 relig_income >>
 pivot_longer(
  cols = ['<$10k', '$10-20k', '$20-30k', '$30-40k', '$40-50k', '$50-75k',
          '$75-100k', '$100-150k', '>150k', "Don't know/refused"], 
  names_to = "income", 
  values_to = "count")
)



























import pandas as pd
from siuba import *


ames_housing = pd.read_csv("data/ames.csv")

ames_housing.info()
ames_housing.describe()


##### Selección de columnas ######

(
ames_housing >>
  select(_.Lot_Area, _.Neighborhood, _.Year_Sold, _.Sale_Price)
)

nueva_tabla = (
  ames_housing >>
    select(_.Longitude, _.Latitude)
)

(
 ames_housing >> 
  select(_.contains("Area"))
 )


(
 ames_housing >> 
 select(_.contains("^Garage"))
)

(
 ames_housing >> 
 select(_.contains("Area$"))
)


##### Filtrar observaciones ####

ames_housing['Sale_Condition'].value_counts()
ames_housing['Street'].value_counts()

(
  ames_housing >> 
  filter(_.Sale_Condition == "Normal")
)

(
  ames_housing >> 
  filter( (_.Lot_Area > 1000) & (_.Sale_Price >= 150000) )
)

(
  ames_housing >> 
  filter((_.Lot_Area < 1000) | (_.Sale_Price <= 150000))
)


(
ames_housing >>
  filter( ((_.Street.isin(["Pave", "Grvl"]) ) & (_.Sale_Price < 10000000)) | (_.Lot_Area < 1000) )
)

#### Ordenar registros ####

(
  ames_housing 
  >> arrange(_.Sale_Price)
  >> select(_.Sale_Price)
)

(
  ames_housing >> 
  arrange(-_.Sale_Price)
  >> select(_.Sale_Price)
)

(
ames_housing >> 
 arrange(_.Sale_Condition, -_.Sale_Price, _.Lot_Area) >>
 select(_.Sale_Condition, _.Sale_Price, _.Lot_Area)
)


#### Agregar o modificar columnas ####

ejemplo_mutate = (
 ames_housing >> 
   select(_.Year_Sold, _.Year_Remod_Add) >>
   mutate(Antique = _.Year_Sold - _.Year_Remod_Add)
)

ejemplo_mutate

(
ejemplo_mutate >> 
 mutate(Antique = _.Antique * 12)
)


#### Agregaciones ####

(
ames_housing >> 
 select(_.Year_Sold, _.Year_Remod_Add) >>
 mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >>
 arrange(_.Antique) >>
 summarize(
  Mean_Antique = _.Antique.mean(),
  Median_Antique = _.Antique.median(),
  First_Antique = _.Antique.iloc[0],
  Last_Antique = _.Antique.iloc[-1],
  )
)


#### Agrupamiento ####

(
ames_housing >> 
 mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >> 
 group_by(_.Neighborhood, _.Overall_Cond) >> 
 summarize(Mean_Antique = _.Antique.mean().round(0) )
)


(
ames_housing >> 
 mutate(Antique = _.Year_Sold - _.Year_Remod_Add) >> 
 group_by(_.Overall_Cond) >> 
 summarize(Mean_Sale_Price = _.Sale_Price.mean().round(0) ) >>
 arrange(_.Mean_Sale_Price)
)












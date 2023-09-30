# pip install mlxtend==0.23.0
from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from plydata.one_table_verbs import pull
from sklearn.model_selection import train_test_split
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import statsmodels.api as sm


#### CARGA DE DATOS ####
ames = pd.read_csv("data/ames.csv")

ames_y = ames >> pull("Sale_Price")    # ames[["Sale_Price"]]
ames_x = select(ames, -_.Sale_Price)   # ames.drop('Sale_Price', axis=1)

#### DIVISIÓN DE DATOS ####
ames_x_train, ames_x_test, ames_y_train, ames_y_test = train_test_split(
 ames_x, ames_y, 
 test_size = 0.20, 
 random_state = 195
 )


#### FEATURE ENGINEERING ####

## SELECCIÓN DE VARIABLES

# Seleccionamos las variales numéricas de interés
num_cols = ["Full_Bath", "Half_Bath"]

# Seleccionamos las variables categóricas de interés
cat_cols = ["Overall_Cond"]

# Juntamos todas las variables de interés
columnas_seleccionadas = num_cols + cat_cols

pipe = ColumnSelector(columnas_seleccionadas)
ames_x_train_selected = pipe.fit_transform(ames_x_train)

ames_train_selected = pd.DataFrame(
  ames_x_train_selected, 
  columns = columnas_seleccionadas
  )

ames_train_selected.info()


## TRANSFORMACIÓN DE COLUMNAS

# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('onehotencoding', OneHotEncoder(drop='first'), cat_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
)

transformed_data = preprocessor.fit_transform(ames_train_selected)
new_column_names = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
  transformed_data.todense(), 
  columns=new_column_names
  )
  
transformed_df
transformed_df.info()



## PIPELINE Y MODELADO

# Crear el pipeline con la regresión lineal
pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', LinearRegression())
])

# Entrenar el pipeline
results = pipeline.fit(ames_train_selected, ames_y_train)



## PREDICCIONES
y_pred = pipeline.predict(ames_x_test)

ames_test = (
  ames_x_test >>
  mutate(Sale_Price_Pred = y_pred, Sale_Price = ames_y_test)
)

ames_test.info()

(
  ames_test >>
    ggplot(aes(x = "Sale_Price_Pred", y = "Sale_Price")) +
    geom_point() +
    scale_y_continuous(labels = dollar_format(prefix='$', digits=0, big_mark=','), limits = [0, 600000] ) +
    scale_x_continuous(labels = dollar_format(prefix='$', digits=0, big_mark=','), limits = [0, 500000] ) +
    geom_abline(color = "red") +
    labs(
      title = "Comparación entre predicción y observación",
      x = "Predicción",
      y = "Observación")
)


X_train_with_intercept = sm.add_constant(transformed_df)
model = sm.OLS(ames_y_train, X_train_with_intercept).fit()

model.summary()













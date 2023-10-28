# pip install mlxtend==0.23.0
from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate

from plydata.one_table_verbs import pull
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import numpy as np
import math
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
num_cols = ["Full_Bath", "Half_Bath", "Gr_Liv_Area"]

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

def custom_function(X, col):
  X[col] = np.log1p(X[col].astype(float)) # esta función calcula el logaritmo de x+1. 
  # Evita problemas al calcular log(0)
  return X

  
custom_transformer = FunctionTransformer(
 custom_function, feature_names_out = 'one-to-one', validate=False,
 kw_args={'col': 'Gr_Liv_Area'}
 )

# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers = [
      ('log_std_transform', Pipeline([
          ('log', custom_transformer), 
          ('scaler', StandardScaler())]), ['Gr_Liv_Area']),
      ('scaler', StandardScaler(), list(set(num_cols) - set(["Gr_Liv_Area"]))),
      ('onehotencoding', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    remainder = 'passthrough',  # Mantener las columnas restantes sin cambios
    verbose_feature_names_out = True
)

transformed_data = preprocessor.fit_transform(ames_train_selected)
new_column_names = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
  transformed_data, columns=new_column_names
  )

transformed_df
transformed_df.info()


#### PIPELINE Y MODELADO

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
  select(_.Sale_Price, _.Sale_Price_Pred)
)


##### Extracción de coeficientes

X_train_with_intercept = sm.add_constant(transformed_df)
model = sm.OLS(ames_y_train, X_train_with_intercept).fit()

model.summary()


##### Métricas de desempeño

pd.options.display.float_format = '{:.2f}'.format

y_obs = ames_test["Sale_Price"]
y_pred = ames_test["Sale_Price_Pred"]

me = np.mean(y_obs - y_pred)
mae = mean_absolute_error(y_obs, y_pred)
mape = mean_absolute_percentage_error(y_obs, y_pred)
mse = mean_squared_error(y_obs, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_obs, y_pred)

n = len(y_obs)  # Número de observaciones
p = 9  # Número de predictores 
r2_adj = 1 - (n - 1) / (n - p - 1) * (1 - r2)

metrics_data = {
    "Metric": ["ME", "MAE", "MAPE", "MSE", "RMSE", "R^2", "R^2 Adj"],
    "Value": [me, mae, mape, mse, rmse, r2, r2_adj]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df

#### Gráficos de desempeño de modelo

(
  ames_test >>
    ggplot(aes(x = "Sale_Price_Pred", y = "Sale_Price")) +
    geom_point() +
    scale_y_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 600000] ) +
    scale_x_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 500000] ) +
    geom_abline(color = "red") +
    coord_equal() +
    labs(
      title = "Comparación entre predicción y observación",
      x = "Predicción",
      y = "Observación")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "error")) +
  geom_histogram(color = "white", fill = "black") +
  geom_vline(xintercept = 0, color = "red") +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  ylab("Conteos de clase") + xlab("Errores") +
  ggtitle("Distribución de error")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(sample = "error")) +
  geom_qq(alpha = 0.3) + stat_qq_line(color = "red") +
  scale_y_continuous(labels=dollar_format(big_mark=',', digits = 0)) + 
  xlab("Distribución normal") + ylab("Distribución de errores") +
  ggtitle("QQ-Plot")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "Sale_Price")) +
  geom_linerange(aes(ymin = 0, ymax = "error"), colour = "purple") +
  geom_point(aes(y = "error"), size = 0.05, alpha = 0.5) +
  geom_abline(intercept = 0, slope = 0) +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  scale_y_continuous(labels=dollar_format(big_mark=',', digits=0)) +
  xlab("Precio real") + ylab("Error de estimación") +
  ggtitle("Relación entre error y precio de venta")
)

#### Validación cruzada ####

# Definir el objeto K-Fold Cross Validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Definir las métricas de desempeño que deseas calcular como funciones de puntuación
scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}

# Realizar la validación cruzada y calcular métricas de desempeño utilizando cross_val_score
results = cross_validate(
  pipeline, 
  ames_train_selected, ames_y_train,
  cv=kf, 
  scoring=scoring
  )

# Calcular estadísticas resumidas (media y desviación estándar) de las métricas
mean_rmse = np.mean(np.sqrt(-results['test_neg_mean_squared_error']))
std_rmse = np.std(np.sqrt(-results['test_neg_mean_squared_error']))

mean_r2 = np.mean(results['test_r2'])
std_r2 = np.std(results['test_r2'])

mean_mae = np.mean(-results['test_neg_mean_absolute_error'])
std_mae = np.std(-results['test_neg_mean_absolute_error'])

mean_mape = np.mean(-results['test_mape'])
std_mape = np.std(-results['test_mape'])


# Imprimir los resultados
print(f"MAE: {mean_mae} +/- {std_mae}")
print(f"MAPE: {mean_mape} +/- {std_mape}")
print(f"R^2: {mean_r2} +/- {std_r2}")
print(f"RMSE: {mean_rmse} +/- {std_rmse}")


np.sort(results['test_r2'])




















# ----------------------------------------------#
#           LECTURA DE DATOS
# ----------------------------------------------#

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import ColumnSelector

import pandas as pd
from siuba import *
from plydata.one_table_verbs import pull

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import numpy as np
from plotnine import *


pd.set_option('display.max_columns', 4)

telco = pd.read_csv("data/Churn.csv")
telco.info()


# Segmentacion de datos 
telco_y = telco >> pull("Churn")    # telco[["Churn"]]
telco_x = select(telco, -_.Churn, -_.customerID)   # telco.drop('Churn', axis=1)

telco_x_train, telco_x_test, telco_y_train, telco_y_test = train_test_split(
 telco_x, telco_y, 
 train_size = 0.80, 
 random_state = 195,
 stratify = telco_y
 ) 

# ----------------------------------------------#
#       TRANSFORMACION DE DATOS
# ----------------------------------------------#

# Seleccionamos las variales numéricas de interés
num_cols = ["MonthlyCharges"]

# Seleccionamos las variables categóricas de interés
cat_cols = ["PaymentMethod", "Dependents"]

# Juntamos todas las variables de interés
columnas_seleccionadas = num_cols + cat_cols

pipe = ColumnSelector(columnas_seleccionadas)
telco_x_train_selected = pipe.fit_transform(telco_x_train)

telco_train_selected = pd.DataFrame(
  telco_x_train_selected, 
  columns = columnas_seleccionadas
  )

telco_train_selected.info()


# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('onehotencoding', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
)

transformed_data = preprocessor.fit_transform(telco_train_selected)
new_column_ntelco = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
  transformed_data, 
  columns=new_column_ntelco
  )
  
transformed_df
transformed_df.info()

# ----------------------------------------------#
#       CREACION Y AJUSTE DEL MODELO
# ----------------------------------------------#

# Crear el pipeline con la regresión logit
pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', LogisticRegression())
])

# Entrenar el pipeline
results = pipeline.fit(telco_train_selected, telco_y_train)

# Predicción con nuevos datos 
y_pred = pipeline.predict(telco_x_test)

telco_test = (
  telco_x_test >>
  mutate(Churn_Pred = y_pred, Churn = telco_y_test)
)

(
telco_test >>
  select(_.Churn, _.Churn_Pred)
)


# ----------------------------------------------#
#            METRICAS DE DESEMPEÑO 
# ----------------------------------------------#

matriz_confusion = confusion_matrix(telco_y_test, y_pred)
matriz_confusion


warnings.filterwarnings("ignore")

# Crear un DataFrame a partir de la matriz de confusión
confusion_df = pd.DataFrame(
  matriz_confusion, 
  columns=['Predicción Negativa', 'Predicción Positiva'], 
  index=['Real Negativa', 'Real Positiva']
  )

# Crear una figura utilizando Seaborn
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False);

plt.title('Matriz de Confusión');
plt.xlabel('Predicción');
plt.ylabel('Realidad');
plt.show();


# ----------------------------------------------#
#         ESTIMACION DE PROBABILIDADES 
# ----------------------------------------------#


y_pred = pipeline.predict_proba(telco_x_test)[:,0]
Churn_Pred = np.where(y_pred >= 0.7, "No", "Yes")

results = (
  telco_x_test >>
  mutate(
    Churn_Prob = y_pred, 
    Churn_Pred = Churn_Pred,
    Churn = telco_y_test) >>
  select(_.Churn_Prob, _.Churn_Pred, _.Churn)
)

results

(
  results
  >> group_by(_.Churn_Pred)
  >> summarize(n = _.Churn_Pred.count() )
)  


confusion_df = pd.DataFrame(
  confusion_matrix(telco_y_test, Churn_Pred), 
  columns=['Predicción Negativa', 'Predicción Positiva'], 
  index=['Real Negativa', 'Real Positiva']
  )

# Crear una figura utilizando Seaborn
plt.plot();
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False);

plt.title('Matriz de Confusión');
plt.xlabel('Predicción');
plt.ylabel('Realidad');
plt.show();


fpr, tpr, thresholds = roc_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1), 
  y_score = y_pred
  )

roc_thresholds = pd.DataFrame({
  'thresholds': thresholds, 
  'tpr': tpr, 
  'fpr': fpr}
  )

(
  roc_thresholds >>
  ggplot(aes(x = fpr, y = tpr)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray") +
  xlab("Tasa de falsos positivos") +
  ylab("Sensibilidad") +
  ggtitle("Curva ROC")
)

roc_auc_score(np.where(telco_y_test == "Yes", 0, 1), y_pred)


precision, recall, thresholds = precision_recall_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1),
  probas_pred = y_pred
  )
  
pr_thresholds = pd.DataFrame({
  'thresholds': np.append(0, thresholds), 
  'precision': precision, 
  'recall': recall}
  )

(
  pr_thresholds >>
  ggplot(aes(x = recall, y =precision)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray", intercept = 1, slope = -1) +
  xlim(0, 1) + ylim(0, 1) +
  xlab("Recall") +
  ylab("Precision") +
  ggtitle("Curva PR")
)


average_precision_score(np.where(telco_y_test == "Yes", 0, 1), y_pred)

# ----------------------------------------------#
#         VALIDACION CRUZADA 
# ----------------------------------------------#


# Definir el objeto K-Fold Cross Validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Definir las métricas de desempeño que deseas calcular como funciones de puntuación
scoring = {
  'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
  'average_precision': make_scorer(average_precision_score, greater_is_better=True)
  }


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(telco_y_train)

# Realizar la validación cruzada y calcular métricas de desempeño utilizando cross_val_score
results = cross_validate(
  pipeline, 
  telco_train_selected, 1-y,
  cv=kf, 
  scoring=scoring
  )

auc_roc_scores = results['test_roc_auc']
auc_roc_scores




auc_pr_scores = results['test_average_precision']
auc_pr_scores


# Calcular estadísticas resumidas (media y desviación estándar) de las métricas
mean_roc = np.mean(auc_roc_scores)
std_roc = np.std(auc_pr_scores)
np.sort(auc_roc_scores)

mean_pr = np.mean(auc_pr_scores)
std_pr = np.std(auc_pr_scores)
np.sort(auc_pr_scores)


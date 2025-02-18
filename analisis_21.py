# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:38:22 2025

@author: guima
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


#LEVANTO LA MUESTRA
#apoyos = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.apoyos_bbdd.csv', low_memory=False)
#servicios = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ft_servicios_documentos.csv', low_memory=False)
matricula = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_21.csv', low_memory=False)
matricula_og = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_documento.csv', low_memory=False)
notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_mat_21.csv', low_memory=False)
notas_og = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_bbdd.csv', low_memory=False)#pps = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps_documento.csv', low_memory=False)
#responsables = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.responsables_personas.csv', low_memory = False)
#dse = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dse_personas.csv', low_memory = False)
#dd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dd_personas.csv', low_memory = False)
#ds = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ds_personas.csv', low_memory = False)
#pases = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pases.csv', low_memory = False)
#localidades = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.localidades.csv', low_memory = False, delimiter = ';')
#provincias = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.provincias.csv', low_memory = False, delimiter = ';')


'''
###########################################################################
                        MODELO SOLO CON MATRICULA
###########################################################################
'''

#############################################################################
###################   PROCESAMIENTO DE MATRICULA  ###########################
#############################################################################

matricula['repite'].sum()

#pivoteo para quedarme solo con una fila x estudiante
matricula['ciclo_lectivo'] = matricula['ciclo_lectivo'].astype(str)
matricula['ciclo_prefijo'] = matricula['ciclo_lectivo'].str[-2:]

matricula_p = matricula.pivot(index=['documento', 'id_miescuela'], columns='ciclo_prefijo')
matricula_p.columns = [f"{col[1]}_{col[0]}" for col in matricula_p.columns]
matricula_p.reset_index(inplace=True)
matricula_p = matricula_p.dropna(subset=['22_ciclo_lectivo', '22_ciclo_lectivo'])
matricula_p = matricula_p.drop(columns=['21_ciclo_lectivo','22_ciclo_lectivo','23_ciclo_lectivo',
                                        #'21_Direccion','22_Direccion','23_Direccion',
                                        #'21_Direccion2','22_Direccion2','23_Direccion2',
                                        '21_coord_x','22_coord_x','23_coord_x',
                                        '21_coord_y','22_coord_y','23_coord_y'])



#elimino columnas qeu van a atener correlacion perfecta con la var Y
matricula_p = matricula_p.drop(columns=['21_anio','22_anio','23_anio'])

#matricula_p.to_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv',index=False)
#matricula_p = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv', low_memory=False)

#chequear la correlación entre las columnas
corr_matrix = matricula_p.corr().round(3)
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

'''#######################################################################'''
'''                PROCESAMIENTO PARA METER EN EL MODELO                  '''
'''#######################################################################'''

matricula_p = matricula_p[matricula_p['21_turno'].notnull()]
matricula_modelo = matricula_p.copy()

#las saco del primer modelo
col_drop = ['21_dependencia_funcional','21_modalidad','22_modalidad','23_modalidad',
            '21_calle','22_calle','23_calle','21_altura','22_altura','23_altura'
            #,'21_latitud','22_latitud','23_latitud','21_longitud','22_longitud','23_longitud'
            ]
matricula_modelo = matricula_modelo.drop(columns = col_drop)

#las paso a numericas
col_num = ['21_repite','22_repite','23_repite','21_sobreedad','22_sobreedad',
           '23_sobreedad','21_capacidad_maxima','22_capacidad_maxima',#'23_mantiene_cue',
           '23_capacidad_maxima']
matricula_modelo[col_num] = matricula_modelo[col_num].apply(pd.to_numeric, errors='coerce')

## MIDO CORRELACION ANTES DE HACER EL OHE PARA EL PRIMER MODELO
corr_matrix = matricula_modelo.corr().round(3)
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('CORR MODELO 1 - SOLO MATRICULA SIN LOCALIZACION', fontsize=20)  # Aquí se agrega el título
plt.show()

#elimino las columnas de 2023 menos 23_repite que es la y del modelo
matricula_modelo = matricula_modelo.drop(columns=[col for col in matricula_modelo.columns if col.startswith("23") and col != "23_repite"])


#armo las OHE

matricula_modelo['22_dependencia_funcional'].unique()
matricula_modelo['22_dependencia_funcional'] = matricula_modelo['22_dependencia_funcional'].str.replace('Dirección de Escuelas ', '').str.replace('Dirección de Educación ', '')
#matricula_modelo['24_dependencia_funcional'].unique()
#matricula_modelo['23_dependencia_funcional'] = matricula_modelo['24_dependencia_funcional'].str.replace('Dirección de Escuelas', '').str.replace('Dirección de Educación', '')

col_ohe = ['21_turno', '22_turno','21_jornada','22_jornada',
           '21_cueanexo','22_cueanexo','22_dependencia_funcional',
           '21_distrito_escolar','22_distrito_escolar','21_comuna','22_comuna',
           '21_barrio','22_barrio']
matricula_modelo_ohe = pd.get_dummies(matricula_modelo, columns=col_ohe, prefix=col_ohe)


####################################
###### MODELO 1 - SOLO MATRICULA
####################################

matricula_modelo_ohe["23_repite"].unique()
matricula_modelo_ohe = matricula_modelo_ohe[matricula_modelo_ohe["23_repite"].notna()]

matricula_modelo_ohe['23_repite'] = matricula_modelo_ohe['23_repite'].astype(int)
#tengo que eliminar var de 24 porque en realidad si quiero predecir 24_repite 
#es algo que tengo que poder decir antes de que ese estudiante llegue al CL 24
columnas_a_eliminar = [col for col in matricula_modelo_ohe.columns 
                       if col.startswith(("23_", "21_cueanexo", "22_cueanexo",
                                          "21_barrio", "22_barrio", "21_comuna", "22_comuna"))
                       and col != "23_repite"]
matricula_modelo_ohe = matricula_modelo_ohe.drop(columns=columnas_a_eliminar)

#elimino VD y armo los conjuntos
X = matricula_modelo_ohe.drop(columns=['documento', 'id_miescuela', '23_repite'])  # Excluir las columnas que no se usarán
y = matricula_modelo_ohe['23_repite']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# regresion logistica con xgboost
#model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',use_label_encoder=False)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]), 
    #lo que hace es penalizar la clase mayoritaria para ver si sube la precisión
    gamma=3,
    #reduce el sobreajuste e incentiva la detección de la clase minoritaria
    max_delta_step=5,  # Mejor estabilidad en datos desbalanceados
    random_state=42
)
#model.fit(X_train, y_train)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
print(model.get_params())
#evaluar el modelo
y_pred = model.predict(X_test)
residuals = y_test - y_pred  
residuals.sum() # Residuales -- 55
accuracy = accuracy_score(y_test, y_pred) #97.7%
print(f"Accuracy: {accuracy:.4f}")
#prediccion
predicciones = model.predict(X_test)
#graficos
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
#veo con que variables interactúan las top5 features // 23_capacidad_maxima
# 22_capacidad_maxima y 23_distrito_escolar_6
shap.dependence_plot("22_capacidad_maxima", shap_values.values, X_test)
shap.dependence_plot("21_capacidad_maxima", shap_values.values, X_test)
shap.dependence_plot("21_sobreedad", shap_values.values, X_test)
shap.dependence_plot("21_turno_Doble", shap_values.values, X_test)
shap.dependence_plot("21_jornada_Completa", shap_values.values, X_test)
#analisis de las predicciones
print(classification_report(y_test, y_pred))  # Para obtener precisión, recall, f1-score
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")




## PRUEBO CON UN RANDOMIZED SEARCH A VER SI SUBO EL AUC ROC DEL MODELO


param_dist = {
    'n_estimators': [100, 300, 500, 700],  # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
    'max_depth': [4, 6, 8, 10],  # Profundidad máxima de los árboles
    'colsample_bytree': [0.7, 0.8, 0.9],  # Fracción de características por árbol
    'subsample': [0.7, 0.8, 0.9],  # Fracción de muestras por árbol
    #'scale_pos_weight': [1, 2, 3, 5],  # Ajuste del peso para la clase minoritaria
    'gamma': [0, 1, 3, 5],  # Regularización para evitar sobreajuste
    'max_delta_step': [0, 1, 5],  # Paso máximo para mejorar la estabilidad
    'min_child_weight': [1, 5, 10],  # Peso mínimo de las instancias en una hoja
    }

model_rs = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Incluir scale_pos_weight aquí
    )

random_search = RandomizedSearchCV(
    estimator=model_rs,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )

#fiteo del modelo
random_search.fit(X_train, y_train)
print("Mejores parámetros:", random_search.best_params_)
print("Mejor AUC:", random_search.best_score_)
best_model = random_search.best_estimator_

# Predicción
y_pred = best_model.predict(X_test)
# Métricas
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")



## pruebo con una regresion logistica base para comparar

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(classification_report(y_test, log_reg.predict(X_test)))



###############    PROCESAMIENTO DE MATRICULA 22-24       ###################

#AHORA QUE TENGO UN MODELO CON 97.7% DE ACCURACY SOBRE LA MATRICULA DE 2021-2023
#QUIERO VER SI ESE MODELO AGUANTA TESTEANDOLO SOBRE 22-24

matricula_aux = matricula.copy()
matricula_aux['ciclo_lectivo'].unique()
orden_ciclos = {year: i+1 for i, year in enumerate(sorted(matricula_aux['ciclo_lectivo'].unique()))}
matricula_aux['ciclo_prefijo'] = matricula_aux['ciclo_lectivo'].map(orden_ciclos)
matricula_aux['identif'] = 'train'

matricula_og['ciclo_lectivo'] = matricula_og['ciclo_lectivo'].astype(str)
orden_ciclos = {year: i+1 for i, year in enumerate(sorted(matricula_og['ciclo_lectivo'].unique()))}
matricula_og['ciclo_prefijo'] = matricula_og['ciclo_lectivo'].map(orden_ciclos)
matricula_og['identif'] = 'test'

matricula_c = pd.concat([matricula_aux, matricula_og], ignore_index=True)

matricula_cp = matricula_c.pivot(index=['documento', 'id_miescuela','identif'], columns='ciclo_prefijo')
matricula_cp.columns = [f"{col[1]}_{col[0]}" for col in matricula_cp.columns]
matricula_cp.reset_index(inplace=True)
matricula_cp = matricula_cp.dropna(subset=['1_anio', '2_anio', '3_anio'])

matricula_cp['identif'].value_counts()

matricula_cp = matricula_cp.drop(columns=['1_ciclo_lectivo', '2_ciclo_lectivo','3_ciclo_lectivo',
                                           '1_coord_x','2_coord_x','3_coord_x',
                                           '1_coord_y','2_coord_y','3_coord_y'])

#elimino columnas qeu van a atener correlacion perfecta con la var Y
matricula_cp = matricula_cp.drop(columns=['1_anio','2_anio','3_anio'])

#chequear la correlación entre las columnas
corr_matrix = matricula_cp.corr().round(3)
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

'''#######################################################################'''
'''                PROCESAMIENTO PARA METER EN EL MODELO                  '''
'''#######################################################################'''

matricula_modelo_c = matricula_cp.copy()

#las saco del primer modelo
col_drop = ['1_dependencia_funcional','1_modalidad','2_modalidad','3_modalidad',
            '1_calle','2_calle','3_calle','1_altura','2_altura','3_altura'
            #,'21_latitud','22_latitud','23_latitud','21_longitud','22_longitud','23_longitud'
            ]
matricula_modelo_c = matricula_modelo_c.drop(columns = col_drop)

#las paso a numericas
col_num = ['1_repite','2_repite','3_repite','1_sobreedad','2_sobreedad',
           '3_sobreedad','1_capacidad_maxima','2_capacidad_maxima',
           '3_capacidad_maxima']
matricula_modelo_c[col_num] = matricula_modelo_c[col_num].apply(pd.to_numeric, errors='coerce')


#elimino las columnas de 2023 menos 23_repite que es la y del modelo
matricula_modelo_c = matricula_modelo_c.drop(columns=[col for col in matricula_modelo_c.columns if col.startswith("3") and col != "3_repite"])

#armo las OHE
matricula_modelo_c['2_dependencia_funcional'].unique()
matricula_modelo_c['2_dependencia_funcional'] = matricula_modelo_c['2_dependencia_funcional'].str.replace('Dirección de Escuelas ', '').str.replace('Dirección de Educación ', '')

col_ohe = ['1_turno', '2_turno','1_jornada','2_jornada',
           '1_cueanexo','2_cueanexo','2_dependencia_funcional',
           '1_distrito_escolar','2_distrito_escolar','1_comuna','2_comuna',
           '1_barrio','2_barrio']
matricula_modelo_c_ohe = pd.get_dummies(matricula_modelo_c, columns=col_ohe, prefix=col_ohe)


####################################
###### MODELO 1 - SOLO MATRICULA
####################################

matricula_modelo_c_ohe["3_repite"].unique()
matricula_modelo_c_ohe = matricula_modelo_c_ohe[matricula_modelo_c_ohe["3_repite"].notna()]

matricula_modelo_c_ohe['3_repite'] = matricula_modelo_c_ohe['3_repite'].astype(int)
#tengo que eliminar var de 24 porque en realidad si quiero predecir 24_repite 
#es algo que tengo que poder decir antes de que ese estudiante llegue al CL 24
columnas_a_eliminar = [col for col in matricula_modelo_c_ohe.columns 
                       if col.startswith(("3_", "1_cueanexo", "2_cueanexo",
                                          "1_barrio", "2_barrio", "1_comuna", "2_comuna"))
                       and col != "3_repite"]
matricula_modelo_c_ohe = matricula_modelo_c_ohe.drop(columns=columnas_a_eliminar)


###############################################################################
#                   PRUEBO EL RANDOMSEARCH SOBRE EL DF CONJUNTO
###############################################################################

train_df = matricula_modelo_c_ohe[matricula_modelo_c_ohe['identif'] == 'train']
test_df = matricula_modelo_c_ohe[matricula_modelo_c_ohe['identif'] == 'test']
# Separar features y target
X_train = train_df.drop(columns=['3_repite', 'identif','documento'])  # Eliminar la variable objetivo y 'identif'
y_train = train_df['3_repite']
X_test = test_df.drop(columns=['3_repite', 'identif','documento'])
y_test = test_df['3_repite']


param_dist = {
    'n_estimators': [100, 300, 500, 700],  # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
    'max_depth': [4, 6, 8, 10],  # Profundidad máxima de los árboles
    'colsample_bytree': [0.7, 0.8, 0.9],  # Fracción de características por árbol
    'subsample': [0.7, 0.8, 0.9],  # Fracción de muestras por árbol
    #'scale_pos_weight': [1, 2, 3, 5],  # Ajuste del peso para la clase minoritaria
    'gamma': [0, 1, 3, 5],  # Regularización para evitar sobreajuste
    'max_delta_step': [0, 1, 5],  # Paso máximo para mejorar la estabilidad
    'min_child_weight': [1, 5, 10],  # Peso mínimo de las instancias en una hoja
    }

model_rs = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Incluir scale_pos_weight aquí
    )

random_search = RandomizedSearchCV(
    estimator=model_rs,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )

#fiteo del modelo
random_search.fit(X_train, y_train)
print("Mejores parámetros:", random_search.best_params_)
print("Mejor AUC en train:", random_search.best_score_)
best_model = random_search.best_estimator_

# Predicción
y_pred = best_model.predict(X_test)
# Métricas
print(classification_report(y_test, y_pred))
print(f"AUC en test: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")


###############################################################################
#                   PRUEBO EL BASE SOBRE EL DF CONJUNTO
###############################################################################

train_df = matricula_modelo_c_ohe[matricula_modelo_c_ohe['identif'] == 'train']
test_df = matricula_modelo_c_ohe[matricula_modelo_c_ohe['identif'] == 'test']
# Separar features y target
X_train = train_df.drop(columns=['3_repite', 'identif','documento'])  # Eliminar la variable objetivo y 'identif'
y_train = train_df['3_repite']
X_test = test_df.drop(columns=['3_repite', 'identif','documento'])
y_test = test_df['3_repite']

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)
model.fit(X_train, y_train)
#evaluar el modelo
y_pred = model.predict(X_test)
residuals = y_test - y_pred  # Residuales -- 64
accuracy = accuracy_score(y_test, y_pred) #0.9978165938864629
print(f"Accuracy: {accuracy:.4f}")
#prediccion
predicciones = model.predict(X_test)
#graficos
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
#veo con que variables interactúan las top5 features // 23_capacidad_maxima
# 22_capacidad_maxima y 23_distrito_escolar_6
shap.dependence_plot("23_capacidad_maxima", shap_values.values, X_test)
shap.dependence_plot("22_capacidad_maxima", shap_values.values, X_test)
shap.dependence_plot("23_distrito_escolar_6.0", shap_values.values, X_test)
shap.dependence_plot("22_turno_Tarde", shap_values.values, X_test)
shap.dependence_plot("22_distrito_escolar_6.0", shap_values.values, X_test)
#analisis de las predicciones
print(classification_report(y_test, y_pred))  # Para obtener precisión, recall, f1-score
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")




'''
###########################################################################
                        MODELO CON MATRICULA Y NOTAS
###########################################################################
'''

#proceso notas de la mat 2021
exclude_columns = ['ciclo_lectivo', 'nivel', 'id_alumno']
# Crear un diccionario con los valores únicos de las columnas restantes
unique_values = {
    column: notas[column].unique().tolist()  # Convertir los valores únicos en una lista
    for column in notas.columns if column not in exclude_columns
}

#notas a conciliar
escala = {
    "bueno (b)": 8,"regular (r)": 7, 
    "muy bueno (mb)": 9,"sobresaliente (s)": 10,
    "promoción acompañada": 5,"insuficiente (i)": 5, 
    "suficiente": 7.5,"avanzado": 9.5,
    "en proceso": 5,"no corresponde": np.nan,
    "logrado (l)":7,"avanzado (a)":9,
    "en proceso (ep)":5,"-":np.nan # Usamos NaN en vez de "NaN"
}

# Normalizar el texto (convertir a minúsculas y quitar espacios extras)
columnas_notas = ['a_n1_mate', 'a_n2_mate', 'a_n3_mate', 'a_n4_mate', 
                  'a_n1_lengua', 'a_n2_lengua', 'a_n3_lengua', 'a_n4_lengua']
for col in columnas_notas:
    notas[col] = notas[col].astype(str).str.strip().str.lower()  # Limpieza de texto
    notas[col] = notas[col].replace(escala)  # Reemplazo según el diccionario
    notas[col] = notas[col].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)  # Conversión de números en texto

notas['rank'] = notas.groupby('id_alumno')['ciclo_lectivo'].rank(method='dense', ascending=True).astype(int)
notas_pivot = notas.pivot(index='id_alumno', columns='rank', values=[col for col in notas.columns if col not in ['id_alumno', 'rank']])

# Renombrar columnas con prefijos 1_, 2_, 3_
notas_pivot.columns = [f"{rank}_{col}" for col, rank in notas_pivot.columns]
notas_pivot.reset_index(inplace=True)

#elimino las columnas de ciclo lectivo y ademas las de las notas del segundo semestre
#la idea es dejar solo las del primer semestre para que pueda dar margen de accion
notas_pivot = notas_pivot[[col for col in notas_pivot.columns if 'ciclo_lectivo' not in col and 'nivel' not in col and not any(pattern in col for pattern in ['3_a_n3', '3_a_n4'])]]


'''
#proceso las notas para la mat 2022
'''

exclude_columns = ['ciclo_lectivo', 'nivel', 'id_alumno']

unique_values = {
    column: notas_og[column].unique().tolist()  # Convertir los valores únicos en una lista
    for column in notas_og.columns if column not in exclude_columns
}

# Filtrar por nivel
aux_p = notas_og[notas_og['nivel'] == 'Primario']
valores_unicos_p = {col: aux_p[col].unique() for col in aux_p.columns}

aux_s = notas_og[notas_og['nivel'] == 'Secundario']
valores_unicos_s = {col: aux_s[col].unique() for col in aux_s.columns}

# Diccionario de escalas para conversión de notas
escala = {
    "bueno (b)": 8, "regular (r)": 7, 
    "muy bueno (mb)": 9, "sobresaliente (s)": 10,
    "promoción acompañada": 5, "insuficiente (i)": 5, 
    "suficiente": 7.5, "avanzado": 9.5,
    "en proceso": 5, "no corresponde": np.nan,
    "-": np.nan # Usamos NaN en vez de "NaN"
}

# Columnas a procesar
columnas_notas = ['a_n1_mate', 'a_n2_mate', 'a_n3_mate', 'a_n4_mate', 
                  'a_n1_lengua', 'a_n2_lengua', 'a_n3_lengua', 'a_n4_lengua']

for col in columnas_notas:
    notas_og[col] = notas_og[col].astype(str).str.strip().str.lower()  # Limpieza de texto
    notas_og[col] = notas_og[col].replace(escala)  # Reemplazo según el diccionario
    notas_og[col] = notas_og[col].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)  # Conversión de números en texto

# Pivotear el DataFrame para tener una fila por id_alumno
notas_og['rank'] = notas_og.groupby('id_alumno')['ciclo_lectivo'].rank(method='dense', ascending=True).astype(int)
notas_pivot_og = notas_og.pivot(index='id_alumno', columns='rank', values=[col for col in notas_og.columns if col not in ['id_alumno', 'rank']])

# Renombrar columnas con prefijos 1_, 2_, 3_
notas_pivot_og.columns = [f"{rank}_{col}" for col, rank in notas_pivot_og.columns]
notas_pivot_og.reset_index(inplace=True)

# Eliminar columnas de ciclo lectivo, nivel y notas del segundo semestre
notas_pivot_og = notas_pivot_og[[col for col in notas_pivot_og.columns if 'ciclo_lectivo' not in col and 'nivel' not in col and not any(pattern in col for pattern in ['3_a_n3', '3_a_n4'])]]


##UNO TODO
notas_final = pd.concat([notas_pivot, notas_pivot_og], ignore_index=True)
columnas_ordenadas = ['id_alumno'] + sorted([col for col in notas_final.columns if col != 'id_alumno'])
notas_final = notas_final[columnas_ordenadas]
notas_final.iloc[:, 1:] = notas_final.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


#############################################################################
#############################################################################
###############       MODELO CON MATRICULA Y NOTAS         ##################
#############################################################################
#############################################################################

modelo_notas = matricula_modelo_c_ohe.merge(notas_final, left_on='id_miescuela', right_on='id_alumno', how='left')

#separo los dfs
train_notas = modelo_notas[modelo_notas['identif'] == 'train']
test_notas = modelo_notas[modelo_notas['identif'] == 'test']
# Separar features y target
X_train_notas = train_notas.drop(columns=['3_repite', 'identif','documento','id_alumno','id_miescuela'])  # Eliminar la variable objetivo y 'identif'
y_train_notas = train_notas['3_repite']
X_test_notas = test_notas.drop(columns=['3_repite', 'identif','documento','id_alumno','id_miescuela'])
y_test_notas = test_notas['3_repite']

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)
model.fit(X_train_notas, y_train_notas)
#evaluar el modelo
y_pred_notas = model.predict(X_test_notas)
residuals = y_test_notas - y_pred_notas  # Residuales -- 64
accuracy_notas = accuracy_score(y_test_notas, y_pred_notas) #0.9978165938864629
print(f"Accuracy: {accuracy_notas:.4f}")
#prediccion
predicciones = model.predict(X_test_notas)
#graficos
explainer = shap.Explainer(model, X_train_notas)
shap_values = explainer(X_test_notas)
shap.summary_plot(shap_values, X_test_notas)
shap.summary_plot(shap_values, X_test_notas, plot_type="bar")
#veo con que variables interactúan las top5 features // 23_capacidad_maxima
# 22_capacidad_maxima y 23_distrito_escolar_6
#analisis de las predicciones
print(classification_report(y_test_notas, y_pred_notas))  # Para obtener precisión, recall, f1-score
print(f"AUC: {roc_auc_score(y_test_notas, model.predict_proba(X_test_notas)[:, 1]):.4f}")



###############################################################################
#                   PRUEBO EL RANDOMSEARCH SOBRE EL DF CONJUNTO
###############################################################################

param_dist = {
    'n_estimators': [100, 300, 500, 700],  # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
    'max_depth': [4, 6, 8, 10],  # Profundidad máxima de los árboles
    'colsample_bytree': [0.7, 0.8, 0.9],  # Fracción de características por árbol
    'subsample': [0.7, 0.8, 0.9],  # Fracción de muestras por árbol
    #'scale_pos_weight': [1, 2, 3, 5],  # Ajuste del peso para la clase minoritaria
    'gamma': [0, 1, 3, 5],  # Regularización para evitar sobreajuste
    'max_delta_step': [0, 1, 5],  # Paso máximo para mejorar la estabilidad
    'min_child_weight': [1, 5, 10],  # Peso mínimo de las instancias en una hoja
    }

model_rs = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Incluir scale_pos_weight aquí
    )

random_search = RandomizedSearchCV(
    estimator=model_rs,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )

#fiteo del modelo
random_search.fit(X_train_notas, y_train_notas)
print("Mejores parámetros:", random_search.best_params_)
print("Mejor AUC en train:", random_search.best_score_)
best_model = random_search.best_estimator_

# Predicción
y_pred_notas = best_model.predict(X_test_notas)
# Métricas
print(classification_report(y_test_notas, y_pred_notas))
print(f"AUC en test: {roc_auc_score(y_test_notas, best_model.predict_proba(X_test_notas)[:, 1]):.4f}")


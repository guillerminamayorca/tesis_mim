# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:16:33 2024

@author: guima
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import ast
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import shap
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import re
from collections import defaultdict
from unidecode import unidecode



#LEVANTO LA MUESTRA
apoyos = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.apoyos_bbdd.csv', low_memory=False)
servicios = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ft_servicios_documentos.csv', low_memory=False)
matricula = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_documento.csv', low_memory=False)
#matricula = pd.read_csv('C:/Users/guillermina.mayorca/Downloads/0.matricula_documento.csv', low_memory=False)
notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_bbdd.csv', low_memory=False)
#notas = pd.read_csv('C:/Users/guillermina.mayorca/Downloads/0.notas_bbdd.csv', low_memory=False)
pps = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps_documento.csv', low_memory=False)
#pps = pd.read_csv('C:/Users/guillermina.mayorca/Downloads/0.pps_documento.csv', low_memory=False)
responsables = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.responsables_personas.csv', low_memory = False)
dse = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dse_personas.csv', low_memory = False)
dd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dd_personas.csv', low_memory = False)
ds = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ds_personas.csv', low_memory = False)
pases = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pases.csv', low_memory = False)
#pases = pd.read_csv('C:/Users/guillermina.mayorca/Downloads/0.pases.csv', low_memory = False)
#localidades = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.localidades.csv', low_memory = False, delimiter = ';')
#provincias = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.provincias.csv', low_memory = False, delimiter = ';')
secciones_notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_secciones.csv',sep=';', low_memory=False,  quotechar="'")


'''
###########################################################################
                                FUNCIONES
###########################################################################
'''

#devuelve unique-1 cols para cada conjunto de valores de la columna
def one_hot_encode_columns(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)    
    return df_encoded

#expandir columnas a partir de un ajson
def expandir_columna_json(df, col_json):
    valid_json = df[col_json].dropna().apply(lambda x: x if isinstance(x, str) and x.strip() else '{}')
    json_data = []
    for item in valid_json:
        try:
            json_data.append(json.loads(item))
        except json.JSONDecodeError:
            json_data.append({})
    json_df = pd.json_normalize(json_data)
    suffix = f"{col_json}_" 
    json_df.columns = [f"{suffix}{col}" for col in json_df.columns]
    return df.join(json_df).drop(columns=[col_json])

'''
###########################################################################
                    PROCESAMIENTO DE LAS DISTINTAS BASES
###########################################################################
'''


#############################################################################
###################   PROCESAMIENTO DE MATRICULA  ###########################
#############################################################################

matricula['repite'].sum()

matricula['altura'] = matricula['altura'].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() != '' else '')
matricula['Direccion'] = matricula['calle'].fillna('') + ' ' + matricula['altura'] + ', ' + matricula['barrio'].fillna('') + ', Ciudad de Buenos Aires, Argentina'
matricula['Direccion2'] = matricula['calle'].fillna('') + ' ' + matricula['altura'] + ', Ciudad de Buenos Aires , Argentina'


#pivoteo para quedarme solo con una fila x estudiante
matricula['ciclo_lectivo'] = matricula['ciclo_lectivo'].astype(str)
matricula['ciclo_prefijo'] = matricula['ciclo_lectivo'].str[-2:]

matricula_p = matricula.pivot(index=['documento', 'id_miescuela'], columns='ciclo_prefijo')
matricula_p.columns = [f"{col[1]}_{col[0]}" for col in matricula_p.columns]
matricula_p.reset_index(inplace=True)
matricula_p = matricula_p.dropna(subset=['23_ciclo_lectivo', '24_ciclo_lectivo'])
matricula_p = matricula_p.drop(columns=['23_ciclo_lectivo','22_ciclo_lectivo','24_ciclo_lectivo',
                                        '22_Direccion','23_Direccion','24_Direccion',
                                        '22_Direccion2','23_Direccion2','24_Direccion2',
                                        '22_coord_x','23_coord_x','24_coord_x',
                                        '22_coord_y','23_coord_y','24_coord_y'])

#correccion de dos casos
matricula_p = matricula_p[matricula_p['documento'] != '49260247']
matricula_p.loc[matricula_p['documento'] == 96138637, '23_repite'] = 0

#elimino columnas qeu van a atener correlacion perfecta con la var Y
matricula_p = matricula_p.drop(columns=['23_anio','22_anio','24_anio'])


#matricula_p.to_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv',index=False)
#matricula_p = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv', low_memory=False)

#chequear la correlación entre las columnas
corr_matrix = matricula_p.corr().round(3)
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


matricula_p = matricula_p[matricula_p['22_turno'].notnull()]
matricula_modelo = matricula_p.copy()

#las saco del primer modelo
col_drop = ['22_dependencia_funcional','22_modalidad','23_modalidad','24_modalidad',
            '22_calle','23_calle','24_calle','22_altura','23_altura','24_altura'
            #,'22_latitud','23_latitud','24_latitud','22_longitud','23_longitud','24_longitud'
            ]
matricula_modelo = matricula_modelo.drop(columns = col_drop)

#las paso a numericas
col_num = ['22_repite','23_repite','24_repite','22_sobreedad','23_sobreedad',
           '24_sobreedad','22_capacidad_maxima','23_capacidad_maxima',#'24_mantiene_cue'
           '24_capacidad_maxima']
matricula_modelo[col_num] = matricula_modelo[col_num].apply(pd.to_numeric, errors='coerce')

## MIDO CORRELACION ANTES DE HACER EL OHE PARA EL PRIMER MODELO
corr_matrix = matricula_modelo.corr().round(3)
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('CORR MODELO 1 - SOLO MATRICULA SIN LOCALIZACION', fontsize=20)  # Aquí se agrega el título
plt.show()

#armo las OHE

matricula_modelo['23_dependencia_funcional'].unique()
matricula_modelo['24_dependencia_funcional'].unique()
matricula_modelo['23_dependencia_funcional'] = matricula_modelo['23_dependencia_funcional'].str.replace('Dirección de Escuelas ', '').str.replace('Dirección de Educación ', '')
matricula_modelo['24_dependencia_funcional'] = matricula_modelo['24_dependencia_funcional'].str.replace('Dirección de Escuelas ', '').str.replace('Dirección de Educación ', '')

col_ohe = ['22_turno', '23_turno', '24_turno','22_jornada','23_jornada','24_jornada',
           '22_cueanexo','23_cueanexo','24_cueanexo','23_dependencia_funcional','24_dependencia_funcional',
           '22_distrito_escolar','23_distrito_escolar','24_distrito_escolar','22_comuna','23_comuna',
           '24_comuna','22_barrio','23_barrio','24_barrio']
matricula_modelo_ohe = pd.get_dummies(matricula_modelo, columns=col_ohe, prefix=col_ohe)


print(f"Valor máximo: {matricula_modelo_ohe.drop(columns=['documento','id_miescuela']).max().max()}")
print(f"Dónde está el valor máximo: {matricula_modelo_ohe.drop(columns=['documento','id_miescuela']).max().idxmax()}")
print(f"Valor mínimo: {matricula_modelo_ohe.drop(columns=['documento','id_miescuela']).min().min()}")
print(f"Dónde está el valor mínimo: {matricula_modelo_ohe.drop(columns=['documento','id_miescuela']).min().idxmax()}")


####################################
###### MODELO 1 - SOLO MATRICULA
####################################

matricula_modelo_ohe['24_repite'] = matricula_modelo_ohe['24_repite'].astype(int)
#tengo que eliminar var de 24 porque en realidad si quiero predecir 24_repite 
#es algo que tengo que poder decir antes de que ese estudiante llegue al CL 24
columnas_a_eliminar = [col for col in matricula_modelo_ohe.columns 
                       if col.startswith(("24_", "22_cueanexo", "23_cueanexo",
                                          "22_barrio", "23_barrio", "22_comuna", "23_comuna"))
                       and col != "24_repite"]
matricula_modelo_ohe = matricula_modelo_ohe.drop(columns=columnas_a_eliminar)

#elimino VD y armo los conjuntos
X = matricula_modelo_ohe.drop(columns=['documento', 'id_miescuela', '24_repite'])  # Excluir las columnas que no se usarán
y = matricula_modelo_ohe['24_repite']
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
    random_state=42
)
model.fit(X_train, y_train)
#evaluar el modelo
y_pred = model.predict(X_test)
residuals = y_test - y_pred  # Residuales -- 64
accuracy = accuracy_score(y_test, y_pred) #0.9978165938864629
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.4f}")
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


#matriz de valores
y_test.sum()
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")




##### MISMO PERO CON UN RANDOMSEARCH

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
print(f"AUC ROC sobre train: {roc_auc_score(y_train, random_search.predict_proba(X_train)[:, 1]):.4f}")
best_model = random_search.best_estimator_

# Predicción
y_pred = best_model.predict(X_test)

explainer_1rs = shap.Explainer(best_model, X_train)
shap_values_1rs = explainer_1rs(X_test)
shap.summary_plot(shap_values_1rs, X_test)
shap.summary_plot(shap_values_1rs, X_test, plot_type="bar")
#veo con que variables interactúan las top5 features // 23_capacidad_maxima
# 22_capacidad_maxima y 23_distrito_escolar_6
shap.dependence_plot("23_capacidad_maxima", shap_values_1rs.values, X_test)
shap.dependence_plot("22_capacidad_maxima", shap_values_1rs.values, X_test)
shap.dependence_plot("23_distrito_escolar_6.0", shap_values_1rs.values, X_test)
shap.dependence_plot("22_turno_Tarde", shap_values_1rs.values, X_test)
shap.dependence_plot("22_distrito_escolar_6.0", shap_values_1rs.values, X_test)
# Métricas
print(classification_report(y_test, y_pred))
print(f"AUC ROC en test: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test, y_pred):.6f}')


#matriz de valores
y_test.sum()
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")




#############################################################################
#####################    PROCESAMIENTO DE PPS       ##########################
#############################################################################

columnas_json = ['actitud','convivencia', 'trayectoria','vinculo','antecedentes','intervenciones','jornada']

for col in columnas_json:
    if col == 'actitud':
        json_data = pd.json_normalize(pps['actitud'].dropna().apply(json.loads))
        json_data.columns = [f"actitud_{col}" for col in json_data.columns]
        pps = pps.join(json_data).drop(columns=['actitud'])
    else:
        pps = expandir_columna_json(pps, col)

        
col_texto = ['actitud_observaciones','convivencia_observaciones','trayectoria_destaca','trayectoria_interes',
             'trayectoria_contenidos','trayectoria_ajustesAreas','trayectoria_cualesAjustes',
             'vinculo_observaciones','antecedentes_antecedentes','antecedentes_poseeCertificado','antecedentes_informe.url',
             'antecedentes_informe.filename','intervenciones_informe.url','intervenciones_informe.filename',
             'actitud_como','actitud_pedagogica','trayectoria_ajustesRazonables','jornada_cual','jornada_observaciones',
             'trayectoria_cuales','trayectoria_observaciones','intervenciones_derivacion','convivencia_resuelve',
             'convivencia_vinculapares','actitud_trabaja','actitud_autonomo','actitud_participa',
             'trayectoria_requirioadecuaciones']

pps = pps.drop(columns=col_texto)

## Chequeo cuantos nulos tengo en als columnas 
nulos_por_columna = pps.isnull().sum()
plt.figure(figsize=(15, 5))
nulos_por_columna.plot(kind="bar", color="red", alpha=0.7)
plt.xlabel("Columna")
plt.ylabel("Cantidad de Nulos")
plt.title("Cantidad de valores nulos por columna - PPS")
plt.xticks(rotation=90)  # Rotar los nombres de las columnas para mejor visibilidad
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()



valores_unicos_dict = {col: pps[col].unique().tolist() for col in pps.columns}

#veo que hay cols con '' o nulos
conteo_nulos = pps.isna().sum()  # Cuenta NaNs
conteo_vacios = (pps == '').sum()  # Cuenta valores vacíos
conteo_total = conteo_nulos + conteo_vacios  # Suma ambos conteos


#adaptacion a valores aptos
cols_a_convertir = ['trayectoria_requirio', 'jornada_participa', 'trayectoria_requirioacompañada','trayectoria_requiriopedagogico'] 
pps[cols_a_convertir] = pps[cols_a_convertir].replace({'Si': 1, 'Sí': 1, 'No': 0, '': np.nan, None: np.nan, np.nan: np.nan})

cols_frec = ['actitud_demuestra', 'actitud_logra', 'actitud_consulta','actitud_cumple',
             'actitud_manifiesta','actitud_puedeOrganizarse','convivencia_acude','convivencia_mantiene',
             'convivencia_respeta','convivencia_vincula','vinculo_acompaña','vinculo_participa'] 
pps[cols_frec] = pps[cols_frec].replace({'Con poca frecuencia': 0, 'Frecuentemente': 1, 'Siempre':2, '': np.nan, None: np.nan, np.nan: np.nan})

valores_unicos_check = {col: pps[col].unique().tolist() for col in pps.columns}


# Definir los prefijos y agrupar las columnas por estos
prefijos = ["actitud", "convivencia", "vinculo", 'trayectoria']
grupos = defaultdict(list)

# Agrupar columnas según prefijos
for col in pps.columns:
    for prefijo in prefijos:
        if col.startswith(prefijo):
            grupos[prefijo].append(col)

#la saco porque la convierto por separado, tiene valores categóricos
if 'vinculo_adulto' in grupos['vinculo']:
    grupos['vinculo'].remove('vinculo_adulto')
if 'trayectoria_interrumpida' in grupos['trayectoria']:
    grupos['trayectoria'].remove('trayectoria_interrumpida')


# Mostrar los grupos creados
for prefijo, variables in grupos.items():
    print(f"{prefijo}: {variables}")

for index, row in pps.iterrows():
    # Limpiar 'vinculo_adulto' (eliminar caracteres especiales, espacios, etc.)
    cleaned_vinculo_adulto = re.sub(r'[^A-Za-z0-9]+', '', str(row['vinculo_adulto'])).strip()
    
    # Si 'vinculo_adulto' queda vacío después de limpiar, establecer las demás columnas de 'vinculo' a 0
    if not cleaned_vinculo_adulto:
        for var in grupos['vinculo']:
            if var != 'vinculo_adulto':  # Excluir 'vinculo_adulto' de la asignación
                pps.at[index, var] = 0

# Imputar los valores faltantes con el promedio de las otras variables del grupo
for prefijo, variables in grupos.items():
    for index, row in pps.iterrows():
        # Identificar las columnas con valores no nulos
        non_null_values = row[variables].dropna()
        
        if len(non_null_values) > 0:
            # Calcular el promedio de los valores no nulos
            mean_value = non_null_values.mean()
            
            # Imputar los valores faltantes con el promedio calculado
            for var in variables:
                if pd.isna(row[var]):
                    pps.at[index, var] = mean_value

# Redondear los valores en las columnas de los grupos de los prefijos y reemplazar los nulos con 0
for prefijo, variables in grupos.items():
    # Redondear valores a 2 decimales en las columnas correspondientes
    pps[variables] = pps[variables].round(0)
    
    # Reemplazar valores nulos con 0
    pps[variables] = pps[variables].fillna(0)

#texto libre ver
'trayectoria_interrumpida'
'vinculo_adulto'

tray_int_uni = (pps['trayectoria_interrumpida'].dropna().str.lower().str.replace(r'[^a-záéíóúüñ ]', '', regex=True).str.strip().unique())
vinculo_adulto_uni = (pps['vinculo_adulto'].dropna().str.lower().str.replace(r'[^a-záéíóúüñ ]', '', regex=True).str.strip().unique())

#MAPEO TENTATIVO A PARTIR DE LOS VALORES DE VINCULO -- MAS FACIL OHE CREO
map_vinculo = {
    'vinculo_adulto_nadie': ['nadie', 'docentes no conocen a los padres'],
    'vinculo_adulto_madre': ['mamá', 'mama', 'madre', 'masdre', 'made', 'progenitora', 'progenitores', 'ambos', 'padres'],
    'vinculo_adulto_padre': ['papá', 'papa', 'padre', 'progenitor', 'progenitores', 'ambos', 'padres'],
    'vinculo_adulto_hermana': ['hermana', 'hermanas', 'hermanos'],
    'vinculo_adulto_hermano': ['hermano', 'hermanos'],
    'vinculo_adulto_cuniados': ['cuñada', 'cuñado', 'cuñados', 'cuñadas'],
    'vinculo_adulto_abuela': ['abuela', 'abuelas', 'abuelos'],
    'vinculo_adulto_abuelo': ['abuelo', 'abuelos'],
    'vinculo_adulto_tios': ['tía', 'tia', 'tío', 'tíos', 'tías'],
    'vinculo_adulto_padrino_madrina': ['padrinos', 'madrina', 'padrino'],
    'vinculo_adulto_pareja_progenitores': ['madrastra', 'padrastro', 'pareja de la madre', 'pareja del padre', 'mujer del padre', 'esposo de la madre'],
    'vinculo_adulto_docentes': ['docentes', 'maestra', 'maestro', 'maestra integradora', 'maestro integrador'],
    'vinculo_adulto_hogar': ['operador del hogar', 'operadora del hogar', 'operadores del hogar',
                              'cat n°', 'director del hogar', 'directora del hogar', 'hogar',
                              'equipo técnico del hogar', 'referentes del hogar'],
    'vinculo_adulto_tutores': ['tutora legal', 'tutora', 'tutor legal', 'tutor', 'tutores', 'tutoras',
                               'representante legal', 'representante']
}

# Asegurar que la columna no tenga NaN y convertir a minúsculas
pps['vinculo_adulto'] = pps['vinculo_adulto'].fillna('').str.lower()

# Crear las columnas binarias
for col, keywords in map_vinculo.items():
    pps[col] = pps['vinculo_adulto'].apply(lambda x: 1 if any(kw in x for kw in keywords) else 0)



### chequeo nulos
null_summary = pd.DataFrame({
    'Null Count':  pps.isna().sum(),
    'Null Percentage': ( pps.isna().sum() / len(pps)) * 100

})

pps = pps.drop(columns=['jornada_participa'])

####################################
###### MODELO 2 - MATRICULA Y PPS
####################################

matricula_m2 = matricula_modelo_ohe.copy()

#le agrego la info de pps sacando la col de trayectoria_requirio xq no la proc
#y la de vinculo_adulto porque le hice OHE
matricula_m2 = matricula_m2.merge(pps, on="documento", how="inner")
matricula_m2 = matricula_m2.drop(columns=['ciclo_lectivo','id_miesucela','trayectoria_interrumpida',
                                          'vinculo_adulto','23_sobreedad'])


#elimino VD y armo los conjuntos
X_m2 = matricula_m2.drop(columns=['documento', 'id_miescuela', '24_repite'])  # Excluir las columnas que no se usarán
y_m2 = matricula_m2['24_repite']
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(X_m2, y_m2, test_size=0.2, random_state=42)


#modelo
model_m2 = xgb.XGBClassifier(
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
model_m2.fit(X_train_m2, y_train_m2)
y_pred_m2 = model_m2.predict(X_test_m2)
accuracy_m2 = accuracy_score(y_test_m2, y_pred_m2)
print(f"AUC sobre el train: {accuracy_m2:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m2, model_m2.predict_proba(X_train_m2)[:, 1]):.4f}")
#prediccion
predicciones_m2 = model_m2.predict(X_test_m2)
#var imp
importance_m2 = model_m2.get_booster().get_score(importance_type='weight')
sorted_importance_m2 = sorted(importance_m2.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m2, y_pred_m2))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m2, model_m2.predict_proba(X_test_m2)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m2, y_pred_m2):.6f}')


#matriz de valores
y_test_m2.sum()
tn, fp, fn, tp = confusion_matrix(y_test_m2, y_pred_m2).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")

#graficos
explainer_m2 = shap.Explainer(model_m2, X_train_m2)
expected_value_m2 = explainer_m2.expected_value
prob_base_m2 = 1 / (1 + np.exp(-expected_value_m2))
print('Valor de prediccion de base: ', prob_base_m2)
shap_values_m2 = explainer_m2(X_test_m2)
shap.summary_plot(shap_values_m2, X_test_m2)
shap.summary_plot(shap_values_m2, X_test_m2, plot_type="bar")

#quiero ver con que variables interactúan las de actitud que parecen ser las más
#significativas y si es que cambia la interaccion de las top5 de antes
shap.dependence_plot("actitud_consulta", shap_values_m2.values, X_test_m2)
shap.dependence_plot("actitud_manifiesta", shap_values_m2.values, X_test_m2)
shap.dependence_plot("23_capacidad_maxima", shap_values_m2.values, X_test_m2)
shap.dependence_plot("22_capacidad_maxima", shap_values_m2.values, X_test_m2)



##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m2[y_train_m2 == 0]) / len(y_train_m2[y_train_m2 == 1])  # Incluir scale_pos_weight aquí
    )

random_search2 = RandomizedSearchCV(
    estimator=model_rs2,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search2.fit(X_train_m2, y_train_m2)
print("Mejores parámetros:", random_search2.best_params_)
print("Mejor AUC del train:", random_search2.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m2, random_search2.predict_proba(X_train_m2)[:, 1]):.4f}")
best_model_2rs = random_search2.best_estimator_

# Predicción
y_pred_m2 = best_model_2rs.predict(X_test_m2)

# Métricas
print(classification_report(y_test_m2, y_pred_m2))
print(f"AUC: {roc_auc_score(y_test_m2, best_model_2rs.predict_proba(X_test_m2)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m2, y_pred_m2):.6f}')


#matriz de valores
print(y_test_m2.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m2, y_pred_m2).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")

#analisis grafico
explainer_2rs = shap.Explainer(best_model_2rs, X_train_m2)
shap_values_2rs = explainer_2rs(X_test_m2)
shap.summary_plot(shap_values_2rs, X_test_m2)
shap.summary_plot(shap_values_2rs, X_test_m2, plot_type="bar")


#############################################################################
###################   PROCESAMIENTO DE PASES  ###########################
#############################################################################

pases_p = pases[pases["descripcion_solicitud_pase"] == "Aprobado"][['anio_pase', 'documento', 'cue_destino', 'fecha_de_pase']].copy()
# Agregar la columna 'pase' con valor 1
pases_p['pase'] = 1
#m quedo solo con los pases que se dan antes de la mitad del primer año
pases_p['fecha_de_pase'] = pd.to_datetime(pases_p['fecha_de_pase'])
fecha_limite = pd.to_datetime('2023-07-15')
pases_p = pases_p[pases_p['fecha_de_pase'] <= fecha_limite]

#conteo pases
pases_p = pases_p.groupby(['documento', 'anio_pase'])['cue_destino'].nunique().reset_index()
pases_p.rename(columns={'cue_destino': 'cant_pases'}, inplace=True)

pases_p['anio_pase'] = pases_p['anio_pase'].astype(str)
pases_p['anio_prefijo'] = pases_p['anio_pase'].str[-2:]
pases_p = pases_p.pivot(index='documento', columns='anio_prefijo')

pases_p.columns = [f"{col[1]}_{col[0]}" for col in pases_p.columns]
pases_p.reset_index(inplace=True)
pases_p = pases_p.drop(columns=['22_anio_pase','23_anio_pase'])
pases_p[['22_cant_pases', '23_cant_pases']] = pases_p[['22_cant_pases', '23_cant_pases']].fillna(0)


####################################
###### MODELO 3 - MATRICULA, PPS y PASES
####################################

matricula_m3 = matricula_m2.copy()
matricula_m3 = matricula_m3.merge(pases_p, on="documento", how="left")

# Rellenar las columnas específicas con 0 cuando haya valores nulos
matricula_m3['22_cant_pases'].fillna(0, inplace=True)
matricula_m3['23_cant_pases'].fillna(0, inplace=True)


#elimino VD y armo los conjuntos
X_m3 = matricula_m3.drop(columns=['documento', 'id_miescuela', '24_repite'])  # Excluir las columnas que no se usarán
y_m3 = matricula_m3['24_repite']
X_train_m3, X_test_m3, y_train_m3, y_test_m3 = train_test_split(X_m3, y_m3, test_size=0.2, random_state=42)

#modelo
model_m3 = xgb.XGBClassifier(
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
model_m3.fit(X_train_m3, y_train_m3)
y_pred_m3 = model_m3.predict(X_test_m3)
accuracy_m3 = accuracy_score(y_test_m3, y_pred_m3)
print(f"Accuracy sobre el train: {accuracy_m3:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m3, model_m3.predict_proba(X_train_m3)[:, 1]):.4f}")
#prediccion
predicciones_m3 = model_m3.predict(X_test_m3)
#var imp
importance_m3 = model_m3.get_booster().get_score(importance_type='weight')
sorted_importance_m3 = sorted(importance_m3.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m3, y_pred_m3))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m3, model_m3.predict_proba(X_test_m3)[:, 1]):.4f}")

#graficos
explainer_m3 = shap.Explainer(model_m3, X_train_m3)
expected_value_m3 = explainer_m3.expected_value
prob_base_m3 = 1 / (1 + np.exp(-expected_value_m3))
print('Valor de prediccion de base: ', prob_base_m3)
shap_values_m3 = explainer_m3(X_test_m3)
shap.summary_plot(shap_values_m3, X_test_m3)
shap.summary_plot(shap_values_m3, X_test_m3, plot_type="bar")


print(y_test_m3.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m3, y_pred_m3).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")




##### MISMO PERO CON UN RANDOMSEARCH

param_dist = {
    'n_estimators': [100, 300, 500, 700],  # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasa de aprendizaje
    'max_depth': [4, 6, 8, 10],  # Profundidad máxima de los árboles
    'colsample_bytree': [0.7, 0.8, 0.9],  # Fracción de características por árbol
    'subsample': [0.7, 0.8, 0.9],  # Fracción de muestras por árbol
    'gamma': [0, 1, 3, 5],  # Regularización para evitar sobreajuste
    'max_delta_step': [0, 1, 5],  # Paso máximo para mejorar la estabilidad
    'min_child_weight': [1, 5, 10],  # Peso mínimo de las instancias en una hoja
    }

model_rs3 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m3[y_train_m3 == 0]) / len(y_train_m3[y_train_m3 == 1])  # Incluir scale_pos_weight aquí
    )

random_search3 = RandomizedSearchCV(
    estimator=model_rs3,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search3.fit(X_train_m3, y_train_m3)
print("Mejores parámetros:", random_search3.best_params_)
print("AUC ROC Cruzado CV=3:", random_search3.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m3, random_search3.predict_proba(X_train_m3)[:, 1]):.4f}")
best_model_3rs = random_search3.best_estimator_

# Predicción
y_pred_m3 = best_model_3rs.predict(X_test_m3)

# Métricas
print(classification_report(y_test_m3, y_pred_m3))
print(f"ROC AUC TEST: {roc_auc_score(y_test_m3, best_model_3rs.predict_proba(X_test_m3)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m3, y_pred_m3):.6f}')

#matriz de valores
print(y_test_m3.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m3, y_pred_m3).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")

#analisis grafico
explainer_3rs = shap.Explainer(best_model_3rs, X_train_m3)
shap_values_3rs = explainer_3rs(X_test_m3)
shap.summary_plot(shap_values_3rs, X_test_m3)
shap.summary_plot(shap_values_3rs, X_test_m3, plot_type="bar")


#############################################################################
###################   PROCESAMIENTO DE NOTAS  ###########################
#############################################################################

exclude_columns = ['ciclo_lectivo', 'nivel', 'id_alumno']

# Crear un diccionario con los valores únicos de las columnas restantes
unique_values = {
    column: notas[column].unique().tolist()  # Convertir los valores únicos en una lista
    for column in notas.columns if column not in exclude_columns
}


aux_p = notas[notas['nivel'] == 'Primario']
valores_unicos_p = {col: aux_p[col].unique() for col in aux_p.columns}

aux_s = notas[notas['nivel'] == 'Secundario']
valores_unicos_s = {col: aux_s[col].unique() for col in aux_s.columns}


#notas a conciliar
escala = {
    "bueno (b)": 8,"regular (r)": 7, 
    "muy bueno (mb)": 9,"sobresaliente (s)": 10,
    "promoción acompañada": 5,"insuficiente (i)": 5, 
    "suficiente": 7.5,"avanzado": 9.5,
    "en proceso": 5,"no corresponde": np.nan,
    "-":np.nan # Usamos NaN en vez de "NaN"
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
notas_pivot = notas_pivot.replace(['nan', ''], np.nan)
notas_pivot.iloc[:, 1:] = notas_pivot.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').astype(float)
notas_pivot.dtypes


## me quedo con las unicas columnas que debería tener el DF al momento de predecir
## mitad de año del 2023
sorted_columns = ['id_alumno',
    '1_a_n1_mate', '1_a_n2_mate', '1_a_n3_mate', '1_a_n4_mate',
    '1_a_n1_lengua', '1_a_n2_lengua', '1_a_n3_lengua', '1_a_n4_lengua',
    '2_a_n1_mate', '2_a_n2_mate', '2_a_n1_lengua', '2_a_n2_lengua'
]

notas_pivot = notas_pivot[sorted_columns]


####################################
###### MODELO 4 - MATRICULA, PPS, PASES y NOTAS
####################################


#tengo notas para todos los etudiantes del df de PPS
matricula_m3['id_miescuela'].nunique()
notas_pivot['id_alumno'].nunique()
len(set(matricula_m3['id_miescuela']) & set(notas_pivot['id_alumno']))

#hago el merge con el df del modelo anterior
matricula_m4 = matricula_m3.copy()
matricula_m4 = matricula_m4.merge(notas_pivot, left_on="id_miescuela", right_on="id_alumno", how="left")

#elimino VD y armo los conjuntos
X_m4 = matricula_m4.drop(columns=['documento', 'id_miescuela', '24_repite','id_alumno'])  # Excluir las columnas que no se usarán
y_m4 = matricula_m4['24_repite']
X_train_m4, X_test_m4, y_train_m4, y_test_m4 = train_test_split(X_m4, y_m4, test_size=0.2, random_state=42)

#modelo
model_m4 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)

model_m4.fit(X_train_m4, y_train_m4)
y_pred_m4 = model_m4.predict(X_test_m4)
accuracy_m4 = accuracy_score(y_test_m4, y_pred_m4)
print(f"Accuracy sobre el train: {accuracy_m4:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m4, model_m4.predict_proba(X_train_m4)[:, 1]):.4f}")
#prediccion
predicciones_m4 = model_m4.predict(X_test_m4)
#var imp
importance_m4 = model_m4.get_booster().get_score(importance_type='weight')
sorted_importance_m4 = sorted(importance_m4.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m4, y_pred_m4))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m4, model_m4.predict_proba(X_test_m4)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m4, y_pred_m4):.6f}')

#matriz de valores
print(y_test_m4.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m4, y_pred_m4).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#graficos
explainer_m4 = shap.Explainer(model_m4, X_train_m4)
expected_value_m4 = explainer_m4.expected_value
prob_base_m4 = 1 / (1 + np.exp(-expected_value_m4))
print('Valor de prediccion de base: ', prob_base_m4)
shap_values_m4 = explainer_m4(X_test_m4)
shap.summary_plot(shap_values_m4, X_test_m4)
shap.summary_plot(shap_values_m4, X_test_m4, plot_type="bar")



##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs4 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m4[y_train_m4 == 0]) / len(y_train_m4[y_train_m4 == 1])  # Incluir scale_pos_weight aquí
    )

random_search4 = RandomizedSearchCV(
    estimator=model_rs4,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search4.fit(X_train_m4, y_train_m4)
print("Mejores parámetros:", random_search4.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search4.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m4, random_search4.predict_proba(X_train_m4)[:, 1]):.4f}")
best_model_4rs = random_search4.best_estimator_

# Predicción
y_pred_m4 = best_model_4rs.predict(X_test_m4)

# Métricas
print(classification_report(y_test_m4, y_pred_m4))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m4, best_model_4rs.predict_proba(X_test_m4)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m4, y_pred_m4):.6f}')

#matriz de valores
print(y_test_m4.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m4, y_pred_m4).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m4_rs = shap.Explainer(best_model_4rs, X_train_m4)
expected_value_m4 = explainer_m4.expected_value
prob_base_m4 = 1 / (1 + np.exp(-expected_value_m4))
print('Valor de prediccion de base: ', prob_base_m4)
shap_values_m4 = explainer_m4(X_test_m4)
shap.summary_plot(shap_values_m4, X_test_m4)
shap.summary_plot(shap_values_m4, X_test_m4, plot_type="bar")




#############################################################################
################   PROCESAMIENTO DE NOTAS RELATIVAS  ########################
#############################################################################

secciones_notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_secciones.csv',sep=';', low_memory=False,  quotechar="'")

#########
# TENGO QUE IMPUTAR LOS NULOS Y ARMAR LA COLUMNA DE RELATIVOS
#########

# Crear un DataFrame resumen
resumen_nulos = pd.DataFrame({'Cantidad de Nulos': secciones_notas.isnull().sum(), 
                              'Porcentaje de Nulos (%)': (secciones_notas.isnull().sum() / len(secciones_notas)) * 100})

#paso las notas a numericas con la misma escala que antes

columnas_notas = ['a_n1_mate', 'a_n2_mate', 'a_n3_mate', 'a_n4_mate', 
                  'a_n1_lengua', 'a_n2_lengua', 'a_n3_lengua', 'a_n4_lengua']

for col in columnas_notas:
    secciones_notas[col] = secciones_notas[col].astype(str).str.strip().str.lower()  # Limpieza de texto
    secciones_notas[col] = secciones_notas[col].replace(escala)  # Reemplazo según el diccionario
    secciones_notas[col] = pd.to_numeric(secciones_notas[col], errors='coerce')
    

# tengo 3 secciones sin notas en ningun momento, necesito imputarles el valor medio de
# cada instanica de notas de su DE, porque no cuento con otra seccion de esa misma escuela

#ciclo lectivo en matricula es object, la casteo
secciones_notas["ciclo_lectivo"] = secciones_notas["ciclo_lectivo"].astype(str)
secciones_notas = secciones_notas.merge(
    matricula[["id_miescuela", "ciclo_lectivo", "distrito_escolar"]],
    left_on=["id_alumno", "ciclo_lectivo"], 
    right_on=["id_miescuela", "ciclo_lectivo"], 
    how="left"
)


# tengo chicos que no están en la matricula de PPS pero si en los cursos a los que
# los estudiantes de PPS van despues, les asigno el de segun la seccion

mapeo_distrito = secciones_notas.groupby("id_seccion_miescuela")["distrito_escolar"].first()
secciones_notas["distrito_escolar"] = secciones_notas["distrito_escolar"].fillna(
    secciones_notas["id_seccion_miescuela"].map(mapeo_distrito)
)


columnas_notas = ["a_n1_mate", "a_n2_mate", "a_n3_mate", "a_n4_mate",
                  "a_n1_lengua", "a_n2_lengua", "a_n3_lengua", "a_n4_lengua"]


# le saco el promedio a los que tienen notas subidas 

secciones_notas["promedio_lengua"] = secciones_notas.apply(
    lambda row: row[["a_n1_lengua", "a_n2_lengua", "a_n3_lengua", "a_n4_lengua"]].mean(skipna=True) 
    if row["ciclo_lectivo"] == "2022" 
    else row[["a_n1_lengua", "a_n2_lengua"]].mean(skipna=True), 
    axis=1
)

secciones_notas["promedio_mate"] = secciones_notas.apply(
    lambda row: row[["a_n1_mate", "a_n2_mate", "a_n3_mate", "a_n4_mate"]].mean(skipna=True) 
    if row["ciclo_lectivo"] == "2022" 
    else row[["a_n1_mate", "a_n2_mate"]].mean(skipna=True), 
    axis=1
)

# voy a imputarle la media de la seccion en caso de que al menos un 25% de la misma
# tenga calificaciones, en caso contrario imputo la media del distrito para evitar sesgos
# que pueden existir si los calificados son muy buenos o muy malos alumnos 

def imputar_si_suficientes(x):
    num_estudiantes = len(x)
    num_calificados = x.notna().sum()
    porcentaje_calificados = num_calificados / num_estudiantes

    # Si al menos el 25% tienen nota, completar con la media
    if porcentaje_calificados >= 0.25:
        return x.fillna(x.mean())
    else:
        return x  # Dejar los valores NaN si no se cumple la condición

# Aplicar la función a cada columna de notas
secciones_notas["promedio_lengua"] = secciones_notas.groupby(["ciclo_lectivo", "id_seccion_miescuela"])["promedio_lengua"].transform(imputar_si_suficientes)
secciones_notas["promedio_mate"] = secciones_notas.groupby(["ciclo_lectivo", "id_seccion_miescuela"])["promedio_mate"].transform(imputar_si_suficientes)


## para los que sigo teniendo nulo, les imputo el promedio de la nota del bimestre
## de su distrito y promedio con eso

#promedio de notas por distrito
secciones_notas["distrito_escolar"] = secciones_notas["distrito_escolar"].astype(int)
promedios_por_distrito = secciones_notas.groupby(["distrito_escolar",'ciclo_lectivo'])[columnas_notas].mean()

# Recorrer cada columna de notas en 'secciones_notas' y completar con los promedios del distrito
for col in columnas_notas:
    # Imputar las notas faltantes con el promedio del distrito correspondiente
    secciones_notas[col] = secciones_notas.apply(
        lambda row: promedios_por_distrito.loc[(row['distrito_escolar'], row['ciclo_lectivo']), col]
        if pd.isna(row[col]) else row[col], axis=1
    )
    
# ahora les hago el promedio con eso

secciones_notas["promedio_lengua"] = secciones_notas.apply(
    lambda row: row[["a_n1_lengua", "a_n2_lengua", "a_n3_lengua", "a_n4_lengua"]].mean(skipna=True) 
    if row["ciclo_lectivo"] == "2022" 
    else row[["a_n1_lengua", "a_n2_lengua"]].mean(skipna=True), 
    axis=1
)

secciones_notas["promedio_mate"] = secciones_notas.apply(
    lambda row: row[["a_n1_mate", "a_n2_mate", "a_n3_mate", "a_n4_mate"]].mean(skipna=True) 
    if row["ciclo_lectivo"] == "2022" 
    else row[["a_n1_mate", "a_n2_mate"]].mean(skipna=True), 
    axis=1
)


secciones_notas['promedio_lengua'].isna().sum()


## AHORA QUE YA LE IMPUTE EL PROMEDIO Y LAS NOTAS A TODOS, HAGO EL RK DE MATERIAS
## DENTRO DE LA SECCION -- LO NORMALIZO SOBRE LA CANTIDAD DE ESTUDIANTES PARA HACERLO
## COMPARABLE ENTRE SECCIONES CON DIFERENTE CANTIDAD DE GENTE


# Calcular el ranking dentro de cada id_seccion_miescuela
secciones_notas["rk_mate"] = secciones_notas.groupby("id_seccion_miescuela")["promedio_mate"].rank(method="dense", ascending=False)
secciones_notas["rk_lengua"] = secciones_notas.groupby("id_seccion_miescuela")["promedio_lengua"].rank(method="dense", ascending=False)

# Obtener el total de estudiantes por seccion
total_estudiantes = secciones_notas["id_seccion_miescuela"].map(secciones_notas["id_seccion_miescuela"].value_counts())

# Normalizar los rankings para hacerlos comparables entre secciones
secciones_notas["rk_mate"] = secciones_notas["rk_mate"] / total_estudiantes
secciones_notas["rk_lengua"] = secciones_notas["rk_lengua"] / total_estudiantes

# por como está armado, un numero más bajo de rk_mate o rk_lengua implica un mejor
# desempeño del estudiante en esa materia


#sabiendo el rk de cada alumno, me llevo su data para correr el modelo 

sn_modelo = secciones_notas

# Lista de columnas a transponer
columnas_a_transformar = ["a_n1_mate", "a_n2_mate", "a_n3_mate", "a_n4_mate",
                          "a_n1_lengua", "a_n2_lengua", "a_n3_lengua", "a_n4_lengua",
                          "promedio_lengua", "promedio_mate", "rk_mate", "rk_lengua"]

# Transformar de ancho a largo
sn_modelo_melted = sn_modelo.melt(id_vars=["id_alumno", "ciclo_lectivo"], 
                                  value_vars=columnas_a_transformar, 
                                  var_name="variable", 
                                  value_name="valor")

# Agregar el prefijo del ciclo lectivo
sn_modelo_melted["variable"] = sn_modelo_melted["ciclo_lectivo"].astype(str).str[-2:] + "_" + sn_modelo_melted["variable"]

# Pivotear para que cada id_alumno tenga una sola fila
sn_modelo_pivot = sn_modelo_melted.pivot(index="id_alumno", columns="variable", values="valor").reset_index()

# Guardar el resultado en sn_modelo
sn_modelo = sn_modelo_pivot

sn_modelo.columns

#borro las cols de n3 y n4 de 2023 (no las use para el promedio pero quedaron ahi)
sn_modelo = sn_modelo.drop(columns=['23_a_n3_lengua', '23_a_n3_mate', '23_a_n4_lengua', '23_a_n4_mate'])


####################################
###### MODELO 4 BIS - MATRICULA, PPS, PASES y NOTAS + DESEMPEÑO RELATIVO
####################################

# parto dede la mtricul m3, porque la m4 tiene notas pero sin el procesamiento
# tengo notas para todos los etudiantes del df de PPS
len(set(matricula_m3['id_miescuela']) & set(sn_modelo['id_alumno']))

#hago el merge con el df del modelo anterior
matricula_m4b = matricula_m3.copy()
matricula_m4b = matricula_m4b.merge(sn_modelo, left_on="id_miescuela", right_on="id_alumno", how="left")

matricula_m4b['23_rk_mate'].isna().sum()

#elimino VD y armo los conjuntos
X_m4b = matricula_m4b.drop(columns=['documento', 'id_miescuela', '24_repite','id_alumno'])  # Excluir las columnas que no se usarán
y_m4b = matricula_m4b['24_repite']
X_train_m4b, X_test_m4b, y_train_m4b, y_test_m4b = train_test_split(X_m4b, y_m4b, test_size=0.2, random_state=42)

#modelo
model_m4b = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)

model_m4b.fit(X_train_m4b, y_train_m4b)
y_pred_m4b = model_m4b.predict(X_test_m4b)
accuracy_m4b = accuracy_score(y_test_m4b, y_pred_m4b)
print(f"Accuracy sobre el train: {accuracy_m4b:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m4b, model_m4b.predict_proba(X_train_m4b)[:, 1]):.4f}")
#prediccion
predicciones_m4b = model_m4b.predict(X_test_m4b)
#var imp
importance_m4b = model_m4b.get_booster().get_score(importance_type='weight')
sorted_importance_m4b = sorted(importance_m4b.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m4b, y_pred_m4b))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m4b, model_m4b.predict_proba(X_test_m4b)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m4b, y_pred_m4b):.6f}')

#matriz de valores
print(y_test_m4b.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m4b, y_pred_m4b).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#graficos
explainer_m4b = shap.Explainer(model_m4b, X_train_m4b)
expected_value_m4b = explainer_m4b.expected_value
prob_base_m4b = 1 / (1 + np.exp(-expected_value_m4b))
print('Valor de prediccion de base: ', prob_base_m4b)
shap_values_m4b = explainer_m4b(X_test_m4b)
shap.summary_plot(shap_values_m4b, X_test_m4b)
shap.summary_plot(shap_values_m4b, X_test_m4b, plot_type="bar")



##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs4b = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m4b[y_train_m4b == 0]) / len(y_train_m4b[y_train_m4b == 1])  # Incluir scale_pos_weight aquí
    )

random_search4b = RandomizedSearchCV(
    estimator=model_rs4b,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search4b.fit(X_train_m4b, y_train_m4b)
print("Mejores parámetros:", random_search4b.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search4b.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m4b, random_search4b.predict_proba(X_train_m4b)[:, 1]):.4f}")
best_model_4rsb = random_search4b.best_estimator_

# Predicción
y_pred_m4b = best_model_4rsb.predict(X_test_m4b)

# Métricas
print(classification_report(y_test_m4b, y_pred_m4b))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m4b, best_model_4rsb.predict_proba(X_test_m4b)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m4b, y_pred_m4b):.6f}')

#matriz de valores
print(y_test_m4b.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m4b, y_pred_m4b).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m4_rsb = shap.Explainer(best_model_4rsb, X_train_m4b)
expected_value_m4b = explainer_m4_rsb.expected_value
prob_base_m4b = 1 / (1 + np.exp(-expected_value_m4b))
print('Valor de prediccion de base: ', prob_base_m4b)
shap_values_m4b = explainer_m4b(X_test_m4b)
shap.summary_plot(shap_values_m4b, X_test_m4b)
shap.summary_plot(shap_values_m4b, X_test_m4b, plot_type="bar")

####################################
###### MODELO 4 C - MATRICULA, PPS, PASES y NOTAS + DESEMPEÑO RELATIVO SOLO PROMEDIOS
####################################

snc_modelo = sn_modelo[['id_alumno','22_promedio_lengua','22_promedio_mate','22_rk_mate',
                        '22_rk_lengua','23_promedio_lengua','23_promedio_mate','23_rk_mate','23_rk_lengua']]

#hago el merge con el df del modelo anterior
matricula_m4c = matricula_m3.copy()
matricula_m4c = matricula_m4c.merge(snc_modelo, left_on="id_miescuela", right_on="id_alumno", how="left")


#elimino VD y armo los conjuntos
X_m4c = matricula_m4c.drop(columns=['documento', 'id_miescuela', '24_repite','id_alumno'])  # Excluir las columnas que no se usarán
y_m4c = matricula_m4c['24_repite']
X_train_m4c, X_test_m4c, y_train_m4c, y_test_m4c = train_test_split(X_m4c, y_m4c, test_size=0.2, random_state=42)


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

model_rs4c = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m4c[y_train_m4c == 0]) / len(y_train_m4c[y_train_m4c == 1])  # Incluir scale_pos_weight aquí
    )

random_search4c = RandomizedSearchCV(
    estimator=model_rs4c,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search4c.fit(X_train_m4c, y_train_m4c)
print("Mejores parámetros:", random_search4c.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search4c.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m4c, random_search4c.predict_proba(X_train_m4c)[:, 1]):.4f}")
best_model_4rsc = random_search4c.best_estimator_

# Predicción
y_pred_m4c = best_model_4rsc.predict(X_test_m4c)

# Métricas
print(classification_report(y_test_m4c, y_pred_m4c))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m4c, best_model_4rsc.predict_proba(X_test_m4c)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m4c, y_pred_m4c):.6f}')

#matriz de valores
print(y_test_m4c.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m4c, y_pred_m4c).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m4_rsc = shap.Explainer(best_model_4rsc, X_train_m4c)
expected_value_m4c = explainer_m4_rsc.expected_value
prob_base_m4c = 1 / (1 + np.exp(-expected_value_m4c))
print('Valor de prediccion de base: ', prob_base_m4c)
shap_values_m4c = explainer_m4_rsc(X_test_m4c)
shap.summary_plot(shap_values_m4c, X_test_m4c)
shap.summary_plot(shap_values_m4c, X_test_m4c, plot_type="bar")



#############################################################################
###################    PROCESAMIENTO DE APOYOS       ########################
#############################################################################

'''
## TRABAJO CON LA BASE DE APOYOS PARA PODER DEJAR COLS BINARIAS AGRUPADAS X EL CL

apoyos_consolidado = apoyos.groupby(['id_alumno', 'periodo']).agg(
    ag_apoyo=('ag_apoyo', lambda x: 'Sí' if 'Sí' in x.values else 'No'),
    ag_apoyo_tipo=('ag_apoyo_tipo', lambda x: list({item for sublist in x.dropna().apply(eval) for item in sublist})),
).reset_index()

unique_values = set(val for sublist in apoyos_consolidado['ag_apoyo_tipo'] for val in sublist)

for value in unique_values:
    apoyos_consolidado[f'ag_at_{value}'] = apoyos_consolidado['ag_apoyo_tipo'].apply(lambda x: 1 if value in x else 0)

apoyos_merge = apoyos_consolidado.drop(columns=['ag_apoyo_tipo'])

## ACA VALE LA PENA DEJARLO APERTURADO POR PERIODO O DEJO UNA FILA X ALUMNO
'''

#############################################################################
###################   PROCESAMIENTO DE SERVICIOS  ###########################
#############################################################################

# quiero ver la distribución de los programas entre la matrícula

servicios_filtrado = servicios[servicios['documento'].isin(matricula_m4['documento'])]

check = (
    servicios_filtrado
    .groupby("ciclo_lectivo")
    .agg({"documento": "nunique", **{col: "sum" for col in servicios.columns if col not in ["ciclo_lectivo", "documento"]}})
    .reset_index()
)

#saco aprende programando y transporte porque son 0 en todos los casos
#saco becas media para 2022 porque no es un programa qeu aplique en ese caso

servicios = servicios.drop(columns=['transporte', 'aprende_programando'])
servicios_p = servicios.melt(id_vars=['documento', 'id_persona', 'ciclo_lectivo'], 
                                   var_name='programa', 
                                   value_name='valor')

#Crear una nueva columna combinando 'ciclo_lectivo' y 'programa'
servicios_p['columna'] = servicios_p['ciclo_lectivo'].astype(str) + '_' + servicios_p['programa']

#Pivotar para que cada 'documento'/'id_persona' tenga una fila
servicios_p = servicios_p.pivot_table(index=['documento', 'id_persona'], 
                                        columns='columna', 
                                        values='valor', 
                                        aggfunc='first', 
                                        fill_value=0)

servicios_p = servicios_p.reset_index()
#borro porque es un programa de secu y estan en primaria
servicios_p = servicios_p.drop(columns=['2022_becas_media','id_persona'])


####################################
###### MODELO 6 - MATRICULA, PPS, PASES, NOTAS c/ RK y SERVICIOS
####################################

#matricula_m6 = matricula_m4.copy()
matricula_m6 = matricula_m4b.copy()
matricula_m6 = matricula_m6.merge(servicios_p, on='documento', how="left")

#elimino VD y armo los conjuntos
X_m6 = matricula_m6.drop(columns=['documento', 'id_miescuela', '24_repite'])  # Excluir las columnas que no se usarán
y_m6 = matricula_m6['24_repite']
X_train_m6, X_test_m6, y_train_m6, y_test_m6 = train_test_split(X_m6, y_m6, test_size=0.2, random_state=42)

#modelo
model_m6 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)

model_m6.fit(X_train_m6, y_train_m6)
y_pred_m6 = model_m6.predict(X_test_m6)
accuracy_m6 = accuracy_score(y_test_m6, y_pred_m6)
print(f"Accuracy sobre el train: {accuracy_m6:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m6, model_m6.predict_proba(X_train_m6)[:, 1]):.4f}")
#prediccion
predicciones_m6 = model_m6.predict(X_test_m6)
#var imp
importance_m6 = model_m6.get_booster().get_score(importance_type='weight')
sorted_importance_m6 = sorted(importance_m6.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m6, y_pred_m6))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m6, model_m6.predict_proba(X_test_m6)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m6, y_pred_m6):.6f}')

#matriz de valores
print(y_test_m6.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m6, y_pred_m6).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#graficos
explainer_m6 = shap.Explainer(model_m6, X_train_m6)
expected_value_m6 = explainer_m6.expected_value
prob_base_m6 = 1 / (1 + np.exp(-expected_value_m6))
print('Valor de prediccion de base: ', prob_base_m6)
shap_values_m6 = explainer_m6(X_test_m6)
shap.summary_plot(shap_values_m6, X_test_m6)
shap.summary_plot(shap_values_m6, X_test_m6, plot_type="bar")


##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs6 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m6[y_train_m6 == 0]) / len(y_train_m6[y_train_m6 == 1])  # Incluir scale_pos_weight aquí
    )

random_search6 = RandomizedSearchCV(
    estimator=model_rs6,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search6.fit(X_train_m6, y_train_m6)
print("Mejores parámetros:", random_search6.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search6.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m6, random_search6.predict_proba(X_train_m6)[:, 1]):.4f}")
best_model_6rs = random_search6.best_estimator_

# Predicción
y_pred_m6 = best_model_6rs.predict(X_test_m6)

# Métricas
print(classification_report(y_test_m6, y_pred_m6))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m6, best_model_6rs.predict_proba(X_test_m6)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m6, y_pred_m6):.6f}')

#matriz de valores
print(y_test_m6.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m6, y_pred_m6).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m6_rs = shap.Explainer(best_model_6rs, X_train_m6)
expected_value_m6 = explainer_m6_rs.expected_value
prob_base_m6 = 1 / (1 + np.exp(-expected_value_m6))
print('Valor de prediccion de base: ', prob_base_m6)
shap_values_m6 = explainer_m6_rs(X_test_m6)
shap.summary_plot(shap_values_m6, X_test_m6)
shap.summary_plot(shap_values_m6, X_test_m6, plot_type="bar")




#############################################################################
################    PROCESAMIENTO DE RESPONSABLES       #####################
#############################################################################
resp_bu = responsables.copy()


responsables['nac_resp'].unique()
responsables['nivel_educativo'].unique()
responsables['vinculo'].unique()

#nivel educativo
nivel_educativo = {'sin estudios':0, 'primario incompleto':1,
                   'primario completo':2, 'secundario incompleto':3,
                   'secundario completo':4, 'terciario incompleto':5, 
                   'terciario completo':6, 'universitario incompleto':7,
                   'universitario completo':8,'posgrado':9}
responsables['nivel_educativo'] = responsables['nivel_educativo'].str.strip().str.lower().replace(nivel_educativo)

#nacionalidad del responsable
nacionalidades_hispanas = [
    'Argentina', 'Bolivia', 'Perú', 'Venezuela', 'Paraguay', 'España', 
    'Uruguay', 'República Dominicana', 'Ecuador', 'Colombia', 'Chile', 
    'México', 'Cuba'
]

# Crear una columna que indique si la nacionalidad es de habla hispana
responsables['nac_hispana'] = responsables['nac_resp'].apply(lambda x: 1 if x in nacionalidades_hispanas else 0)

#imputo los responsables
map_vinculo = {
    'resp_nadie': ['no aplica'],
    'resp_padres': ['madre', 'padre'],
    'resp_hermanos': ['hermano/a'],
    'resp_abuelos': ['abuelo/a'],
    'resp_tios': ['tío/a'],
    'resp_padrastros': ['padrastro', 'madrastra'],
    'resp_tutores': ['tutor/a','autorizado/a'],
    'resp_primos': ['primo/a']
}


def mapear_vinculo(vinculo):
    for grupo, valores in map_vinculo.items():
        if vinculo in valores:
            return grupo
    return 'otro'  # En caso de que no coincida con ninguno de los valores mapeados

# Aplicar la función a la columna 'vinculo' del DataFrame
responsables['vinculo_map'] = responsables['vinculo'].apply(mapear_vinculo)

# Crear columnas binarias para cada grupo en el mapeo
for grupo in map_vinculo.keys():
    responsables[grupo] = responsables['vinculo_map'].apply(lambda x: 1 if x == grupo else 0)

#me quedo con las col que me interesan para el modelo
resp_modelo = responsables.drop(columns=['doc_resp','nac_resp','vinculo',
                                         'vinculo_map','doc_alu'])

resp_modelo = resp_modelo.sort_values(by='id_miescuela')

# me quedo con el registro, para cada alumno, de su responsable con mayor nivel educativo
# en caso de que tengan más de uno

resp_modelo_max = resp_modelo.loc[resp_modelo.groupby('id_miescuela')['nivel_educativo'].idxmax()]


####################################
###### MODELO 7 - MATRICULA, PPS, PASES, PROMEDIOS c/ RK y RESPONSABLES
####################################

matricula_m7 = matricula_m4c.copy()
matricula_m7 = matricula_m7.merge(resp_modelo_max, on='id_miescuela', how="left")

#elimino VD y armo los conjuntos
X_m7 = matricula_m7.drop(columns=['documento', 'id_miescuela', '24_repite','id_alumno'])  # Excluir las columnas que no se usarán
y_m7 = matricula_m7['24_repite']
X_train_m7, X_test_m7, y_train_m7, y_test_m7 = train_test_split(X_m7, y_m7, test_size=0.2, random_state=42)

#modelo
model_m7 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=500,  # Aumentar el número de árboles
    learning_rate=0.05,  # Reducir la tasa de aprendizaje para evitar overfitting
    max_depth=6,  # Controla la profundidad de los árboles
    colsample_bytree=0.8,  # Para usar una fracción de las features en cada árbol
    subsample=0.8,  # Para usar una fracción de los datos en cada iteración
    random_state=42
)

model_m7.fit(X_train_m7, y_train_m7)
y_pred_m7 = model_m7.predict(X_test_m7)
accuracy_m7 = accuracy_score(y_test_m7, y_pred_m7)
print(f"Accuracy sobre el train: {accuracy_m7:.4f}")
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m7, model_m7.predict_proba(X_train_m7)[:, 1]):.4f}")
#prediccion
predicciones_m7 = model_m7.predict(X_test_m7)
#var imp
importance_m7 = model_m7.get_booster().get_score(importance_type='weight')
sorted_importance_m7 = sorted(importance_m7.items(), key=lambda x: x[1], reverse=True)
#analisis de las predicciones
print(classification_report(y_test_m7, y_pred_m7))  # Para obtener precisión, recall, f1-score
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m7, model_m7.predict_proba(X_test_m7)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m7, y_pred_m7):.6f}')

#matriz de valores
print(y_test_m7.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m7, y_pred_m7).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#graficos
explainer_m7 = shap.Explainer(model_m7, X_train_m7)
expected_value_m7 = explainer_m7.expected_value
shap_values_m7 = explainer_m7(X_test_m7)
shap.summary_plot(shap_values_m7, X_test_m7)
shap.summary_plot(shap_values_m7, X_test_m7, plot_type="bar")


##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs7 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m7[y_train_m7 == 0]) / len(y_train_m7[y_train_m7 == 1])  # Incluir scale_pos_weight aquí
    )

random_search7 = RandomizedSearchCV(
    estimator=model_rs7,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search7.fit(X_train_m7, y_train_m7)
print("Mejores parámetros:", random_search7.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search7.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m7, random_search7.predict_proba(X_train_m7)[:, 1]):.4f}")
best_model_7rs = random_search7.best_estimator_

# Predicción
y_pred_m7 = best_model_7rs.predict(X_test_m7)

# Métricas
print(classification_report(y_test_m7, y_pred_m7))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m7, best_model_7rs.predict_proba(X_test_m7)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m7, y_pred_m7):.6f}')

#matriz de valores
print(y_test_m7.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m7, y_pred_m7).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m7_rs = shap.Explainer(best_model_7rs, X_train_m7)
expected_value_m7 = explainer_m7_rs.expected_value
shap_values_m7 = explainer_m7_rs(X_test_m7)
shap.summary_plot(shap_values_m7, X_test_m7)
shap.summary_plot(shap_values_m7, X_test_m7, plot_type="bar")





###############
#           SIMULO QUE ESTOY A PPIO DE 2023, QUÉ IMPORTA MAS?
###############


col_m7 = matricula_m7.columns

matricula_m7b = matricula_m7.loc[:, ~matricula_m7.columns.str.startswith("23_")]

#elimino VD y armo los conjuntos
X_m7b = matricula_m7b.drop(columns=['documento', 'id_miescuela', '24_repite','id_alumno'])  # Excluir las columnas que no se usarán
y_m7b = matricula_m7b['24_repite']
X_train_m7b, X_test_m7b, y_train_m7b, y_test_m7b = train_test_split(X_m7b, y_m7b, test_size=0.2, random_state=42)

##### MISMO PERO CON UN RANDOMSEARCH

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

model_rs7b = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m7b[y_train_m7b == 0]) / len(y_train_m7b[y_train_m7b == 1])  # Incluir scale_pos_weight aquí
    )

random_search7b = RandomizedSearchCV(
    estimator=model_rs7b,
    param_distributions=param_dist,  # Espacio de parámetros para la búsqueda aleatoria
    n_iter=100,  # Número de combinaciones aleatorias que se probarán
    scoring='roc_auc',  # Queremos maximizar AUC
    cv=3,  # Validación cruzada con 3 particiones
    verbose=1,  # Muestra el progreso
    random_state=42,
    n_jobs=-1  # Usamos todos los núcleos de la CPU
    )


#fiteo del modelo
random_search7b.fit(X_train_m7b, y_train_m7b)
print("Mejores parámetros:", random_search7b.best_params_)
print("Mejor AUC-ROC cruzado CV=3:", random_search7b.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m7b, random_search7b.predict_proba(X_train_m7b)[:, 1]):.4f}")
best_model_7rsb = random_search7b.best_estimator_

# Predicción
y_pred_m7b = best_model_7rsb.predict(X_test_m7b)

# Métricas
print(classification_report(y_test_m7b, y_pred_m7b))
print(f"AUC ROC sobre test: {roc_auc_score(y_test_m7b, best_model_7rsb.predict_proba(X_test_m7b)[:, 1]):.4f}")
print(f'Accuracy del test: {accuracy_score(y_test_m7b, y_pred_m7b):.6f}')

#matriz de valores
print(y_test_m7b.sum())
tn, fp, fn, tp = confusion_matrix(y_test_m7b, y_pred_m7b).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


#analisis grafico
explainer_m7_rsb = shap.Explainer(best_model_7rsb, X_train_m7b)
expected_value_m7b = explainer_m7_rsb.expected_value
shap_values_m7b = explainer_m7_rsb(X_test_m7b)
shap.summary_plot(shap_values_m7b, X_test_m7b)
shap.summary_plot(shap_values_m7b, X_test_m7b, plot_type="bar")












































#############################################################################
###################     PROCESAMIENTO DE DOMICILIO   ########################
#############################################################################

dd = dd.sort_values(by=['documento', 'domicilio_renaper', 'mas_reciente'], ascending=[True, False, True])
dd = dd.groupby('documento').first().reset_index()


# Inicializa el geocodificador
geolocator = Nominatim(user_agent="geoapi_direcciones")

dd['ciudad_aux'] = dd['ciudad'].fillna(dd['localidad']).fillna(dd['partido'])
dd['ciudad_aux'] = dd['ciudad_aux'].str.replace('_', ' ')
dd['Direccion'] = dd['calle'].fillna('') + ' ' + dd['altura'].fillna('').astype(str) + ', ' + dd['ciudad_aux'].fillna('') + ', ' + dd['provincia'].fillna('') + ', Argentina'
dd['Direccion'] = dd['Direccion'].str.replace(r'(,\s*)+', ', ', regex=True).str.strip(', ')

dd[['latitud', 'longitud']] = dd.apply(obtener_coordenadas, axis=1)



#############################################################################
#################   PROCESAMIENTO DE SOCIOECONOMICO   #######################
#############################################################################

dse = dse.sort_values(by=['documento', 'flag_renaper'], ascending=[True, False])

## check de nulos
dse['ingresos_grupo_familiar'].notnull().sum() 
((dse['ingresos_grupo_familiar'].notnull()) & (dse['ingresos_grupo_familiar'] > 0)).sum() 
#14419 // hay un solo reg con datos pero sin upd // 10768 mayres a 0
dse['sueldo'].notnull().sum() 
((dse['sueldo'].notnull()) & (dse['sueldo'] > 0)).sum() 
#10927 no nulos// 164 con sueldo no nulo y mayor a 0
dse['pension'].notnull().sum()
((dse['pension'].notnull()) & (dse['pension'] > 0)).sum() 
#10699 no nulos // 19 no nulos y mayores a 0


#ver cuantos pares de no nulos de ambos lados tenemos para cada columna
columnas_ingresos = ['ingresos_grupo_familiar', 'sueldo', 'pension']
columnas_upd = ['upd_ingresos_grupo_familiar', 'upd_sueldo', 'upd_pension']
resultados = {}
for ingresos, upd in zip(columnas_ingresos, columnas_upd):
    no_nulos = dse[ingresos].notnull() & dse[upd].notnull()
    resultados[f'{ingresos} - {upd}'] = no_nulos.sum()


condiciones = [(dse[ingresos].notnull() & (dse[ingresos] > 0) & dse[upd].notnull())
    for ingresos, upd in zip(columnas_ingresos, columnas_upd)]

filtro_final = condiciones[0]
for condicion in condiciones[1:]:
    filtro_final |= condicion
filtro_final.sum()
# 10841 tienen registros no nulos y mayores a 0 para algunos de los pares

#############################################################################
###############     PROCESAMIENTO DE DATOS PERSONALES   #####################
#############################################################################

# CHEQUEAR SI ACA YA TENGO UNA SOLA FILA PARA CADA PERSONA Y PUEDO MERGEAR X DOC
#O TENGO QUE MERGEAR X DOC + FLAG_RENAPER

dse['documento'].nunique()
dse['documento'] = dse['documento'].astype(str).str.strip().str.upper()
dse_m = dse.sort_values(by=['documento', 'flag_renaper'], ascending=[True, False]) \
                             .drop_duplicates(subset='documento', keep='first')
dd['documento'] = dd['documento'].str.strip().str.upper()
dd_m = dd.sort_values(by=['documento', 'flag_renaper'], ascending=[True, False]) \
                             .drop_duplicates(subset='documento', keep='first')
ds['documento'] = ds['documento'].str.strip().str.upper()
ds_m = ds.sort_values(by=['documento', 'flag_renaper'], ascending=[True, False]) \
                             .drop_duplicates(subset='documento', keep='first')
                             
## CREACION DE LA TABLA DE DATOS PERSONALES DE TODOS LOS INVOLUCRADOS

doc_alu = matricula['documento'].unique()
doc = pd.DataFrame(doc_alu)
doc.columns = ['documento']
doc = pd.merge(doc,dd_m,on='documento',how='left')
doc = pd.merge(doc,dse_m,on='documento',how='left')
doc = pd.merge(doc,ds,on='documento',how='left')
doc = doc.drop(columns=['flag_renaper_x','flag_renaper_y','flag_renaper.1','depto','piso',
                        'nhp','certificado_discapacidad','nombre_obra_soc','num_obra_soc',
                        'alergias','disc_motora','disc_otros','epileptico','trat_neurologia',
                        'otro_tratamiento','coord_x','coord_y','sistema_salud','grupo_sanguineo',
                        'posee_alergias','disc_mental','disc_sensorial','disc_otros_descripcion',
                        'trat_psicopedagogia','trat_terapia','trat_psicologia'])


# tratamiento de variables de discapacidad para poder tenerlas
doc[['discapacitado', 'disc_ninguno']].drop_duplicates()
doc['flag_discapacidad'] = np.where((doc['discapacitado'] == 1), 1,np.where(
        (doc['discapacitado'] == 0), 0,np.where((doc['discapacitado'].isnull()) 
                                                & (doc['disc_ninguno'].isnull()), 
            np.nan,np.where((doc['discapacitado'].isnull()) & (doc['disc_ninguno'] == 0),
                            1,np.nan))))
doc = doc.drop(columns=['discapacitado','disc_ninguno'])
doc['flag_discapacidad'].value_counts(dropna=False)

#tratamiento de las variables de domicilio
doc['provincia'] = doc['provincia'].replace('CIUDAD AUTÓNOMA DE BUENOS AIRES', 'CABA')

doc['provincia'].unique()
doc['provincia'].value_counts(dropna=False)

doc['provincia'] = doc.apply(
    lambda row: 'CABA' if pd.isnull(row['provincia']) and 
                (row['localidad'] in ['CABA', 'C.A.B.A.', 'CIUDAD AUTÓNOMA DE BUENOS AIRES', 'CAPITAL FEDERAL'] or 
                 'CABA' in str(row['localidad']).upper()) 
                else ('BUENOS AIRES' if row['localidad'] == 'PROVINCIA' else row['provincia']),axis=1)

#imputar el valor de villa buscnado en barrio
valores = [
    "VILLA 1-11-14", "VILLA 13 BIS", "VILLA 15", "VILLA 16", "VILLA 17",
    "VILLA 19", "VILLA 20", "VILLA 21-24", "VILLA 3- BO. FATIMA", 
    "VILLA 31", "VILLA 31 BIS", "VILLA 6", "VILLA CALACITA", "VILLA PILETONES"
]
doc['villa'] = doc.apply(lambda row: 1 if ('ASENTAMIENTO' in str(row['barrio']).upper() 
                                           or str(row['barrio']).upper() in valores) else row['villa'],axis=1)

#en la col barrio solo tengo asentamientos, entonces la convierto en asentamiento
#y me quedo el nombre ahi
doc.rename(columns={'barrio': 'asentamiento'}, inplace=True)
doc['asentamiento'] = doc.apply(lambda row: row['asentamiento'] if row['villa'] == 1 else np.nan,axis=1)

#completo la ciudad de los que declaran vivr en un asentamiento con el barrio
#donde queda ese asentamiento
asentamiento_ciudad_map = {"VILLA 21-24": "BARRACAS","VILLA 1-11-14": "FLORES",
    "VILLA 20": "VILLA LUGANO","VILLA 19": "VILLA LUGANO","VILLA 31 BIS": "RETIRO",
    "VILLA 15": "VILLA LUGANO","BARRIO PAPA FRANCISCO": "VILLA LUGANO",
    "VILLA 3- BO. FATIMA": "VILLA SOLDATI","NHT ZAVALETA": "BARRACAS",
    "ASENTAMIENTO BARRIO OBRERO": "VILLA LUGANO","ASENTAMIENTO SCAPINO": "VILLA LUGANO",
    "ASENTAMIENTO RODRIGO BUENO": "PUERTO MADERO","ASENTAMIENTO LA CARBONILLA": "LA PATERNAL",
    "VILLA 6": "PARQUE AVELLANEDA","ASENTAMIENTO FRAGA": "CHACARITA","VILLA 31": "RETIRO",
    "VILLA PILETONES": "VILLA SOLDATI","ASENTAMIENTO MARÍA AUXILIADORA": "VILLA LUGANO",
    "VILLA 17": "VILLA LUGANO","ASENTAMIENTO WARNES Y NEWBERY": "CHACARITA",
    "NHT DEL TRABAJO": "VILLA LUGANO","ASENTAMIENTO PORTELA": "VILLA SOLDATI",
    "ASENTAMIENTO LOS PINOS":"VILLA SOLDATI","VILLA CALACITA":"VILLA SOLDATI",
    "VILLA 13 BIS":"FLORES","ASENTAMIENTO BERMEJO":"VILLA LUGANO",
    "ASENTAMIENTO CALLE B. MITRE":"BALVANERA","ASENTAMIENTO EL PUEBLITO":"POMPEYA",
    "ASENTAMIENTO LAMADRID":"LA BOCA","ASENTAMIENTO SALDÍAS":"RETIRO","ASENTAMIENTO BOSCH":"BARRACAS"}

doc['ciudad'] = doc.apply(
    lambda row: asentamiento_ciudad_map[row['asentamiento']] 
                if row['asentamiento'] in asentamiento_ciudad_map 
                else row['ciudad'],axis=1)

#en caso de que la provincia sea CABA que el partido sea nulo porque no aplica
doc.loc[doc['provincia'] == 'CABA', 'partido'] = np.nan


localidades = doc['localidad'].unique()

###########  fill de nulos en localidad

#tengo CPS con localidades duplicadas, entonces no me sirve
#porque me multiplica los registros

'''
cps = pd.merge(localidades,provincias,left_on='idProvincia',right_on='id', how='left')
cps = cps[['localidad', 'provincia', 'cp']]
cps['cp'] = cps['cp'].astype(str)
cps['localidad'] = cps['localidad'].str.replace(r'\s*\(.?\)\s', '', regex=True)
cps = cps.sort_values(by='cp', ascending=True)

#hago string este campo y no integer el otro porque corro riesgo de fabricar
#un cp falso a partir de info basura

dd_comp = pd.merge(dd_m,cps,left_on='codigo_postal',right_on='cp',how='left')
dd_comp['ciudad'] = dd_comp['ciudad'].combine_first(dd_comp['localidad'])
dd_comp['provincia'] = dd_comp['provincia'].combine_first(dd_comp['provincia_y'])

cps['cp'].nunique()
codigo_postal_unicos = dd_m['codigo_postal'].unique()
'''
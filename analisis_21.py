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
#notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_bbdd.csv', low_memory=False)
#pps = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps_documento.csv', low_memory=False)
#responsables = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.responsables_personas.csv', low_memory = False)
#dse = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dse_personas.csv', low_memory = False)
#dd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dd_personas.csv', low_memory = False)
#ds = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ds_personas.csv', low_memory = False)
#pases = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pases.csv', low_memory = False)
#localidades = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.localidades.csv', low_memory = False, delimiter = ';')
#provincias = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.provincias.csv', low_memory = False, delimiter = ';')


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

#Función para obtener coordenadas
def obtener_coordenadas(row, index, columna_direccion):
    try:
        # Imprimir el progreso cada 10 filas
        if index % 10 == 0:
            print(f"Procesando fila {index}...")
        direccion = row[columna_direccion]  # Dirección original
        
        # Primera estrategia: búsqueda directa con Nominatim
        ubicacion = geolocator.geocode(direccion, timeout=10)
        # Segunda estrategia: quitar ", Argentina"
        if not ubicacion:
            direccion_alternativa = direccion.replace(", Argentina", "").strip()
            ubicacion = geolocator.geocode(direccion_alternativa, timeout=10)

        # Tercera estrategia: usar la API pública de Google Geocoding (sin API Key)
        if not ubicacion:
            google_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={direccion}&sensor=false"
            response = requests.get(google_url)
            data = response.json()
            if data.get("status") == "OK":
                ubicacion = data["results"][0]["geometry"]["location"]

        # Si encontramos una ubicación, devolver las coordenadas
        if ubicacion:
            # Si Google Maps encontró las coordenadas, usamos las de Google
            if isinstance(ubicacion, dict):  # En caso de ser un diccionario como con Google
                return pd.Series([ubicacion["lat"], ubicacion["lng"]])
            # Si se encontró con Geopy, usamos sus atributos lat/lng
            return pd.Series([ubicacion.latitude, ubicacion.longitude])
        else:
            return pd.Series([None, None])
    except GeocoderTimedOut:
        return pd.Series([None, None])
    except Exception as e:
        print(f"Error en la obtención de coordenadas: {e}")
        return pd.Series([None, None])

'''
###########################################################################
                    PROCESAMIENTO DE LAS DISTINTAS BASES
###########################################################################
'''

#############################################################################
###################   PROCESAMIENTO DE MATRICULA  ###########################
#############################################################################

matricula['repite'].sum()

'''
matricula['altura'] = matricula['altura'].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() != '' else '')
matricula['Direccion'] = matricula['calle'].fillna('') + ' ' + matricula['altura'] + ', ' + matricula['barrio'].fillna('') + ', Ciudad de Buenos Aires, Argentina'
matricula['Direccion2'] = matricula['calle'].fillna('') + ' ' + matricula['altura'] + ', Ciudad de Buenos Aires , Argentina'
geolocator = Nominatim(user_agent="mi_aplicacion")
direcciones = pd.DataFrame({'Direccion': matricula['Direccion'].dropna().unique(),'Direccion2': matricula['Direccion2'].dropna().unique()})
direcciones[['latitud', 'longitud']] = direcciones.apply(lambda row: obtener_coordenadas(row, row.name,'Direccion2'), axis=1)
direcciones = direcciones.sort_values(by=['latitud'], ascending=True, na_position='first')
#vuelvo a correr sobre los casos que no se completaron antes
mascara_nulos = direcciones['latitud'].isna() & direcciones['longitud'].isna()
diccionario = {'HUMBERTO Iº 1573, Ciudad de Buenos Aires , Argentina':[-34.62105777913497, -58.38821210318528],
               'CNEL. MARTINIANO CHILAVERT 5460, Ciudad de Buenos Aires , Argentina':[-34.67904025544257, -58.467149062704536],
               'PALOS Y JUAN MANUEL BLANES S/N , Ciudad de Buenos Aires , Argentina':[-34.632700550024396, -58.3642277627076],
               'AVDA. GRAL. LAS HERAS 4078, Ciudad de Buenos Aires , Argentina':[-34.5817899417371, -58.41622037435185],
               'CNEL. RAMON LISTA 5256, Ciudad de Buenos Aires , Argentina':[-34.61400660950096, -58.52352952838179],
               'CNEL. RAMON L. FALCON 2934, Ciudad de Buenos Aires , Argentina':[-34.61386154691175, -58.523588178052464],
               'CNEL. RAMON L. FALCON 2248, Ciudad de Buenos Aires , Argentina':[-34.629294753227924, -58.459864462707834],
               'GRAL. HORNOS 530, Ciudad de Buenos Aires , Argentina':[-34.63374807439701, -58.377102976199886],
               'ALMTE. FRANCISCO J. SEGUI 2580, Ciudad de Buenos Aires , Argentina':[-34.596656971759174, -58.46113334921762],
               'RIO CUARTO Y MONTESQUIEU - AVDA. GRAL. IRIARTE ALT. 3501 , Ciudad de Buenos Aires , Argentina':[-34.651149819693615, -58.396749837083256],
               'GRAL. MARTIN DE GAINZA 1050, Ciudad de Buenos Aires , Argentina':[-34.6091136408391, -58.44679349868182],
               'CNEL. RAMON L. FALCON 4151, Ciudad de Buenos Aires , Argentina':[-34.635426865076724, -58.485636862707416],
               'CNEL. RAMON L. FALCON 6702, Ciudad de Buenos Aires , Argentina':[-34.64020314378938, -58.52147440688657],
               'PJE. LA CONSTANCIA 2524, Ciudad de Buenos Aires , Argentina':[-34.647468126906894, -58.43216219154246],
               'PJE. L E/LACARRA Y LAGUNA , Ciudad de Buenos Aires , Argentina':[-34.65858792216857, -58.456352218526405],
               'GRAL. URQUIZA 277, Ciudad de Buenos Aires , Argentina':[-34.613582814804055, -58.410103389693404],
               'CNEL. RAMON L. FALCON 4126, Ciudad de Buenos Aires , Argentina':[-34.63581977836494, -58.48471490503556],
               'PTE. CAMILO TORRES Y TENORIO 2147, Ciudad de Buenos Aires , Argentina':[-34.64612146079697, -58.43853780503492],
               'PJE. LA PORTEÑA 54, Ciudad de Buenos Aires , Argentina':[-34.62932618438913, -58.468228191543666],
               'AVDA. GRAL. FERNANDEZ DE LA CRUZ 3605, Ciudad de Buenos Aires , Argentina':[-34.66378610950963, -58.449064862705484],
               'HUMBERTO Iº 343, Ciudad de Buenos Aires , Argentina':[-34.62022109903295, -58.37055973387255],
               'PTE. LUIS SAENZ PEÑA 463, Ciudad de Buenos Aires , Argentina':[-34.614186930974675, -58.38775653640017],
               'GRAL. MANUEL A. RODRIGUEZ 2332, Ciudad de Buenos Aires , Argentina':[-34.5985894242755, -58.457134605037965],
               'GRAL. LUCIO NORBERTO MANSILLA 3643, Ciudad de Buenos Aires , Argentina':[-34.59051286249745, -58.41487614736672],
               'CNEL. MARTINIANO CHILAVERT 2690, Ciudad de Buenos Aires , Argentina':[-34.65540386512049, -58.440250989690625],
               'AVDA. PTE. MANUEL QUINTANA 31, Ciudad de Buenos Aires , Argentina':[-34.59207402674483, -58.38504687805387],
               'GRAL. GREGORIO ARAOZ DE LAMADRID 499, Ciudad de Buenos Aires , Argentina':[-34.63707314410623, -58.35924266085589],
               'AVDA. CNEL. CARDENAS 2652, Ciudad de Buenos Aires , Argentina':[-34.66623303972807, -58.5022398050336],
               'CNEL. RAMON L. FALCON 4801, Ciudad de Buenos Aires , Argentina':[-34.63784686400738, -58.49498879154311],
               'GRAL. ENRIQUE MARTINEZ 1432, Ciudad de Buenos Aires , Argentina':[-34.57509895686678, -58.46068836271128],
               'AVDA. CNEL. ROCA - PTA. 9 , Ciudad de Buenos Aires , Argentina':[-34.698415981561396, -58.47039687434422],
               'CNEL. APOLINARIO FIGUEROA 661, Ciudad de Buenos Aires , Argentina':[-34.606169299115095, -58.44876154736575],
               'AVDA. GRAL. LAS HERAS 3086, Ciudad de Buenos Aires , Argentina':[-34.58363018134349, -58.4054547203825],
               'MCAL. ANTONIO JOSE DE SUCRE 1367, Ciudad de Buenos Aires , Argentina':[-34.557530422661195, -58.44467006133722],
               'CNEL. PEDRO CALDERON DE LA BARCA 3073, Ciudad de Buenos Aires , Argentina':[-34.61354566555254, -58.52241267434972],
               'HUMBERTO Iº 3171, Ciudad de Buenos Aires , Argentina':[-34.623391353056235, -58.41015896270817],
               'GRAL. CESAR DIAZ 3050, Ciudad de Buenos Aires , Argentina':[-34.6154251022417, -58.48088467805227],
               'JOSE ZUBIAR 4189, Ciudad de Buenos Aires , Argentina':[-34.67173995263086, -58.45714586455621],
               ##la direccion real es 23 de junio y pascual perez, pero en el df original está asi
               'GRAL. JOSE GERVASIO DE ARTIGAS 878, Ciudad de Buenos Aires , Argentina':[-34.620613707419025, -58.4679637915443],
               'AVDA. INTENDENTE CANTILO Y LA PAMPA ALT. 99 , Ciudad de Buenos Aires , Argentina':[-34.550722561616595, -58.429474505041114],
               'CNEL. APOLINARIO FIGUEROA 1077, Ciudad de Buenos Aires , Argentina':[-34.60900630575182, -58.452953220380984],
               'PTE. LUIS SAENZ PEÑA 1215, Ciudad de Buenos Aires , Argentina':[-34.62263283825489, -58.38733508408148],
               'TTE. GRAL. JUAN DOMINGO PERON 1140, Ciudad de Buenos Aires , Argentina':[-34.60625161359731, -58.382609076201646],
               'AVDA. ESCALADA S/N E/AVDA. GRAL. FERNANDEZ DE LA CRUZ Y VIAS DEL FFCC BELGRANO SUR , Ciudad de Buenos Aires , Argentina':[-34.67187935035184, -58.461509412237575],
               'CARLOS H. PERETTE Y CALLE 10 , Ciudad de Buenos Aires , Argentina':[-34.58177284319252, -58.38040295370624]
               }

diccionario = {'HUMBERTO 1º 1573, Ciudad de Buenos Aires , Argentina':[-34.62105777913497, -58.38821210318528],
               'CORONEL MARTINIANO CHILAVERT 5460, Ciudad de Buenos Aires , Argentina':[-34.67904025544257, -58.467149062704536],
               'PALOS Y JUAN MANUEL BLANES, Ciudad de Buenos Aires , Argentina':[-34.632700550024396, -58.3642277627076],
               'AVENIDA GENERAL LAS HERAS 4078, Ciudad de Buenos Aires , Argentina':[-34.5817899417371, -58.41622037435185],
               'CORONEL RAMON LISTA 5256, Ciudad de Buenos Aires , Argentina':[-34.61400660950096, -58.52352952838179],
               'CORONEL RAMON FALCON 2934, Ciudad de Buenos Aires , Argentina':[-34.61386154691175, -58.523588178052464],
               'CORONEL RAMON FALCON 2248, Ciudad de Buenos Aires , Argentina':[-34.629294753227924, -58.459864462707834],
               'GENERAL HORNOS 530, Ciudad de Buenos Aires , Argentina':[-34.63374807439701, -58.377102976199886],
               'ALMIRANTE FRANCISCO SEGUI 2580, Ciudad de Buenos Aires , Argentina':[-34.596656971759174, -58.46113334921762],
               'AVENIDA GENERAL IRIARTE 3501 , Ciudad de Buenos Aires , Argentina':[-34.651149819693615, -58.396749837083256],
               'GENERAL MARTIN DE GAINZA 1050, Ciudad de Buenos Aires , Argentina':[-34.6091136408391, -58.44679349868182],
               'CORONEL RAMON FALCON 4151, Ciudad de Buenos Aires , Argentina':[-34.635426865076724, -58.485636862707416],
               'CORONEL RAMON FALCON 6702, Ciudad de Buenos Aires , Argentina':[-34.64020314378938, -58.52147440688657],
               'PASAJE LA CONSTANCIA 2524, Ciudad de Buenos Aires , Argentina':[-34.647468126906894, -58.43216219154246],
               'PASAJE L & LACARRA, Ciudad de Buenos Aires , Argentina':[-34.65858792216857, -58.456352218526405],
               'GENERAL URQUIZA 277, Ciudad de Buenos Aires , Argentina':[-34.613582814804055, -58.410103389693404],
               'CORONEL RAMON FALCON 4126, Ciudad de Buenos Aires , Argentina':[-34.63581977836494, -58.48471490503556],
               'PRESIDENTE CAMILO TORRES Y TENORIO 2147, Ciudad de Buenos Aires , Argentina':[-34.64612146079697, -58.43853780503492],
               'PASAJE LA PORTEÑA 54, Ciudad de Buenos Aires , Argentina':[-34.62932618438913, -58.468228191543666],
               'AVENIDA GENERAL FERNANDEZ DE LA CRUZ 3605, Ciudad de Buenos Aires , Argentina':[-34.66378610950963, -58.449064862705484],
               'HUMBERTO 1º 343, Ciudad de Buenos Aires , Argentina':[-34.62022109903295, -58.37055973387255],
               'PRESIDENTE LUIS SAENZ PEÑA 463, Ciudad de Buenos Aires , Argentina':[-34.614186930974675, -58.38775653640017],
               'GENERAL MANUEL RODRIGUEZ 2332, Ciudad de Buenos Aires , Argentina':[-34.5985894242755, -58.457134605037965],
               'GENERAL LUCIO NORBERTO MANSILLA 3643, Ciudad de Buenos Aires , Argentina':[-34.59051286249745, -58.41487614736672],
               'CORONEL MARTINIANO CHILAVERT 2690, Ciudad de Buenos Aires , Argentina':[-34.65540386512049, -58.440250989690625],
               'AVENIDA PRESIDENTE MANUEL QUINTANA 31, Ciudad de Buenos Aires , Argentina':[-34.59207402674483, -58.38504687805387],
               'GENERAL GREGORIO ARAOZ DE LAMADRID 499, Ciudad de Buenos Aires , Argentina':[-34.63707314410623, -58.35924266085589],
               'AVENIDA CORONEL CARDENAS 2652, Ciudad de Buenos Aires , Argentina':[-34.66623303972807, -58.5022398050336],
               'CORONEL RAMON FALCON 4801, Ciudad de Buenos Aires , Argentina':[-34.63784686400738, -58.49498879154311],
               'GENERAL ENRIQUE MARTINEZ 1432, Ciudad de Buenos Aires , Argentina':[-34.57509895686678, -58.46068836271128],
               'AVENIDA CORONEL ROCA 9, Ciudad de Buenos Aires , Argentina':[-34.698415981561396, -58.47039687434422],
               'CORONEL APOLINARIO FIGUEROA 661, Ciudad de Buenos Aires , Argentina':[-34.606169299115095, -58.44876154736575],
               'AVENIDA GENERAL LAS HERAS 3086, Ciudad de Buenos Aires , Argentina':[-34.58363018134349, -58.4054547203825],
               'MARISCAL ANTONIO JOSE DE SUCRE 1367, Ciudad de Buenos Aires , Argentina':[-34.557530422661195, -58.44467006133722],
               'CORONEL PEDRO CALDERON DE LA BARCA 3073, Ciudad de Buenos Aires , Argentina':[-34.61354566555254, -58.52241267434972],
               'HUMBERTO 1º 3171, Ciudad de Buenos Aires , Argentina':[-34.623391353056235, -58.41015896270817],
               'GENERAL CESAR DIAZ 3050, Ciudad de Buenos Aires , Argentina':[-34.6154251022417, -58.48088467805227],
               '23 DE JUNIO Y PASCUAL PEREZ, Ciudad de Buenos Aires , Argentina':[-34.67173995263086, -58.45714586455621],
               'GENERAL JOSE GERVASIO DE ARTIGAS 878, Ciudad de Buenos Aires , Argentina':[-34.620613707419025, -58.4679637915443],
               'AVENIDA INTENDENTE CANTILO Y LA PAMPA, Ciudad de Buenos Aires , Argentina':[-34.550722561616595, -58.429474505041114],
               'CORONEL APOLINARIO FIGUEROA 1077, Ciudad de Buenos Aires , Argentina':[-34.60900630575182, -58.452953220380984],
               'PRESIDENTE LUIS SAENZ PEÑA 1215, Ciudad de Buenos Aires , Argentina':[-34.62263283825489, -58.38733508408148],
               'TENIENTE GENERAL JUAN DOMINGO PERON 1140, Ciudad de Buenos Aires , Argentina':[-34.60625161359731, -58.382609076201646],
               'AVENIDA ESCALADA Y AVENIDA GENERAL FERNANDEZ, Ciudad de Buenos Aires , Argentina':[-34.67187935035184, -58.461509412237575],
               'CARLOS H. PERETTE Y CALLE ISLAS GALAPAGOS, Ciudad de Buenos Aires , Argentina':[-34.58177284319252, -58.38040295370624]
               }

for direccion, coordenadas in diccionario.items():
    direcciones.loc[direcciones['Direccion2'] == direccion, ['latitud', 'longitud']] = coordenadas

matricula = matricula.merge(direcciones[['Direccion2', 'latitud', 'longitud']], 
                                            on='Direccion2', 
                                            how='left')
matricula = matricula.sort_values(by='latitud', na_position='first')
'''

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
















#############################################################################
#############################################################################
#############################################################################
#############################################################################
###############    PROCESAMIENTO DE MATRICULA 22-24       ###################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

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

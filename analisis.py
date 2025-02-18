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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV



#LEVANTO LA MUESTRA
apoyos = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.apoyos_bbdd.csv', low_memory=False)
servicios = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ft_servicios_documentos.csv', low_memory=False)
matricula = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_documento.csv', low_memory=False)
notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_bbdd.csv', low_memory=False)
pps = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps_documento.csv', low_memory=False)
responsables = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.responsables_personas.csv', low_memory = False)
dse = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dse_personas.csv', low_memory = False)
dd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dd_personas.csv', low_memory = False)
ds = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ds_personas.csv', low_memory = False)
pases = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pases.csv', low_memory = False)
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
###################    PROCESAMIENTO DE APOYOS       ########################
#############################################################################

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


#############################################################################
###################   PROCESAMIENTO DE SERVICIOS  ###########################
#############################################################################

#no hace falta procesamiento

#############################################################################
###################   PROCESAMIENTO DE MATRICULA  ###########################
#############################################################################

matricula['repite'].sum()

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
'''
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
'''
for direccion, coordenadas in diccionario.items():
    direcciones.loc[direcciones['Direccion2'] == direccion, ['latitud', 'longitud']] = coordenadas

matricula = matricula.merge(direcciones[['Direccion2', 'latitud', 'longitud']], 
                                            on='Direccion2', 
                                            how='left')
matricula = matricula.sort_values(by='latitud', na_position='first')

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

#añado info de cambio de escuela
#matricula_p['24_mantiene_cue'] = (matricula_p['23_cueanexo'] == matricula_p['24_cueanexo']).astype(int)


matricula_p.to_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv',index=False)
matricula_p = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_pivotada.csv', low_memory=False)

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
print(f"AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")




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
             'trayectoria_contenidos','trayectoria_interes','trayectoria_contenidos','trayectoria_ajustesAreas','trayectoria_cualesAjustes',
             'vinculo_observaciones','antecedentes_antecedentes','antecedentes_poseeCertificado','antecedentes_informe.url',
             'antecedentes_informe.filename','intervenciones_informe.url','intervenciones_informe.filename',
             'actitud_como','actitud_pedagogica','trayectoria_ajustesRazonables','jornada_cual','jornada_observaciones',
             'trayectoria_cuales','trayectoria_observaciones','intervenciones_derivacion','convivencia_resuelve',
             'convivencia_vinculapares','actitud_trabaja','actitud_autonomo','actitud_participa','trayectoria_requirioadecuaciones']

pps = pps.drop(columns=col_texto)


valores_unicos_dict = {col: pps[col].unique().tolist() for col in pps.columns}

#veo que hay cols con '' o nulos
conteo_nulos = pps.isna().sum()  # Cuenta NaNs
conteo_vacios = (pps == '').sum()  # Cuenta valores vacíos
conteo_total = conteo_nulos + conteo_vacios  # Suma ambos conteos


#adaptacion a valores aptos
cols_a_convertir = ['trayectoria_requirio', 'jornada_participa', 'trayectoria_requirioacompañada','trayectoria_requiriopedagogico'] 
pps[cols_a_convertir] = pps[cols_a_convertir].replace({'Si': 1, 'Sí': 1, 'No': 0, '':-1, None: -1, np.nan: -1})

cols_frec = ['actitud_demuestra', 'actitud_logra', 'actitud_consulta','actitud_cumple',
             'actitud_manifiesta','actitud_puedeOrganizarse','convivencia_acude','convivencia_mantiene',
             'convivencia_respeta','convivencia_vincula','vinculo_acompaña','vinculo_participa'] 
pps[cols_frec] = pps[cols_frec].replace({'Con poca frecuencia': 0, 'Frecuentemente': 1, 'Siempre':2, '':-1, None: -1, np.nan: -1})

valores_unicos_check = {col: pps[col].unique().tolist() for col in pps.columns}


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
#model2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
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
shap.dependence_plot("actitud_cumple", shap_values_m2.values, X_test_m2)
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

#analisis grafico
explainer_2rs = shap.Explainer(best_model_2rs, X_train_m2)
shap_values_2rs = explainer_2rs(X_test_m2)
shap.summary_plot(shap_values_2rs, X_test_m2)
shap.summary_plot(shap_values_2rs, X_test_m2, plot_type="bar")
#veo con que variables interactúan las top5 features // 23_capacidad_maxima
# 22_capacidad_maxima y 23_distrito_escolar_6
shap.dependence_plot("actitud_cumple", shap_values_2rs.values, X_test_m2)
shap.dependence_plot("actitud_consulta", shap_values_2rs.values, X_test_m2)
shap.dependence_plot("actitud_manifiesta", shap_values_2rs.values, X_test_m2)
shap.dependence_plot("actitud_puedeOrganizarse", shap_values_2rs.values, X_test_m2)
shap.dependence_plot("22_turno_Doble", shap_values_2rs.values, X_test_m2)




#############################################################################
###################   PROCESAMIENTO DE PASES  ###########################
#############################################################################

pases_p = pases[['anio_pase','documento', 'cue_destino']].copy()
# Agregar la columna 'pase' con valor 1
pases_p['pase'] = 1
# Reorganizar el orden de las columnas si es necesario
pases_p = pases_p[['anio_pase','documento', 'pase', 'cue_destino']]

#conteo pases
pases_p = pases_p.groupby(['documento', 'anio_pase'])['cue_destino'].nunique().reset_index()
pases_p.rename(columns={'cue_destino': 'cant_pases'}, inplace=True)

pases_p['anio_pase'] = pases_p['anio_pase'].astype(str)
pases_p['anio_prefijo'] = pases_p['anio_pase'].str[-2:]
pases_p = pases_p.pivot(index='documento', columns='anio_prefijo')

pases_p.columns = [f"{col[1]}_{col[0]}" for col in pases_p.columns]
pases_p.reset_index(inplace=True)
pases_p = pases_p.drop(columns=['22_anio_pase','23_anio_pase','24_anio_pase'])
pases_p[['22_cant_pases', '23_cant_pases', '24_cant_pases']] = pases_p[['22_cant_pases', '23_cant_pases', '24_cant_pases']].fillna(0)


####################################
###### MODELO 3 - MATRICULA, PPS y PASES
####################################

matricula_m3 = matricula_m2.copy()
matricula_m3 = matricula_m3.merge(pases_p, on="documento", how="inner")


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
print(f"AUC sobre el train: {accuracy_m3:.4f}")
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
    #'scale_pos_weight': [1, 2, 3, 5],  # Ajuste del peso para la clase minoritaria
    'gamma': [0, 1, 3, 5],  # Regularización para evitar sobreajuste
    'max_delta_step': [0, 1, 5],  # Paso máximo para mejorar la estabilidad
    'min_child_weight': [1, 5, 10],  # Peso mínimo de las instancias en una hoja
    }

model_rs3 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight= len(y_train_m2[y_train_m2 == 0]) / len(y_train_m2[y_train_m2 == 1])  # Incluir scale_pos_weight aquí
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
print("Mejor AUC del train:", random_search3.best_score_)
print(f"AUC ROC sobre train: {roc_auc_score(y_train_m3, random_search3.predict_proba(X_train_m3)[:, 1]):.4f}")
best_model_3rs = random_search3.best_estimator_

# Predicción
y_pred_m3 = best_model_3rs.predict(X_test_m3)

# Métricas
print(classification_report(y_test_m3, y_pred_m3))
print(f"AUC: {roc_auc_score(y_test_m3, best_model_3rs.predict_proba(X_test_m3)[:, 1]):.4f}")

#analisis grafico
explainer_3rs = shap.Explainer(best_model_3rs, X_train_m3)
shap_values_3rs = explainer_3rs(X_test_m3)
shap.summary_plot(shap_values_3rs, X_test_m3)
shap.summary_plot(shap_values_3rs, X_test_m3, plot_type="bar")

tn, fp, fn, tp = confusion_matrix(y_test_m3, y_pred_m3).ravel()
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN): {fn}")


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


#############################################################################
################    PROCESAMIENTO DE RESPONSABLES       #####################
#############################################################################

responsables['nac_resp'].unique()
responsables['nivel_educativo'].unique()

#nivel educativo
nivel_educativo = {'sin estudios':0, 'primario incompleto':1,
                   'primario completo':2, 'secundario incompleto':3,
                   'secundario completo':4, 'terciario incompleto':5, 
                   'terciario completo':6, 'universitario incompleto':7,
                   'universitario completo':8,'posgrado':9}
responsables['nivel_educativo'] = responsables['nivel_educativo'].str.strip().str.lower().replace(nivel_educativo)

#nacionalidad de los responsables
responsables['nac_resp'] = (responsables['nac_resp'].str.lower().apply(unidecode).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x)))
responsables = pd.get_dummies(responsables, columns=['nac_resp'], prefix='nac_resp_')
responsables[responsables.filter(regex='^nac_resp_').columns] = responsables.filter(regex='^nac_resp_').astype(int)

#vinculo de los responsables
responsables['vinculo'] = (responsables['vinculo'].str.lower().apply(unidecode).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x)))
responsables = pd.get_dummies(responsables, columns=['vinculo'], prefix='vinculo_')
responsables[responsables.filter(regex='^vinculo_').columns] = responsables.filter(regex='^vinculo_').astype(int)


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










'''
###########################################################################
                           OHE + PRESENTISMO
###########################################################################
'''


####################################
# FUNCION AUXILIAR PARA LOS CASOS DONDE APLIQUE EL ONE HOT ENCODING
####################################

#devuelve unique-1 cols para cada conjunto de valores de la columna
def one_hot_encode_columns(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)    
    return df_encoded

#######################
## OHE DIRECTO
#######################

#genero del alumno
bbdd_clean['a_genero'].value_counts(normalize=True)
#trabaja
bbdd_clean['ase_trabaja'].value_counts(normalize=True)
#discapacidad
bbdd_clean['a_disc_ninguno'].value_counts(normalize=True)
#tratamiento psicopedagogia
bbdd_clean['a_tratamiento_psicopedagogia'].value_counts(normalize=True)
#turnos y jornadas
bbdd_clean['e22_turno'].value_counts(normalize=True)
bbdd_clean['e22_jornada'].value_counts(normalize=True)
bbdd_clean['e23_turno'].value_counts(normalize=True)
bbdd_clean['e23_jornada'].value_counts(normalize=True)
#dependencias funcionales y distritos
bbdd_clean['e23_dependencia_funcional'].value_counts(normalize=True)
bbdd_clean['e22_distrito_escolar'].value_counts(normalize=True)
bbdd_clean['e23_distrito_escolar'].value_counts(normalize=True)


columns_ohe = ['a_genero','ase_trabaja','a_disc_ninguno','a_tratamiento_psicopedagogia',
               'e22_turno','e22_jornada','e23_turno','e23_jornada','e23_dependencia_funcional',
               'e22_distrito_escolar','e23_distrito_escolar']

bbdd_clean = one_hot_encode_columns(bbdd_clean,columns_ohe)


#######################
## NACIONALIDAD DEL ALUMNO -- OHE
#######################

##['VENEZUELA', 'OTRO', 'BRASIL', 'PERU', 'BOLIVIA', 'Bolivia', nan,
#       'ARGENTINA', 'Argentina', 'COLOMBIA', 'PARAGUAY', 'CHILE',
#       'URUGUAY', 'RUSIA', 'FRANCIA', 'REPÚBLICA DOMINICANA']

bbdd_clean['a_nac_arg'] = bbdd_clean['a_nacionalidad'].str.lower().isin(['argentina']).astype(int)
bbdd_clean['a_nac_hh_america'] = bbdd_clean['a_nacionalidad'].str.lower().isin(['venezuela', 'peru','bolivia','colombia','paraguay','chile','uruguay','república dominicana']).astype(int)
bbdd_clean['a_nac_hnh_america'] = bbdd_clean['a_nacionalidad'].str.lower().isin(['brasil']).astype(int)
bbdd_clean['a_nac_hnh_otros'] = bbdd_clean['a_nacionalidad'].str.lower().isin(['rusia','francia']).astype(int)

# NACIONALIDAD DEL RESPONSABLE
#['Venezuela', 'Brasil', 'Perú', nan, 'Bolivia', 'Otros',
#       'Argentina', 'Paraguay', 'Uruguay', 'Alemania',
#       'China (República Popular de)', 'Chile', 'Taiwan',
#       'República Dominicana', 'España', 'Afghanistán', 'México',
#       'Ecuador', 'Francia', 'Estados Unidos', 'Colombia', 'Ucrania',
#       'Italia', 'Corea del Sur', 'Rusia', 'Cuba'],

bbdd_clean['r_nac_arg'] = bbdd_clean['resp_nac'].str.lower().isin(['argentina']).astype(int)
bbdd_clean['r_nac_hh_america'] = bbdd_clean['resp_nac'].str.lower().isin(['venezuela','perú','bolivia','paraguay','uruguay','chile','república dominicana','méxico','ecuador','colombia','cuba']).astype(int)
bbdd_clean['r_nac_hnh_america'] = bbdd_clean['resp_nac'].str.lower().isin(['brasil','estados unidos']).astype(int)
bbdd_clean['r_nac_hnh_otros'] = bbdd_clean['resp_nac'].str.lower().isin(['alemania','china (república popular de)', 'taiwan','afghanistán','francia','ucrania','italia','corea del sur','rusia']).astype(int)
bbdd_clean['r_hh_otros'] = bbdd_clean['resp_nac'].str.lower().isin(['españa']).astype(int)

bbdd_clean = bbdd_clean.drop(columns=['resp_nac','a_nacionalidad'])

######################
# NIVEL EDUCATIVO RESPONABLES
######################

bbdd_clean['resp_educativo'].value_counts(normalize=True)


######################
# BINNING PARA CAPACIDADES MAXIMAS
######################

sorted(bbdd_clean['e22_capacidad_maxima'].unique())

bins = [0, 10, 15, 20, 25,30,35,40,50,float('inf')] 
labels = ['0-10', '11-15', '16-20', '21-25','26-30','31-35','36-40','41-50','Más de 50']

# Crear la nueva columna con pd.cut
bbdd_clean['e22_capac_binned'] = pd.cut(
    bbdd_clean['e22_capacidad_maxima'],
    bins=bins,
    labels=labels,
    include_lowest=True)

bbdd_clean['e23_capac_binned'] = pd.cut(
    bbdd_clean['e23_capacidad_maxima'],
    bins=bins,
    labels=labels,
    include_lowest=True)

#despues de esto hay que hacer OHE para que nos queden col dummies
bbdd_clean = one_hot_encode_columns(bbdd_clean,['e22_capac_binned'])
bbdd_clean = one_hot_encode_columns(bbdd_clean,['e23_capac_binned'])

#######################
## PASE A NUMEROS DE LAS CONCEPTUALES y OHE
#######################

bbdd_clean['a_n1_mate_p'].value_counts(normalize=True)
bbdd_clean['a_n2_mate_p'].value_counts(normalize=True)
bbdd_clean['a_n3_mate_p'].value_counts(normalize=True)
bbdd_clean['a_n4_mate_p'].value_counts(normalize=True)
bbdd_clean['a_n1_lengua_p'].value_counts(normalize=True)
bbdd_clean['a_n2_lengua_p'].value_counts(normalize=True)
bbdd_clean['a_n3_lengua_p'].value_counts(normalize=True)
bbdd_clean['a_n4_lengua_p'].value_counts(normalize=True)
bbdd_clean['a_n1_mate_s'].value_counts(normalize=True)
bbdd_clean['a_n2_mate_s'].value_counts(normalize=True)
bbdd_clean['a_n3_mate_s'].value_counts(normalize=True)
bbdd_clean['a_n4_mate_s'].value_counts(normalize=True)
bbdd_clean['a_n1_lengua_s'].value_counts(normalize=True)
bbdd_clean['a_n2_lengua_s'].value_counts(normalize=True)
bbdd_clean['a_n3_lengua_s'].value_counts(normalize=True)
bbdd_clean['a_n4_lengua_s'].value_counts(normalize=True)

##tengo que ver como computar las de primario que tienen concep y num
##las de secu están todas en concept

#######################
##    PPS Y APOYOS
#######################

# OHE para las columnas de frecuencia

columns_q = ['actitud_logra','actitud_cumple','actitud_consulta','actitud_demuestra',
           'actitud_manifiesta','actitud_puedeOrganizarse',
           'actitud_trabaja','actitud_autonomo','actitud_participa',
           'actitud_pedagogica','convivencia_acude','convivencia_respeta','convivencia_vincula',
           'convivencia_mantiene','convivencia_resuelve','convivencia_vinculapares',
           'trayectoria_ajustesRazonables','trayectoria_requiriopedagogico',
           'trayectoria_requirioacompañada','trayectoria_requirioadecuaciones',
           'vinculo_acompaña','vinculo_participa']


unique_values_set = set()

for col in columns_q:
    if col in bbdd_clean.columns: 
        unique_values_set.update(bbdd_clean[col].dropna().unique()) 
        

value_map = {
    "Sí": 1,
    "No": 0,
    "Con poca frecuencia": 0,
    "Frecuentemente": 1,
    "Siempre": 2
}

for col in columns_q:
    if col in bbdd_clean.columns:
        bbdd_clean[col] = bbdd_clean[col].map(value_map)


#######################
## Columnas a analizar x separado
#######################

vinculo_counts = bbdd_clean['vinculo_adulto'].str.lower().value_counts(dropna=False).sort_index().reset_index()

'''
MAPEO TENTATIVO A PARTIR DE LOS VALORES DE VINCULO -- MAS FACIL OHE CREO
nadie = ['nadie','docentes no conocen a los padres']
madre = ['mamá','mama','madre','masdre','made','progenitora','progenitores','ambos','padres']
padre = ['papá','papa','padre','progenitor','progenitores','ambos','padres']
hermana = ['hermana','hermanas','hermanos']
hermano = ['hermano','hermanos']
cuniados = ['cuñada','cuñado','cuñados','cuñadas']
abuela = ['abuela','abuelas','abuelos']
abuelo = ['abuelo','abuelos']
tios = ['tía','tia','tío','tíos','tías']
padrino_madrina = ['padrinos','madrina','padrino']
pareja_progenitores = ['madrastra','padrastro','pareja de la madre','pareja del padre','mujer del padre','esposo de la madre']
docentes = ['docentes','maestra','maestro','maestra integradora','maestro integrador']
hogar = ['operador del hogar','operadora del hogar','operadores del hogar',
         'cat n°','director del hogar','directora del hogar','hogar',
         'equipo técnico del hogar','referentes del hogar']
tutores = ['tutora legal','tutora','tutor legal','tutor','tutores','tutoras',
           'representante legal','representante']
'''

bbdd_clean['trayectoria_requirio'].value_counts(dropna=False).sort_index().reset_index()
bbdd_clean['trayectoria_requirio'] = bbdd_clean['trayectoria_requirio'].str.lower().map({'sí': 1, 'no': 0}).fillna(0)

bbdd_clean['trayectoria_cuales'].str.lower().value_counts(dropna=False).sort_index().reset_index()

def extract_unique_labels(data_column):
    unique_labels = set()
    for item in data_column:
        if isinstance(item, str): 
            try:
                item = ast.literal_eval(item) 
            except (ValueError, SyntaxError):
                continue
        if isinstance(item, list): 
            for d in item:
                if isinstance(d, dict) and 'label' in d: 
                    unique_labels.add(d['label'])
    return unique_labels

unique_labels = extract_unique_labels(bbdd_clean['trayectoria_cuales'])

for label in unique_labels:
    # Crear una nueva columna para cada etiqueta, asignando 1 si está presente, 0 si no lo está
    bbdd_clean[f'trayectoria_cuales_{label}'] = bbdd_clean['trayectoria_cuales'].apply(
        lambda x: 1 if isinstance(x, list) and any(d.get('label') == label for d in x) else 0
    )


#######################
## SISTEMA DE SALUD
#######################

bbdd_clean['a_sistema_salud'].unique()
#['Hospital público', nan, '-1', 'Obra social', 'Pre-paga']
bbdd_clean['a_sistema_salud'].value_counts(dropna=False)
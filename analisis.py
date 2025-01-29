# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:16:33 2024

@author: guima
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from datetime import datetime
import ast
import requests
import geopy
from geopy.geocoders import Nominatim



#LEVANTO LA MUESTRA
apoyos = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.apoyos_bbdd.csv', low_memory=False)
#servicios = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ft_servicios.csv', low_memory=False)
servicios = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ft_servicios_documentos.csv', low_memory=False)
#matricula2 = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula.csv', low_memory=False)
matricula = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.matricula_documento.csv', low_memory=False)
notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.notas_bbdd.csv', low_memory=False)
#pps2 = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps.csv', low_memory=False)
pps = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pps_documento.csv', low_memory=False)
responsables = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.responsables_personas.csv', low_memory = False)
#socioeco = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.socioeco.csv', low_memory = False)
#socioeco = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.socioeco_documentos.csv', low_memory = False)
dse = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dse_personas.csv', low_memory = False)
dd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.dd_personas.csv', low_memory = False)
ds = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.ds_personas.csv', low_memory = False)
pases = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/0.pases.csv', low_memory = False)
localidades = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.localidades.csv', low_memory = False, delimiter = ';')
provincias = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/1.provincias.csv', low_memory = False, delimiter = ';')



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
#no necesario procesar x el moemnto


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
'''
-
Bueno (B)
Regular (R)
Muy Bueno (MB)
Sobresaliente (S)
Promoción acompañada
Insuficiente (I)
Suficiente
Avanzado
En Proceso
no corresponde
'''

#############################################################################
#####################    PROCESAMIENTO DE PPS       ##########################
#############################################################################

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
             'antecedentes_informe.filename','intervenciones_informe.url','intervenciones_informe.filename']
pps = pps.drop(columns=col_texto)

pps['documento'].nunique()

#############################################################################
################    PROCESAMIENTO DE RESPONSABLES       #####################
#############################################################################



#############################################################################
###################   PROCESAMIENTO DE DATOS PERSONALES  ########################
#############################################################################

#############                  DIM DOMICILIO                  ############

dd = dd.sort_values(by=['documento', 'domicilio_renaper', 'mas_reciente'], ascending=[True, False, True])
dd = dd.groupby('documento').first().reset_index()




#############                  DIM SOCIOECONOMICO                  ############

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

#############                  DATOS PERSONALES FULL              ############
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
cps['localidad'] = cps['localidad'].str.replace(r'\s*\(.*?\)\s*', '', regex=True)
cps = cps.sort_values(by='cp', ascending=True)

#hago string este campo y no integer el otro porque corro riesgo de fabricar
#un cp falso a partir de info basura

dd_comp = pd.merge(dd_m,cps,left_on='codigo_postal',right_on='cp',how='left')
dd_comp['ciudad'] = dd_comp['ciudad'].combine_first(dd_comp['localidad'])
dd_comp['provincia'] = dd_comp['provincia'].combine_first(dd_comp['provincia_y'])

cps['cp'].nunique()
codigo_postal_unicos = dd_m['codigo_postal'].unique()
'''




        
##################    PROCESAMIENTO DE SOCIOECONOMICO   #######################

def obtener_coordenadas_osm(direccion):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": direccion,
        "format": "json",
        "addressdetails": 1
    }
    headers = {"User-Agent": "TuAplicacion/1.0"}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            print("No se encontraron resultados.")
    else:
        print("Error en la solicitud:", response.status_code)
    return None

# Ejemplo de uso
direccion = "Calle Falsa 123, Springfield, IL, USA"
coordenadas = obtener_coordenadas_osm(direccion)
print("Coordenadas:", coordenadas)


#-----------------------------------------------------------------------------------------------

# PASO LA COLUMNA DE ASE_UPDATED_AT A FECHA PARA PODER VER EL MINIMO QUE NECESITO
# DE REF PARA LA TIRA DE VALORES DEL BLUE

bbdd['ase_updated_at'] = pd.to_datetime(bbdd['ase_updated_at'], format='%Y-%m-%d', errors='coerce')
bbdd['ase_updated_at'].min()


### JOINEO TODO LO QUE ESTUVE LIMPIANDO

bbdd_clean = bbdd.groupby('id_alumno').apply(lambda group: group.ffill().bfill()).drop_duplicates(subset='id_alumno')
bbdd_clean = pd.merge(bbdd_clean, apoyos_merge, left_on='id_miescuela', right_on='id_alumno', how='left')
bbdd_clean = bbdd_clean.drop(columns=['id_alumno_y']).rename(columns={'id_alumno_x': 'id_alumno'})


## TENGO VARIOS VALORES DE A_DOMICILIO_PROVINCIA, ME QUEDO CON CABA, PBA Y OTROS

bbdd_clean['a_domicilio_grupo'] = bbdd_clean['a_domicilio_provincia'].apply(
    lambda x: 'CABA' if x in ['CIUDAD AUTONOMA DE BS AS', 'CIUDAD DE BUENOS AIRES', 'Ciudad Autónoma de Buenos Aires']
    else 'PBA' if x in ['BUENOS AIRES', 'Buenos Aires']
    else 'OTROS'
)

## ME QUEDO CON LOS RESPONSABLES DE IELY NO LOS DE LA BBDD PORQUE TENGO MENOS NULOS

#borro las columnas de los responsables

'''
columnas_a_eliminar = [
    'm_responsable_principal', 'm_genero', 'm_nacionalidad', 'm_nivel_estudios', 'm_sistema_salud', 
    'm_disc_ninguno', 'm_trabaja', 'm_sueldo', 'm_pensionado', 'm_subsidios', 'm_ase_updated_at',
    'p_responsable_principal', 'p_genero', 'p_nacionalidad', 'p_nivel_estudios', 'p_sistema_salud', 
    'p_disc_ninguno', 'p_trabaja', 'p_sueldo', 'p_pensionado', 'p_subsidios', 'p_ase_updated_at',
    'o_responsable_principal', 'o_vinculo', 'o_genero', 'o_nacionalidad', 'o_nivel_estudios', 
    'o_sistema_salud', 'o_disc_ninguno', 'o_trabaja', 'o_sueldo', 'o_pensionado', 'o_subsidios', 
    'o_tiene_hijos', 'o_ase_updated_at']

# Elimina las columnas del DataFrame
bbdd_clean = bbdd_clean.drop(columns=columnas_a_eliminar)
'''

#veo cuantos nulos tengo
bbdd_clean['resp_nac'].isnull().sum()
bbdd_clean['resp_vinculo'].isnull().sum()
bbdd_clean['resp_educativo'].isnull().sum()

bbdd_clean['resp_madre'] = np.where(bbdd_clean['resp_vinculo'].str.lower() == 'madre', 1, 0)
bbdd_clean['resp_padre'] = np.where(bbdd_clean['resp_vinculo'].str.lower() == 'padre', 1, 0)
bbdd_clean['resp_otros'] = np.where(
    (bbdd_clean['resp_vinculo'].str.lower() != 'madre') & (bbdd_clean['resp_vinculo'].str.lower() != 'padre'), 1, 0
)


##########
## ME BAJO EL CLEAN PARA NO TENER QUE CORRER ESTO TODO EL TIEMPO
##########

bbdd_clean.to_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/bbdd_clean.csv')


'''
###########################################################################
                    ANALISIS PARA EL CHECKPOINT 3
###########################################################################
'''


###########################################################################
#                    ANALISIS DE VALORES NULOS POR COLUMNA
###########################################################################


nulls_col = bbdd_clean.isnull().sum()
nulls_perc = (bbdd_clean.isnull().mean() * 100).round(2)
dtypes = bbdd_clean.dtypes

analisis_nulos = pd.DataFrame({
    'columna': nulls_col.index,
    'cantidad_nulos': nulls_col.values,
    'porcentaje_nulos': nulls_perc.values,
    'dtype': dtypes.values
})

analisis_nulos = analisis_nulos.sort_values(by='porcentaje_nulos', ascending=False)
analisis_nulos.reset_index(drop=True, inplace=True)

analisis_nulos_f = analisis_nulos[analisis_nulos['porcentaje_nulos'] > 10]
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=analisis_nulos_f.values, 
                 colLabels=analisis_nulos_f.columns, 
                 cellLoc = 'center', 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  
plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/analisis_nulos_10perc.jpg', bbox_inches='tight', dpi=300)
plt.close()

# ###########################################################################
#               ANALISIS DISTRIBUCION DE LOS DATOS
########################################################################### 

###############
#   Casos de repitencia
###############

bbdd_clean['repite_23_24'].sum()

###############
#   Distribución de matricula en las comunas
###############

mat_comuna = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/matriculados_comuna.csv', sep=",", header=0, names=['comuna', 'alumnos_comuna'])
comunas_geo = gpd.read_file('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/comunas.geojson.txt')  # Asegúrate de usar la ruta correcta
comunas_geo = comunas_geo.merge(mat_comuna, on='comuna', how='left')

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
comunas_geo.plot(column='alumnos_comuna', ax=ax, legend=True,
                 cmap='Blues',
                 edgecolor='grey',  # Color del borde de las comunas
                 missing_kwds={
                     'color': 'lightgrey', 
                     'label': 'No data',
                     'hatch': '///'
                 })

for x, y, label in zip(comunas_geo.geometry.centroid.x, comunas_geo.geometry.centroid.y, comunas_geo['comuna']):
    ax.text(x, y, str(label), fontsize=10, ha='center', va='center', color='black')

ax.set_title('Matriculados de 7° Grado por Comuna', fontsize=15)
ax.axis('off')
ax.set_xticks([]) 
ax.set_yticks([])

plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/matricula_comuna.jpg', dpi=300, bbox_inches='tight')

plt.show()



###############
#   Distribución de casos en las comunas
###############

conteo_comunas = bbdd_clean.groupby('e22_comuna').size().reset_index(name='cantidad_registros')
conteo_comunas.rename(columns={'e22_comuna': 'comuna'}, inplace=True)
comunas_geo = gpd.read_file('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/comunas.geojson.txt')  # Asegúrate de usar la ruta correcta
comunas_geo = comunas_geo.merge(conteo_comunas, on='comuna', how='left')

mat_comuna = mat_comuna.merge(conteo_comunas, on='comuna', how='left')


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
comunas_geo.plot(column='cantidad_registros', ax=ax, legend=True,
                 cmap='Blues',
                 edgecolor='grey',  # Color del borde de las comunas
                 missing_kwds={
                     'color': 'lightgrey', 
                     'label': 'No data',
                     'hatch': '///'
                 })

for x, y, label in zip(comunas_geo.geometry.centroid.x, comunas_geo.geometry.centroid.y, comunas_geo['comuna']):
    ax.text(x, y, str(label), fontsize=10, ha='center', va='center', color='black')

ax.set_title('Casos por Comuna', fontsize=15)
ax.axis('off')
ax.set_xticks([]) 
ax.set_yticks([])

plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/casos_comuna.jpg', dpi=300, bbox_inches='tight')

plt.show()


###############
#   Distribución de repitencias en las comunas
###############

bbdd_filtered = bbdd_clean[bbdd_clean['repite_23_24'] == 1]
conteo_comunas = bbdd_filtered.groupby('e22_comuna').size().reset_index(name='cantidad_repitencias')
conteo_comunas.rename(columns={'e22_comuna': 'comuna'}, inplace=True)
comunas_geo = gpd.read_file('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/comunas.geojson.txt')  # Asegúrate de usar la ruta correcta
comunas_geo = comunas_geo.merge(conteo_comunas, on='comuna', how='left')

mat_comuna = mat_comuna.merge(conteo_comunas, on='comuna', how='left')


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
comunas_geo.plot(column='cantidad_repitencias', ax=ax, legend=True,
                 cmap='Reds', 
                 edgecolor='grey', 
                 missing_kwds={
                     'color': 'lightgrey', 
                     'label': 'No data',
                     'hatch': '///'
                 })

for x, y, label in zip(comunas_geo.geometry.centroid.x, comunas_geo.geometry.centroid.y, comunas_geo['comuna']):
    ax.text(x, y, str(label), fontsize=10, ha='center', va='center', color='grey')

ax.set_title('Repitencia por Comuna', fontsize=15)

ax.axis('off') 
ax.set_xticks([])
ax.set_yticks([]) 

plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/repitencias_comuna.jpg', dpi=300, bbox_inches='tight')

plt.show()




'''
###########################################################################
                           OHE + PRESENTISMO
###########################################################################
'''

bbdd_clean = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/bbdd_clean.csv', low_memory=False)
back_up = bbdd_clean.copy()

#borro columnas que no necesito para el analisis

columnas_a_eliminar = ['a_nacimiento', 'a_domicilio_provincia','e22_dependencia_funcional',
                       'e22_modalidad','e23_anio','e23_modalidad','e24_anio','ag_apoyo','e22_comuna',
                       'e23_comuna','e24_turno','e24_jornada','e24_capacidad_maxima','e24_cueanexo',
                       'e24_dependencia_funcional','e24_modalidad','e24_distrito_escolar','e24_comuna',
                       'actitud_como','trayectoria_observaciones','trayectoria_interrumpida','intervenciones_derivacion',
                       'jornada_cual','jornada_participa','jornada_observaciones','ag_apoyo','resp_vinculo']

#trayectoria_interrumpida la bajo porque lo podemos ver desde la regularidad
#en los distintos bimestres y de esa manera tiene un criterio uniforme para
#todos los alumnos, en cambio en PPS la interrupción de la trayectoria no está delimitada
#por un marco legal

#ag_apoyo la vuelo porque ya queda implicita en las demás ag_at, si todas dicen 0
#es porque el chico no necesitó apoyos en su trayectoria

bbdd_clean = bbdd_clean.drop(columns=columnas_a_eliminar)


## Analisis para ver que hacer
columnas = list(sorted(bbdd_clean.columns))

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


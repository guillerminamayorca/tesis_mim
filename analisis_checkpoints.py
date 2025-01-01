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

#LEVANTO LA MUESTRA

datos_upd = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/bbdd.csv', low_memory=False)
notas = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/notas_bbdd.csv', low_memory=False)
apoyos = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/apoyos_bbdd.csv', low_memory=False)
resp_iel = pd.read_csv('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/responsables_2.csv', low_memory = False)

## MERGE PARA TENER NOTAS Y DATOS JUNTOS

bbdd = pd.merge(datos_upd, notas, left_on='id_miescuela', right_on='id_alumno', how='left')
bbdd = pd.merge(bbdd, resp_iel, on='id_miescuela', how='left')
bbdd = bbdd.drop(columns=['id_alumno_y','doc_alu']).rename(columns={'id_alumno_x': 'id_alumno','vinculo_y':'resp_vinculo',
                                                                    'nac_resp':'resp_nac','nivel_educativo':'resp_educativo',
                                                                    'vinculo_x':'vinculo'})


#ROMPO LOS JSON CORRESPONDIENTES A LAS COLUMNAS DEL PPS

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
        json_data = pd.json_normalize(bbdd['actitud'].dropna().apply(json.loads))
        json_data.columns = [f"actitud_{col}" for col in json_data.columns]
        bbdd = bbdd.join(json_data).drop(columns=['actitud'])
    else:
        bbdd = expandir_columna_json(bbdd, col)

# PASO LA COLUMNA DE ASE_UPDATED_AT A FECHA PARA PODER VER EL MINIMO QUE NECESITO
# DE REF PARA LA TIRA DE VALORES DEL BLUE

bbdd['ase_updated_at'] = pd.to_datetime(bbdd['ase_updated_at'], format='%Y-%m-%d', errors='coerce')
bbdd['ase_updated_at'].min()

## TRABAJO CON LA BASE DE APOYOS PARA PODER DEJAR COLS BINARIAS AGRUPADAS X EL CL

apoyos_consolidado = apoyos.groupby('id_alumno').agg(
    ag_apoyo=('ag_apoyo', lambda x: 'Sí' if 'Sí' in x.values else 'No'),
    ag_apoyo_tipo=('ag_apoyo_tipo', lambda x: list({item for sublist in x.dropna().apply(eval) for item in sublist})),
).reset_index()

unique_values = set(val for sublist in apoyos_consolidado['ag_apoyo_tipo'] for val in sublist)

for value in unique_values:
    apoyos_consolidado[f'ag_at_{value}'] = apoyos_consolidado['ag_apoyo_tipo'].apply(lambda x: 1 if value in x else 0)

apoyos_merge = apoyos_consolidado.drop(columns=['ag_apoyo_tipo'])

### JOINEO TODO LO QUE ESTUVE LIMPIANDO

bbdd_clean = bbdd.groupby('id_alumno').apply(lambda group: group.ffill().bfill()).drop_duplicates(subset='id_alumno')
bbdd_clean = pd.merge(bbdd_clean, apoyos_merge, left_on='id_miescuela', right_on='id_alumno', how='left')
bbdd_clean = bbdd_clean.drop(columns=['id_alumno_y']).rename(columns={'id_alumno_x': 'id_alumno'})

## COLUMNAS QUE NO ME INTERESAA QUEDARME

bbdd_clean = bbdd_clean.drop(columns=['antecedentes_informe.url','antecedentes_informe.filename','intervenciones_informe.url','intervenciones_informe.filename'])
columnas = list(sorted(bbdd_clean.columns))
aux_texto = bbdd_clean[['id_alumno','id_miescuela','actitud_observaciones','convivencia_observaciones','trayectoria_destaca','trayectoria_interes',
                        'trayectoria_contenidos','trayectoria_interes','trayectoria_contenidos','trayectoria_ajustesAreas','trayectoria_cualesAjustes',
                        'vinculo_observaciones','antecedentes_antecedentes','antecedentes_poseeCertificado']]
bbdd_clean = bbdd_clean.drop(columns=['id_alumno','id_miescuela','actitud_observaciones','convivencia_observaciones','trayectoria_destaca','trayectoria_interes',
                        'trayectoria_contenidos','trayectoria_interes','trayectoria_contenidos','trayectoria_ajustesAreas','trayectoria_cualesAjustes',
                        'vinculo_observaciones','antecedentes_antecedentes','antecedentes_poseeCertificado'])

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
# MODELADO SOBREEDAD
##########

bbdd_clean['a_nacimiento'].dtype
#esta en formato objeto, la tenemos que pasar a datetime para hacer el calculo
#de la edad del estudiante

fecha_referencia = datetime(2022, 6, 30)
bbdd_clean['a_nacimiento'] = pd.to_datetime(bbdd_clean['a_nacimiento'], errors='coerce')
bbdd_clean['edad'] = bbdd_clean['a_nacimiento'].apply(lambda x: fecha_referencia.year - x.year - ((fecha_referencia.month, fecha_referencia.day) < (x.month, x.day)))

# Crear la columna 'sobreedad'
bbdd_clean['sobreedad'] = bbdd_clean['edad'].apply(lambda x: 1 if x > 12 else 0)


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


###############
#   Rsumen distribución casos
###############


# Calcular el total de cada columna
total_alumnos = mat_comuna['alumnos_comuna'].sum()
total_registros = mat_comuna['cantidad_registros'].sum()
total_repitencias = mat_comuna['cantidad_repitencias'].sum()

# Crear nuevas columnas de porcentaje
mat_comuna['alumnos_comuna_pct'] = round((mat_comuna['alumnos_comuna'] / total_alumnos) * 100,2)
mat_comuna['cantidad_registros_pct'] = round((mat_comuna['cantidad_registros'] / total_registros) * 100,2)
mat_comuna['cantidad_repitencias_pct'] = round((mat_comuna['cantidad_repitencias'] / total_repitencias) * 100,2)


mat_comuna = mat_comuna.sort_values(by='comuna', ascending=True)
mat_comuna.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=mat_comuna.values, 
                 colLabels=mat_comuna.columns, 
                 cellLoc = 'center', 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.2) 
plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/distrib_casos_comunas.jpg', bbox_inches='tight', dpi=300)
plt.close()

###############
#   Distribución de casos por genero del alumno
###############

conteo_genero = bbdd_clean.groupby('a_genero').agg(
    cantidad=('a_genero', 'size'),
    cantidad_repite=('repite_23_24', lambda x: (x == 1).sum())
).reset_index()

conteo_genero_melted = conteo_genero.melt(id_vars='a_genero', 
                                           value_vars=['cantidad', 'cantidad_repite'],
                                           var_name='tipo',
                                           value_name='cantidad')

total_cantidad = conteo_genero['cantidad'].sum()
total_repite = conteo_genero['cantidad_repite'].sum()


plt.figure(figsize=(10, 6))  # Tamaño ajustado

# Definir el ancho de las barras
width = 0.4

# Definir posiciones de las barras
x = np.arange(len(conteo_genero['a_genero']))  # posiciones en el eje x

# Gráfico para 'cantidad'
bars1 = plt.bar(x - width/2, conteo_genero['cantidad'], width=width, color='lightblue', label='Cantidad')

# Gráfico para 'cantidad_repite' en el mismo eje y
bars2 = plt.bar(x + width/2, conteo_genero['cantidad_repite'], width=width, color='lightcoral', label='Cantidad Repite')

# Configurar títulos y etiquetas
plt.title('Distribución de Casos por Género', fontsize=15)
plt.xticks(ticks=x, labels=conteo_genero['a_genero'], rotation=0)  # Etiquetas del eje X con los géneros
plt.xlabel('Género', fontsize=12)

# Aumentar el límite superior del eje Y
plt.ylim(0, conteo_genero['cantidad'].max() * 1.2)

# Ocultar el eje y
plt.gca().axes.get_yaxis().set_visible(False)

# Configurar la leyenda en la esquina superior derecha
plt.legend(loc='upper right', fontsize=10)

# Agregar etiquetas a las barras de cantidad
for bar in bars1:
    yval = bar.get_height()
    percentage = yval / total_cantidad * 100  # Calcular porcentaje sobre el total de 'cantidad'
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontsize=9)

# Agregar etiquetas a las barras de cantidad_repite
for bar in bars2:
    yval = bar.get_height()
    percentage = yval / total_repite * 100  # Calcular porcentaje sobre el total de 'cantidad_repite'
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontsize=9)

# Asegurar que los elementos se ajusten automáticamente
plt.tight_layout()  

plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/distribucion_genero.jpg', dpi=300, bbox_inches='tight')

plt.show()



###############
#   Distribución de casos por domicilio del alumno
###############


conteo_domicilio = bbdd_clean.groupby('a_domicilio_grupo').agg(
    cantidad=('a_domicilio_grupo', 'size'),
    cantidad_repite=('repite_23_24', lambda x: (x == 1).sum())
).reset_index()

conteo_domicilio_melted = conteo_domicilio.melt(id_vars='a_domicilio_grupo', 
                                           value_vars=['cantidad', 'cantidad_repite'],
                                           var_name='tipo',
                                           value_name='cantidad')

total_cantidad = conteo_domicilio['cantidad'].sum()
total_repite = conteo_domicilio['cantidad_repite'].sum()


plt.figure(figsize=(10, 6))  # Tamaño ajustado

# Definir el ancho de las barras
width = 0.4

# Definir posiciones de las barras
x = np.arange(len(conteo_domicilio['a_domicilio_grupo']))  # posiciones en el eje x

# Gráfico para 'cantidad'
bars1 = plt.bar(x - width/2, conteo_domicilio['cantidad'], width=width, color='lightblue', label='Cantidad')

# Gráfico para 'cantidad_repite' en el mismo eje y
bars2 = plt.bar(x + width/2, conteo_domicilio['cantidad_repite'], width=width, color='lightcoral', label='Cantidad Repite')

# Configurar títulos y etiquetas
plt.title('Distribución de Casos por Domicilio', fontsize=15)
plt.xticks(ticks=x, labels=conteo_domicilio['a_domicilio_grupo'], rotation=0)  # Etiquetas del eje X con los géneros
plt.xlabel('Domicilio', fontsize=12)

# Aumentar el límite superior del eje Y
plt.ylim(0, conteo_domicilio['cantidad'].max() * 1.2)

# Ocultar el eje y
plt.gca().axes.get_yaxis().set_visible(False)

# Configurar la leyenda en la esquina superior derecha
plt.legend(loc='upper right', fontsize=10)

# Agregar etiquetas a las barras de cantidad
for bar in bars1:
    yval = bar.get_height()
    percentage = yval / total_cantidad * 100  # Calcular porcentaje sobre el total de 'cantidad'
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontsize=9)

# Agregar etiquetas a las barras de cantidad_repite
for bar in bars2:
    yval = bar.get_height()
    percentage = yval / total_repite * 100  # Calcular porcentaje sobre el total de 'cantidad_repite'
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontsize=9)

# Asegurar que los elementos se ajusten automáticamente
plt.tight_layout()  

plt.savefig('C:/Users/guima/OneDrive - Universidad Torcuato Di Tella/02 MiM/TESIS/edu/codigo_bases/distribucion_domicilio.jpg', dpi=300, bbox_inches='tight')

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


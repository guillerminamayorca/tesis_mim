-- ###################################################################################
-- 						BAJADA DE MATRICULA GENERAL 7° GRADO 2022
-- ###################################################################################

select 
	concat('Comuna', de22.comuna) comuna_texto,
	de22.comuna comuna,
	count(distinct m22.id_miescuela) alumnos_comuna
from educacion_produccion.ft_matricula m22
left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
where true 
	and m22.ciclo_lectivo = 2022
	and s22.anio in ('7° Grado','6° y 7° Grado')
group by 1,2


-- ###################################################################################
-- 								BAJADA DE NOTAS
-- ###################################################################################

	select 
		cp22.ciclo_lectivo,
		'Primario' nivel,
		cp22.id_alumno,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'primer bimestre' THEN cp22.nota ELSE NULL end) a_n1_mate,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'segundo bimestre' THEN cp22.nota ELSE NULL end) a_n2_mate,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'tercer bimestre' THEN cp22.nota ELSE NULL end) a_n3_mate,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'cuarto bimestre' THEN cp22.nota ELSE NULL end) a_n4_mate,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'primer bimestre' THEN cp22.nota ELSE NULL end) a_n1_lengua,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'segundo bimestre' THEN cp22.nota ELSE NULL end) a_n2_lengua,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'tercer bimestre' THEN cp22.nota ELSE NULL end) a_n3_lengua,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'cuarto bimestre' THEN cp22.nota ELSE NULL end) a_n4_lengua
	from educacion_staging.calificacion_bimestral_primaria_miescuela cp22 
	LEFT JOIN educacion_staging.espacio_curricular_historico_miescuela ec22 ON ec22.id_espacio_curricular_seccion = cp22.id_espacio_curricular_seccion AND ec22.ciclo_lectivo = cp22.ciclo_lectivo
	left join educacion_staging.calificacion_pps_miescuela cpm on cp22.id_alumno = cpm.id_alumno and cpm.ciclo_lectivo = 2022
	where true 
		and cp22.ciclo_lectivo = 2022
		and cpm.id_alumno is not null
	group by 1,2,3
	union all
	select 
		cs23.ciclo_lectivo,
		'Secundario' nivel, 
		cs23.id_alumno,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota end) end) a_n1_mate_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota END)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota END) end) a_n2_mate_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota end) end) a_n3_mate_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota end) end) a_n4_mate_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota end) end) a_n1_lengua_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota end) end) a_n2_lengua_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota end) end) a_n3_lengua_s,
		MAX(case when cs23.ciclo_lectivo = 2023 then (CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota end)
		  else (CASE WHEN LOWER(ec24.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota end) end) a_n4_lengua_s
	from educacion_staging.calificacion_bimestral_secundaria_miescuela cs23 
	LEFT JOIN educacion_staging.espacio_curricular_historico_miescuela ec23 ON ec23.id_espacio_curricular_seccion = cs23.id_espacio_curricular_seccion AND ec23.ciclo_lectivo = cs23.ciclo_lectivo
	LEFT JOIN educacion_staging.espacio_curricular_seccion_miescuela ec24 ON ec24.id_espacio_curricular_seccion = cs23.id_espacio_curricular_seccion AND ec24.ciclo_lectivo = cs23.ciclo_lectivo
	left join educacion_staging.calificacion_pps_miescuela cpm on cs23.id_alumno = cpm.id_alumno and cpm.ciclo_lectivo = 2022
	where true
		and cs23.ciclo_lectivo  >= 2023
		and cpm.id_alumno is not null
	group by 1,2,3



-- ###################################################################################
-- 									BAJADA DE APOYOS ESCOLARES
-- ###################################################################################

select
	c.id_alumno,
	periodo,
	(select value from json_each_text(c.aspectos_generales::json) where LOWER(key) = 'acompaniamientoareapregunta') ag_acompaniamiento,
	(select value from json_each_text(c.aspectos_generales::json) where LOWER(key) = 'acompaniamientoarea') ag_acompaniamiento_area,
	(select value from json_each_text(c.aspectos_generales::json) where LOWER(key) = 'apoyopregunta') ag_apoyo,
	(select value from json_each_text(c.aspectos_generales::json) where LOWER(key) = 'apoyo') ag_apoyo_tipo
from educacion_staging.calificacion_indicadores_bimestral_primaria_miescuela c
left join educacion_staging.calificacion_pps_miescuela cpm on c.id_alumno = cpm.id_alumno
where true 	
	and c.aspectos_generales is not null
	and c.ciclo_lectivo = 2022
	and cpm.id_alumno is not null
	


-- ###################################################################################
-- 						BAJADA DE RESPONSABLES IEL
-- ###################################################################################

with alumnos as (
	select dp.documento
	from educacion_staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula fm on cpm.id_alumno = fm.id_miescuela and fm.ciclo_lectivo = cpm.ciclo_lectivo
	left JOIN educacion_produccion.dim_seccion s ON s.id = fm.id_seccion
	left join educacion_produccion.dim_Alumno da on fm.id_alumno = da.id 
	left join educacion_produccion.dim_persona dp on da.id_persona = dp.id 
	where cpm.ciclo_lectivo = 2022
			and s.anio in ('7° Grado','6° y 7° Grado')
)	
select 
	fm.id_miescuela,
	ni.documento doc_alu,
	fi.documento doc_resp,
	fi.pais_nacimiento nac_resp,
	fi.vinculo,
	fi.nivel_educativo
from educacion_Staging.familiares_iel fi 
left join educacion_Staging.nomina_iel ni on ni.inscripcion_id = fi.id_inscripcion 
left join educacion_produccion.dim_persona dp on ni.documento = dp.documento 
left join educacion_produccion.dim_persona dp2 on fi.documento = dp.documento 
left join educacion_produccion.dim_alumno da on dp.id = da.id_persona 
left join educacion_produccion.ft_matricula fm on da.id = fm.id_alumno 
left join alumnos a on a.documento = ni.documento
where true
	and a.documento is not null 
	and fi.principal = 1
	and ni.ciclolectivo = 2023
group by 1,2,3,4,5,6
order by 1

-- ###################################################################################
-- 									BAJADA DE PASES
-- ###################################################################################

with alu_pps as (
		select cpm.id_alumno 
		from educacion_staging.calificacion_pps_miescuela cpm 
		left join educacion_produccion.ft_matricula fm on cpm.id_alumno = fm.id_miescuela and fm.ciclo_lectivo = cpm.ciclo_lectivo
		left JOIN educacion_produccion.dim_seccion s ON s.id = fm.id_seccion
		where cpm.ciclo_lectivo = 2022
				and s.anio in ('7° Grado','6° y 7° Grado')
)
select gdp.*
from educacion_staging.gestion_de_pase gdp 
left join alu_pps ap on ap.id_alumno = gdp.id_alumno 
where ap.id_alumno is not null 
	and anio_pase >= 2022
	and UPPER(establecimiento_origen) not like '%DEMO%'
	and UPPER(establecimiento_destino) not like '%DEMO%'
order by gdp.id_alumno, gdp.anio_pase

-- ###################################################################################
-- 						BAJADA DE SERVICIOS
-- ###################################################################################

with alumnos as (
	select 
		cpm.id_alumno
	from educacion_Staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula m22 on m22.id_miescuela = cpm.id_alumno and cpm.ciclo_lectivo = m22.ciclo_lectivo 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
	where true 
		and m22.ciclo_lectivo = 2022
		and s22.anio in ('7° Grado','6° y 7° Grado')
	group by 1
)
SELECT 
	a.ciclo_lectivo, 
	c.documento, 
	id_persona,
	case when ap.documento is not null then 1 else 0 end aprende_programando,
	case when ba.documento is not null then 1 else 0 end becas_alimentarias,
	case when t.documento is not null then 1 else 0 end transporte,
	case when nc.documento_alu is not null then 1 else 0 end colonias,
	case when abm.documento is not null then 1 else 0 end becas_media,
	case when ab.documento is not null then 1 else 0 end boleto
from educacion_produccion.ft_matricula a
left join educacion_produccion.dim_alumno b on a.id_alumno = b.id
left join educacion_produccion.dim_persona c on b.id_persona = c.id
left join educacion_staging.aprende_programando ap on ap.documento = c.documento and a.ciclo_lectivo::varchar = TO_DATE(ap.fecha::text, 'YYYY')::varchar
left join educacion_staging.becas_alimentarias ba on c.documento = ba.documento and ba.ciclo_lectivo::int = a.ciclo_lectivo 
left join educacion_staging.alumno_transporte t on c.documento = t.documento and a.ciclo_lectivo::varchar = to_date(t.fecha_registro::Text,'YYYY')::Varchar
left join educacion_staging.nomina_colonia nc on nc.documento_alu = c.documento and anio_fin_colonia::int = a.ciclo_lectivo 
left join educacion_staging.alumnobeca_becas_media abm on abm.documento = c.documento and abm.ciclo_lectivo = a.ciclo_lectivo -- tiene un filtro de estado_solicitud = 'Revisada' pero no existe entonces tira vacío, sugiero cambiarlo por UPPER(estado_solicitud) LIKE '%ACEPTADA%'
left join educacion_staging.alumno_boleto ab on ab.documento = c.documento and upper(tiene_sube) = 'SI' -- ACA FALTA LA PARTE DEL JOIN DEL CL
left join alumnos al on al.id_alumno = a.id_miescuela
where a.ciclo_lectivo >= 2022
	and al.id_alumno is not null

/*
select 	
	dp.documento,
	a.id_alumno id_miescuela,
	fss.*
from educacion_produccion.ft_servicios fss
left join educacion_produccion.dim_persona dp on dp.id = fss.id_persona 
left join educacion_produccion.dim_alumno da on da.id_persona = dp.id 
left join educacion_produccion.ft_matricula ft on ft.id_alumno = da.id and ft.ciclo_lectivo = fss.ciclo_lectivo 
left join alumnos a on a.id_alumno = ft.id_miescuela
where a.id_alumno is not null
 */

-- ###################################################################################
-- 								BAJADA DE PPS
-- ###################################################################################

select 
	-- datos personales del alumno
	cpm.ciclo_lectivo,
	cpm.id_alumno id_miesucela,
	dp.documento,
--variables del pps
	cpm.actitud,
	cpm.convivencia,
	cpm.trayectoria,
	cpm.vinculo,
	cpm.antecedentes,
	cpm.intervenciones,
	cpm.jornada
FROM educacion_staging.calificacion_pps_miescuela cpm 
left join educacion_produccion.ft_matricula m22 on m22.id_miescuela = cpm.id_alumno and cpm.ciclo_lectivo = m22.ciclo_lectivo 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
    left join educacion_produccion.dim_alumno da on m22.id_alumno = da.id 
    left join educacion_produccion.dim_persona dp on da.id_persona = dp.id
where true 
	and m22.ciclo_lectivo = 2022
	and s22.anio in ('7° Grado','6° y 7° Grado')

-- ###################################################################################
-- 							BAJADA DE DATOS PERSONALES
-- ###################################################################################

with alumnos as (
	select 
		cpm.id_alumno
	from educacion_Staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula m22 on m22.id_miescuela = cpm.id_alumno and cpm.ciclo_lectivo = m22.ciclo_lectivo 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
	where true 
		and m22.ciclo_lectivo = 2022
		and s22.anio in ('7° Grado','6° y 7° Grado')
	group by 1
)
select 
	-- datos personales del alumno
	m.id_miescuela id_miescuela,
	cast(dp.fecha_nacimiento as DATE) a_nacimiento,
	dp.genero a_genero,
	dp.nacionalidad a_nacionalidad,
	ds.sistema_salud a_sistema_salud,
	ds.disc_ninguno a_disc_ninguno,
	--ds.tratamiento_psicologia a_tratamiento_psicologia,
	ds.tratamiento_psicopedagogia a_tratamiento_psicopedagogia,
	dse.ingresos_grupo_familiar ase_ingreso_familiar,
	dse.trabaja ase_trabaja,
	--dse.padres_presos ase_padres_presos,
	dse.tiene_hijos ase_tiene_hijos,
	dse.tiene_subsidios ase_tiene_subsidios,
	CAST(dse.updated_at as date) ase_updated_at,
	--dse.responsable_propio a_responsable_propio,
	dd.provincia a_domicilio_provincia,
	dd.ciudad a_domicilio_ciudad,
	dd.calle,
	dd.altura,
	dd.hotel_familiar a_domicilio_hotel_familiar,
	dd.vive_pension a_domicilio_vive_pension,
	dd.situacion_calle a_domicilio_situacion_calle,
	dd.casa a_domicilio_casa,
	dd.vivienda_alquilada a_domicilio_vicienda_alquilada
FROM educacion_produccion.v_ft_matricula_indicadores m 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
		-- datos personales de los alumnos y sus responsables
	LEFT JOIN educacion_produccion.dim_alumno da ON m.id_alumno = da.id
    left JOIN educacion_produccion.dim_persona dp ON dp.id = da.id_persona
    left join educacion_produccion.dim_socio_economico dse on dse.id = dp.id_socio_economico
    left JOIN educacion_produccion.dim_domicilio dd ON dp.id_domicilio = dd.id
    LEFT JOIN educacion_produccion.dim_salud ds ON dp.id_salud = ds.id
    left join alumnos a on a.id_alumno = m.id_miescuela 
WHERE TRUE
	--AND cpm.ciclo_lectivo = 2022
	AND a.id_alumno is not null
	and s22.anio is not null
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
order by 1,2 

-- ###################################################################################
-- 					BAJADA DE SOCIOECONOMICA CON UPD TABLAS
--					PARA DS Y DD USE LO MISMO CON LAS QUERIES
--					NUEVAS DE CADA UNA DE ESAS TABLAS
-- ###################################################################################

with tablon as (
select 
trim(a.documento) documento
,ROUND(cast(ingresos_grupo_familiar as NUMERIC),0) ingresos_grupo_familiar
,null trabaja
,null sueldo
,null pension
,null es_pensionado
,null padres_presos
,null tiene_hijos
,null tiene_subsidios
,null responsable_propio
,case when pr.documento is not null then 1 else 0 end flag_renaper
,'BEA' as origen
,last_update as updated_at
from educacion_staging.becas_alimentarias a 
left join (select max(last_update) updated_at,documento as documento from educacion_staging.becas_alimentarias group by documento) b on a.documento = b.documento and a.last_update = b.updated_at
left join educacion_staging.persona_renaper pr on trim(replace(a.documento,'.','')) = pr.documento AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%' AND trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or (UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || '%' AND UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))  LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))) 
where true
and trim(a.documento) is not null
and a.ingresos_grupo_familiar is not null
--and b.updated_at >= '2023-01-01'
union all
select 
trim(a.documento) as documento
,null ingresos_grupo_familiar
,trabaja
,sueldo
,pension
,pensionado
,padres_presos
,tiene_hijos
,tiene_subsidios
,responsable_propio
,case when pr.documento is not null then 1 else 0 end flag_renaper
,'ABM' origen
,b.updated_at
from educacion_staging.alumno_becas_media a 
left join (select max(updated_at) updated_at,trim(documento) as documento from educacion_staging.alumno_becas_media group by documento) b on a.documento = b.documento and a.updated_at = b.updated_at
LEFT JOIN educacion_staging.persona_renaper pr ON trim(replace(a.documento, '.', '')) = pr.documento 
	AND ((((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.primer_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || ' ' || 
	        trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.segundo_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))
	        LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.primer_apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || ' ' || 
	        trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.segundo_apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g')))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%'))
	OR (((UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))) LIKE '%' || (trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.primer_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || ' ' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.segundo_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g')))) || '%')
        AND ((UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))) LIKE '%' || (trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.primer_apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || ' ' || 
            trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.segundo_apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g')))) || '%')))
where true
and a.documento is not null
--and b.updated_at >= '2023-01-01'
), mas_reciente as (
select 
	*,
	ROW_NUMBER() OVER (PARTITION BY documento,flag_renaper ORDER BY updated_at DESC) AS mas_reciente
from tablon
-- esta habiendo una mala asignacion en las cols de updated de los distintos valores de dinero porque
-- hay filas que tienen nulo en updated_At, entonces aparecen caso donde el hay un valor en plata y aun asi
-- el valor de updated correspondiente a esa columna es nulo
), asig as (
select
	documento,
	flag_renaper,
	FIRST_VALUE(ingresos_grupo_familiar) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN ingresos_grupo_familiar IS NOT NULL THEN mas_reciente END ASC
        ) AS ingresos_grupo_familiar,
    FIRST_VALUE(trabaja) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN trabaja IS NOT NULL THEN mas_reciente END ASC
        ) AS trabaja,
    FIRST_VALUE(sueldo) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN sueldo IS NOT NULL THEN mas_reciente END ASC
    	) AS sueldo,
    FIRST_VALUE(pension) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN pension IS NOT NULL THEN mas_reciente END ASC
    	) AS pension,
    FIRST_VALUE(es_pensionado) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN es_pensionado IS NOT NULL THEN mas_reciente END ASC
    	) AS es_pensionado,
    FIRST_VALUE(padres_presos) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN padres_presos IS NOT NULL THEN mas_reciente END ASC
    	) AS padres_presos,
    FIRST_VALUE(tiene_hijos) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN tiene_hijos IS NOT NULL THEN mas_reciente END ASC
    	) AS tiene_hijos,
    FIRST_VALUE(tiene_subsidios) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN tiene_subsidios IS NOT NULL THEN mas_reciente END ASC
    	) AS tiene_subsidios,
    FIRST_VALUE(responsable_propio) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN responsable_propio IS NOT NULL THEN mas_reciente END ASC
    	) AS responsable_propio,
    -- ESTE CASE LO QUE HACE ES FIJARSE SI INGRESOS FAMILAIRES DA NULO, EN SE CASO LE PONE NULO A LA COL
    -- DEL UPD INGRESO FAMILIAR Y SINO VA Y BUSCA EL MAS RECIENTE NO NULO // LO MISMO DSP CON SUELDO Y PENSION
    case when FIRST_VALUE(ingresos_grupo_familiar) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN ingresos_grupo_familiar IS NOT NULL THEN mas_reciente END ASC
        ) is not null then 
       FIRST_VALUE(updated_at) OVER (
            PARTITION BY documento, flag_renaper
            ORDER BY CASE WHEN ingresos_grupo_familiar IS NOT NULL THEN mas_reciente else null end ASC
        ) else null END AS upd_ingresos_grupo_familiar,
    case when FIRST_VALUE(sueldo) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN sueldo IS NOT NULL THEN mas_reciente END ASC
    	) is not null then 
       FIRST_VALUE(updated_at) OVER (
            PARTITION BY documento, flag_renaper
            ORDER BY CASE WHEN sueldo IS NULL THEN null else mas_reciente end ASC
        ) else null end AS upd_sueldo,
    case when FIRST_VALUE(pension) OVER (
       PARTITION BY documento,flag_renaper
       ORDER BY CASE WHEN pension IS NOT NULL THEN mas_reciente END ASC
    	) is not null then 
       FIRST_VALUE(updated_at) OVER (
            PARTITION BY documento, flag_renaper
            ORDER BY CASE WHEN pension IS NULL THEN null else mas_reciente end ASC
        ) else null end AS upd_pension
from mas_reciente
), alu as (
	select dp.documento
	from educacion_staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula fm on cpm.id_alumno = fm.id_miescuela and fm.ciclo_lectivo = cpm.ciclo_lectivo
	left JOIN educacion_produccion.dim_seccion s ON s.id = fm.id_seccion
	left join educacion_produccion.dim_Alumno da on fm.id_alumno = da.id 
	left join educacion_produccion.dim_persona dp on da.id_persona = dp.id 
	where cpm.ciclo_lectivo = 2022
		and s.anio in ('7° Grado','6° y 7° Grado')
),alu_resp as (
	select 
		fm.id_miescuela,
		ni.documento doc_alu,
		fi.documento doc_resp
		/*
		fi.pais_nacimiento nac_resp,
		fi.vinculo,
		fi.nivel_educativo
		*/
	from educacion_Staging.familiares_iel fi 
	left join educacion_Staging.nomina_iel ni on ni.inscripcion_id = fi.id_inscripcion 
	left join educacion_produccion.dim_persona dp on ni.documento = dp.documento 
	left join educacion_produccion.dim_alumno da on dp.id = da.id_persona 
	left join educacion_produccion.ft_matricula fm on da.id = fm.id_alumno
	left join alu a on a.documento = ni.documento
	where a.documento is not null
		and fi.principal = 1
		and ni.ciclolectivo = 2023
	group by 1,2,3 --,4,5,6
	order by 1
)
select a.*
from asig a
left join alu_resp al on a.documento = al.doc_alu 
left join alu_resp al2 on a.documento = al2.doc_resp 
where true 
	and (al.doc_alu is not null or al2.doc_resp is not null)
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
order by 1,2



-- ###################################################################################
-- 							BAJADA DE DIM DOMICILIO NUEVA
-- ###################################################################################



with tablon as (
select
	trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) documento,
	direccion as calle,
	null altura,
	null piso,
	depto::varchar depto,
	villa,
	a.barrio,
	null ciudad,
	localidad,
	null cpostal,
	partido,
	prov_alu as provincia,
	nhp,
	hotel_familiar,
	pension vive_pension,
	situacion_calle,
	casa::int casa,
	vivienda_alquilada,
	null::numeric coord_x,
	null::numeric coord_y,
	case when pr.documento is not null then 1 else 0 end flag_renaper,
	0 domicilio_renaper,
	'IEL ALU' origen,
	TO_CHAR(a.updated_at, 'YYYY-MM-DD') updated_at
from educacion_staging.alumnos_iel a
left join (select documento,max(updated_at) updated_at from educacion_staging.alumnos_iel ai group by documento) b on a.documento = b.documento and a.updated_at = b.updated_at
left join educacion_staging.persona_renaper pr on trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) = pr.documento AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%' AND trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or (UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || '%' AND UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))  LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))) 
where TRUE
--	and not exists (select 1 from educacion_produccion.dim_domicilio dp
--						where dp.documento = trim(a.documento))
--	and exists (select 1 from educacion_produccion.dim_domicilio dp where dp.documento = trim(a.documento) and calle is null)
	and trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) is not null
union all 
select 
	trim(UPPER(REGEXP_REPLACE(fi.documento, '[^a-zA-Z0-9]', '', 'g'))) as documento 
	,fi.calle
	,numero altura
	,fi.piso 
	,depto::varchar depto 
	,villa
	,fi.barrio 
	,ciudad
	,localidadprovincia --son todos nulos
	,codigopostal 
	,null partido
	,null provincia
	,nhp 
	,hotelfamiliar 
	,pension 
	,null situacion_calle
	,casa::int casa 
	,null vivienda_alquilada
	,coordenadax::numeric coordenadax
	,coordenaday::numeric coordenaday
	,case when pr.documento is not null then 1 else 0 end flag_renaper
	,0 domicilio_renaper
	,'IEL RESP' origen
	,TO_CHAR(fecha_ultima_mod, 'YYYY-MM-DD') updated_at
from educacion_staging.familiares_iel fi
	left join educacion_staging.persona_renaper pr on trim(UPPER(REGEXP_REPLACE(fi.documento, '[^a-zA-Z0-9]', '', 'g'))) = pr.documento AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(fi.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%' AND trim(UPPER(REGEXP_REPLACE(TRANSLATE(fi.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or (UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(fi.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || '%' AND UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))  LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(fi.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))) 
where fi.calle is not null
--and not exists (select 1 from educacion_produccion.dim_domicilio dp where dp.documento = trim(fi.documento))
--and exists (select 1 from educacion_produccion.dim_domicilio dp where dp.documento = trim(fi.documento) and dp.calle is null)
union all
select 
	trim(UPPER(REGEXP_REPLACE(b.documento, '[^a-zA-Z0-9]', '', 'g'))) documento
	,a.calle
	,null altura
	,a.piso 
	,a.dpto::varchar dpto 
	,null villa
	,null barrio
	,null ciudad
	,localidad
	,cod_postal 
	,null partido
	,null provincia
	,null nhp
	,null hotel_familiar
	,null vive_pension
	,null situacion_calle
	,case when casa::varchar is not null then 1 else 0 end casa
	,null vivienda_alquilada
	,null::numeric coord_x
	,null::numeric coord_y
	,case when pr.documento is not null then 1 else 0 end flag_renaper
	,0 domicilio_renaper
	,'BECAS MEDIA ALU' origen
	,TO_CHAR(updated_at, 'YYYY-MM-DD') updated_at 
from educacion_staging.alumno_domicilio_becas_media a
left join (select trim(UPPER(REGEXP_REPLACE(TRANSLATE(primer_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g')))  || ' ' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(segundo_nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) nombre
			,trim(UPPER(REGEXP_REPLACE(TRANSLATE(PRIMER_APELLIDO, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g')))  || ' ' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(SEGUNDO_APELLIDO, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) apellido,
			alumno_id,documento,max(updated_at) updated_at from educacion_staging.alumno_becas_media group by 1,2,3,4) b on a.alumno_id = b.alumno_id 
left join educacion_staging.persona_renaper pr on trim(UPPER(REGEXP_REPLACE(b.documento, '[^a-zA-Z0-9]', '', 'g'))) = pr.documento AND ((b.nombre LIKE '%' || UPPER(pr.nombre) || ' ' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or ((UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || ' ' || UPPER(TRANSLATE(pr.APELLIDO, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) ) LIKE '%' || b.nombre || ' ' || b.apellido || '%'))
where TRUE
	and trim(UPPER(REGEXP_REPLACE(b.documento, '[^a-zA-Z0-9]', '', 'g'))) is not null
--	and not exists (select 1 from educacion_produccion.dim_domicilio c where c.documento = trim(b.documento))
--	and exists (select 1 from educacion_produccion.dim_domicilio c where c.documento = trim(b.documento) and calle is null)
	and a.calle is not null
union all 
select 
	trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) as documento
	,b.calle
	,numero as altura
	,dpto_piso
	,null::varchar depto
	,null villa
	,null barrio
	,ciudad
	,null localidad
	,codigo_postal
	,null partido
	,b.provincia
	,null nhp
	,null hotel_familiar
	,null vive_pension
	,null situacion_calle
	,null::int casa
	,null vivienda_alquilada
	,null::numeric coord_x
	,null::numeric coord_y
	,case when pr.documento is not null then 1 else 0 end flag_renaper
	,0 domicilio_renaper
	,'APP MIESCUELA RESP' origen
	,TO_CHAR(b.updated_at, 'YYYY-MM-DD') updated_at 
from educacion_staging.responsables_appme a  
left join educacion_staging.domicilios_appme b on a.address_id = b.id
left join (select documento , max(updated_at) updated_at from educacion_staging.responsables_appme b group by documento) c on c.documento = a.documento and c.updated_at = a.updated_at
left join educacion_staging.persona_renaper pr on trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) = pr.documento AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%' AND trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or (UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || '%' AND UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))  LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))) 
where true 
--	and not exists (select 1 from educacion_produccion.dim_domicilio dp where dp.documento = trim(a.documento))
--	and exists (select 1 from educacion_produccion.dim_domicilio dp where dp.documento = trim(a.documento))
	and trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) is not null
-- ESTE LO AGREGO PORQUE NO SE PORQUE ESTABAMOS DEJANDO AFUERA LOS DATOS DE ALUMNOS DE LA APP
--
union all 
select 
	trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) as documento
	,b.calle
	,numero as altura
	,dpto_piso
	,null::varchar depto
	,null villa
	,null barrio
	,ciudad
	,null localidad
	,codigo_postal
	,null partido
	,b.provincia
	,null nhp
	,null hotel_familiar
	,null vive_pension
	,null situacion_calle
	,null::int casa
	,null vivienda_alquilada
	,null::numeric coord_x
	,null::numeric coord_y
	,case when pr.documento is not null then 1 else 0 end flag_renaper
	,0 domicilio_renaper
	,'APP MIESCUELA ALU' origen
	,TO_CHAR(b.updated_at, 'YYYY-MM-DD') updated_at 
from educacion_staging.alumnos_appme a  
left join educacion_staging.domicilios_appme b on a.address_id = b.id
left join (select documento , max(updated_at) updated_at from educacion_staging.responsables_appme b group by documento) c on c.documento = a.documento and c.updated_at = a.updated_at
left join educacion_staging.persona_renaper pr on trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) = pr.documento AND ((trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%' AND trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) LIKE '%' || UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) || '%') or (UPPER(TRANSLATE(pr.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU')) LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.nombre, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))) || '%' AND UPPER(TRANSLATE(pr.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))  LIKE '%' || trim(UPPER(REGEXP_REPLACE(TRANSLATE(a.apellido, 'áéíóúÁÉÍÓÚ', 'aeiouAEIOU'), '\s+', ' ', 'g'))))) 
where true 
	and trim(UPPER(REGEXP_REPLACE(a.documento, '[^a-zA-Z0-9]', '', 'g'))) is not null 
),renaper as (
SELECT *
FROM tablon
UNION all
select 
	trim(UPPER(REGEXP_REPLACE(pr.documento, '[^a-zA-Z0-9]', '', 'g'))) as documento
	,pr.calle
	,pr.numerocalle altura
	,pr.piso
	,pr.dpto
	,null villa
	,pr.barrio
	,null ciudad --ver que onda esto si tenemos la info de las ciudades desde otro lado
	,null localidad
	,pr.cpostal
	,pr.municipio partido
	,pr.provincia
	,null nhp
	,null hotel_familiar
	,null vive_pension
	,null situacion_calle
	,null::int casa
	,null vivienda_alquilada
	,null::numeric coord_x
	,null::numeric coord_y
	,1 flag_renaper
	,1 domicilio_renaper
	,'RENAPER' origen
	,TO_CHAR(pr.emision, 'YYYY-MM-DD') updated_at 
from educacion_staging.persona_renaper pr
left join tablon tp on pr.documento = tp.documento and tp.flag_renaper = 1
), tablon_p as (
	select 
		trim(UPPER(REGEXP_REPLACE(documento, '[^a-zA-Z0-9]', '', 'g'))) documento
		,TRIM(UPPER(calle)) calle
		,TRIM(UPPER(altura)) altura
		,TRIM(piso) piso
		,TRIM(depto) depto
		,villa villa
		,UPPER(TRIM(barrio)) barrio
		--comentar esta linea de ciudad despues
		--,UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(ciudad, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) check_ciudad
		,CASE WHEN UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(ciudad, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) ~ 
		             '(CABA|CIUDAD(\s+AUT[ÓO]NOMA)?(\s+DE)?(\s+BUENOS?)?\s+AIRES?)'
		     THEN 'CIUDAD AUTONOMA DE BUENOS AIRES'
		     ELSE UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(ciudad, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) END AS ciudad
    	,UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(localidad, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) localidad
		,UPPER(replace(replace(cpostal,'.',''),'-','')) codigo_postal
		,UPPER(TRIM(partido)) partido
		,CASE WHEN (CASE WHEN UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(provincia, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) IN ('CABA', 'CIUDAD AUTÓNOMA DE BUENOS AIRES') 
                THEN 'CABA' ELSE UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(provincia, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) END) 
            IS NULL AND UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(ciudad, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' '))) ~ '(CAPITAL FEDERAL|CABA|CIUDAD(\s+AUT[ÓO]NOMA)?(\s+DE)?(\s+BUENOS?)?\s+AIRES?)' THEN 'CABA'
            	ELSE  UPPER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(provincia, '[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ ]', '', 'g'), '\s+', ' ')))
		END AS provincia
		,nhp
		,hotel_familiar
		,vive_pension
		,situacion_calle
		,casa
		,vivienda_alquilada
		,coord_x
		,coord_y
		,flag_renaper
		,domicilio_renaper
		,origen
		,updated_at
	from renaper
), alu as (
	select dp.documento
	from educacion_staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula fm on cpm.id_alumno = fm.id_miescuela and fm.ciclo_lectivo = cpm.ciclo_lectivo
	left JOIN educacion_produccion.dim_seccion s ON s.id = fm.id_seccion
	left join educacion_produccion.dim_Alumno da on fm.id_alumno = da.id 
	left join educacion_produccion.dim_persona dp on da.id_persona = dp.id 
	where cpm.ciclo_lectivo = 2022
		and s.anio in ('7° Grado','6° y 7° Grado')
),alu_resp as (
	select 
		fm.id_miescuela,
		ni.documento doc_alu,
		fi.documento doc_resp
	from educacion_Staging.familiares_iel fi 
	left join educacion_Staging.nomina_iel ni on ni.inscripcion_id = fi.id_inscripcion 
	left join educacion_produccion.dim_persona dp on ni.documento = dp.documento 
	left join educacion_produccion.dim_alumno da on dp.id = da.id_persona 
	left join educacion_produccion.ft_matricula fm on da.id = fm.id_alumno
	left join alu a on a.documento = ni.documento
	where a.documento is not null
		and fi.principal = 1
		and ni.ciclolectivo = 2023
	group by 1,2,3
	order by 1
)
select 
	tp.*,
	ROW_NUMBER() OVER (PARTITION BY tp.documento, tp.flag_renaper ORDER BY tp.updated_at DESC) AS mas_reciente
from tablon_p TP
LEFT JOIN alu_resp a ON tp.documento = a.doc_alu
LEFT JOIN alu_resp a2 ON tp.documento = a2.doc_resp
where true
	and tp.documento is not null
	and length(tp.documento) > 0 -- me traía casos donde el doc era nulo aun con el filtro
	and NOT (tp.domicilio_renaper = 1 AND tp.codigo_postal IS NULL AND tp.provincia IS NULL) -- encuentra DNIS llenos de 0 en renaper pero que no tenían el domicilio en si
	AND (a.doc_alu IS NOT NULL OR a2.doc_resp IS NOT null)
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
order by tp.documento


-- ###################################################################################
-- 							BAJADA DE MATRICULA
-- ###################################################################################

with alumnos as (
	select 
		cpm.id_alumno
	from educacion_Staging.calificacion_pps_miescuela cpm 
	left join educacion_produccion.ft_matricula m22 on m22.id_miescuela = cpm.id_alumno and cpm.ciclo_lectivo = m22.ciclo_lectivo 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
	where true 
		and m22.ciclo_lectivo = 2022
		and s22.anio in ('7° Grado','6° y 7° Grado')
	group by 1
)
select 
	-- datos personales del alumno
	m.ciclo_lectivo,
	m.id_miescuela,
	dp.documento,
	--variables de la matricula 2022
	s22.anio,
	s22.turno turno,
	s22.jornada jornada,
	s22.capacidad_maxima capacidad_maxima,
	de22.cue_anexo cueanexo,
	de22.dependencia_funcional dependencia_funcional,
	de22.modalidad modalidad,
	de22.distrito_escolar distrito_escolar,
	de22.comuna comuna,
	pu.barrio,
	pu.calle,
	pu.num altura,
	pu.point_x coord_x,
	pu.point_y coord_y,
	-- repite
	m.repite,
	m.sobreedad
FROM educacion_produccion.v_ft_matricula_indicadores m 
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
	left join educacion_staging.padron_ueicee_2024_localizacion pu on pu.cueanexo = de22.cue_anexo 
    left join alumnos a on a.id_alumno = m.id_miescuela 
    left join educacion_produccion.dim_alumno da on m.id_alumno = da.id 
    left join educacion_produccion.dim_persona dp on da.id_persona = dp.id
WHERE true
	AND a.id_alumno is not null
	and s22.anio is not null
	and m.ciclo_lectivo >= 2022
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
order by 2,1

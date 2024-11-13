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

with notas_p as (
	select 
		cp22.id_alumno,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'primer bimestre' THEN cp22.nota ELSE NULL end) a_n1_mate_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'segundo bimestre' THEN cp22.nota ELSE NULL end) a_n2_mate_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'tercer bimestre' THEN cp22.nota ELSE NULL end) a_n3_mate_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'matemática' AND LOWER(cp22.periodo) = 'cuarto bimestre' THEN cp22.nota ELSE NULL end) a_n4_mate_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'primer bimestre' THEN cp22.nota ELSE NULL end) a_n1_lengua_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'segundo bimestre' THEN cp22.nota ELSE NULL end) a_n2_lengua_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'tercer bimestre' THEN cp22.nota ELSE NULL end) a_n3_lengua_p,
		MAX(CASE WHEN LOWER(ec22.descripcion) = 'prácticas del lenguaje' AND LOWER(cp22.periodo) = 'cuarto bimestre' THEN cp22.nota ELSE NULL end) a_n4_lengua_p
	from educacion_staging.calificacion_bimestral_primaria_miescuela cp22 
	LEFT JOIN educacion_staging.espacio_curricular_historico_miescuela ec22 ON ec22.id_espacio_curricular_seccion = cp22.id_espacio_curricular_seccion AND ec22.ciclo_lectivo = cp22.ciclo_lectivo
	where true 
		and cp22.ciclo_lectivo = 2022
		and id_alumno in (select id_alumno from educacion_staging.calificacion_pps_miescuela cpm where ciclo_lectivo = 2022)
	group by 1
	
), notas_s as (
	select 
		cs23.id_alumno,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota ELSE NULL end) a_n1_mate_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota ELSE NULL end) a_n2_mate_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota ELSE NULL end) a_n3_mate_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'matemática' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota ELSE NULL end) a_n4_mate_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'primer bimestre' THEN cs23.nota ELSE NULL end) a_n1_lengua_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'segundo bimestre' THEN cs23.nota ELSE NULL end) a_n2_lengua_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'tercer bimestre' THEN cs23.nota ELSE NULL end) a_n3_lengua_s,
		MAX(CASE WHEN LOWER(ec23.descripcion) LIKE 'lengua y literatura' AND LOWER(cs23.periodo) = 'cuarto bimestre' THEN cs23.nota ELSE NULL end) a_n4_lengua_s
	from educacion_staging.calificacion_bimestral_secundaria_miescuela cs23 
	LEFT JOIN educacion_staging.espacio_curricular_historico_miescuela ec23 ON ec23.id_espacio_curricular_seccion = cs23.id_espacio_curricular_seccion AND ec23.ciclo_lectivo = cs23.ciclo_lectivo
	where true
		and cs23.ciclo_lectivo = 2023
		and id_alumno in (select id_alumno from educacion_staging.calificacion_pps_miescuela cpm where ciclo_lectivo = 2022)
	group by 1
		
)

select 
	cpm.id_alumno,
-- notas CL 22
	np.a_n1_mate_p,
	np.a_n2_mate_p,
	np.a_n3_mate_p,
	np.a_n4_mate_p,
	np.a_n1_lengua_p,
	np.a_n2_lengua_p,
	np.a_n3_lengua_p,
	np.a_n4_lengua_p,
	
-- notas CL 23
	ns.a_n1_mate_s,
	ns.a_n2_mate_s,
	ns.a_n3_mate_s,
	ns.a_n4_mate_s,
	ns.a_n1_lengua_s,
	ns.a_n2_lengua_s,
	ns.a_n3_lengua_s,
	ns.a_n4_lengua_s	
from educacion_staging.calificacion_pps_miescuela cpm 
left join notas_p np on np.id_alumno = cpm.id_alumno 
left join notas_s ns on ns.id_alumno = cpm.id_alumno
where cpm.ciclo_lectivo = 2022



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
where true 	
	and c.aspectos_generales is not null
	and c.ciclo_lectivo = 2022
	and C.id_alumno in (select id_alumno from educacion_staging.calificacion_pps_miescuela cpm where ciclo_lectivo = 2022)

	
	

-- ###################################################################################
-- 						BAJADA DE RESPONSABLES IEL
-- ###################################################################################

	
select 
	fm.id_miescuela,
	ni.documento doc_alu,
	fi.pais_nacimiento nac_resp,
	fi.vinculo,
	fi.nivel_educativo
from educacion_Staging.familiares_iel fi 
left join educacion_Staging.nomina_iel ni on ni.inscripcion_id = fi.id_inscripcion 
left join educacion_produccion.dim_persona dp on ni.documento = dp.documento 
left join educacion_produccion.dim_alumno da on dp.id = da.id_persona 
left join educacion_produccion.ft_matricula fm on da.id = fm.id_alumno 
where ni.documento in (select dp.documento
						from educacion_staging.calificacion_pps_miescuela cpm 
						left join educacion_produccion.ft_matricula fm on cpm.id_alumno = fm.id_miescuela and fm.ciclo_lectivo = cpm.ciclo_lectivo
						left JOIN educacion_produccion.dim_seccion s ON s.id = fm.id_seccion
						left join educacion_produccion.dim_Alumno da on fm.id_alumno = da.id 
						left join educacion_produccion.dim_persona dp on da.id_persona = dp.id 
						where cpm.ciclo_lectivo = 2022
								and s.anio in ('7° Grado','6° y 7° Grado'))
	and fi.principal = 1
	and ni.ciclolectivo = 2023
group by 1,2,3,4,5
order by 1

-- ###################################################################################
-- 						BAJADA DE INFORMACION GENERAL
-- ###################################################################################


select 
-- datos personales del alumno
	da.id id_alumno,
	cpm.id_alumno id_miescuela,
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
	dd.hotel_familiar a_domicilio_hotel_familiar,
	dd.vive_pension a_domicilio_vive_pension,
	dd.situacion_calle a_domicilio_situacion_calle,
	dd.casa a_domicilio_casa,
	dd.vivienda_alquilada a_domicilio_vicienda_alquilada,

--variables de la matricula 2022
	s22.turno e22_turno,
	s22.jornada e22_jornada,
	s22.capacidad_maxima e22_capacidad_maxima,
	de22.cue_anexo e22_cueanexo,
	de22.dependencia_funcional e22_dependencia_funcional,
	de22.modalidad e22_modalidad,
	de22.distrito_escolar e22_distrito_escolar,
	de22.comuna e22_comuna,
	
--variables del pps
	cpm.actitud,
	cpm.convivencia,
	cpm.trayectoria,
	cpm.vinculo,
	cpm.antecedentes,
	cpm.intervenciones,
	cpm.jornada,
	
--variables de la matricula 2023
	s23.anio e23_anio,
	s23.turno e23_turno,
	s23.jornada e23_jornada,
	s23.capacidad_maxima e23_capacidad_maxima,
	de23.cue_anexo e23_cueanexo,
	de23.dependencia_funcional e23_dependencia_funcional,
	de23.modalidad e23_modalidad,
	de23.distrito_escolar e23_distrito_escolar,
	de23.comuna e23_comuna,
	
--variables de la matricula 2024
	s24.anio e24_anio,
	s24.turno e24_turno,
	s24.jornada e24_jornada,
	s24.capacidad_maxima e24_capacidad_maxima,
	de24.cue_anexo e24_cueanexo,
	de24.dependencia_funcional e24_dependencia_funcional,
	de24.modalidad e24_modalidad,
	de24.distrito_escolar e24_distrito_escolar,
	de24.comuna e24_comuna,

-- repite
	case when LOWER(s23.anio) = LOWER(s24.anio) then 1 else 0 end repite_23_24
	
FROM educacion_staging.calificacion_pps_miescuela cpm
	-- matricula 2022
	left join educacion_produccion.ft_matricula m22 on m22.id_miescuela = cpm.id_alumno and m22.ciclo_lectivo = 2022
	left JOIN educacion_produccion.dim_seccion s22 ON s22.id = m22.id_seccion
	left join educacion_produccion.dim_establecimiento de22 on de22.id = s22.id_establecimiento
	--LEFT JOIN educacion_staging.calificacion_bimestral_primaria_miescuela cp22 ON cp22.id_alumno = m22.id_miescuela AND cp22.ciclo_lectivo = m22.ciclo_lectivo
	--LEFT JOIN educacion_staging.espacio_curricular_historico_miescuela ec22 ON ec22.id_espacio_curricular_seccion = cp22.id_espacio_curricular_seccion AND ec22.ciclo_lectivo = cp22.ciclo_lectivo
	-- datos personales de los alumnos y sus responsables
	LEFT JOIN educacion_produccion.dim_alumno da ON m22.id_alumno = da.id
    left JOIN educacion_produccion.dim_persona dp ON dp.id = da.id_persona
    left join educacion_produccion.dim_socio_economico dse on dse.id = dp.id_socio_economico
    left JOIN educacion_produccion.dim_domicilio dd ON dp.id_domicilio = dd.id
    LEFT JOIN educacion_produccion.dim_salud ds ON dp.id_salud = ds.id

	--matricula 2023 
	left join educacion_produccion.ft_matricula m23 on m23.id_miescuela = m22.id_miescuela and m23.ciclo_lectivo = 2023
	left JOIN educacion_produccion.dim_seccion s23 ON s23.id = m23.id_seccion
	left join educacion_produccion.dim_establecimiento de23 on de23.id = s23.id_establecimiento
	--matricula 2024
	left join educacion_produccion.ft_matricula m24 on m24.id_miescuela = m23.id_miescuela and m24.ciclo_lectivo = 2024
	left JOIN educacion_produccion.dim_seccion s24 ON s24.id = m24.id_seccion
	left join educacion_produccion.dim_establecimiento de24 on de24.id = s24.id_establecimiento
WHERE TRUE
	AND cpm.ciclo_lectivo = 2022
	AND s22.anio in ('7° Grado','6° y 7° Grado')
	and s23.anio is not null
	and s24.anio is not null
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53




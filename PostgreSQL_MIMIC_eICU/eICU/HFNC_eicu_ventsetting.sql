DROP TABLE IF EXISTS hfnc.ventsettings0; 
CREATE TABLE hfnc.ventsettings0 AS

WITH respchart AS (
	SELECT *
	FROM eicuii.respiratorycharting
)

, nursechart AS (
	SELECT *
	FROM eicuii.nursecharting
)

, pat AS (
	SELECT *
	FROM eicuii.patient
)


-- Extract the type of oxygen therapy.
-- The categories are invasive ventilation,
-- noninvasive ventilation, and supplemental oxygen.
-- `oxygen_therapy_type = -1` indicates oxygen therapy,
-- i.e. more oxygen than in room air is administered.

	SELECT patientunitstayid AS icustay_id
		, charttime
		, CASE

			-- Invasive ventilation
			WHEN
				string IN (
					'plateau pressure',
					'postion at lip',
					'position at lip',
					'pressure control'
				)
				OR string LIKE '%set vt%'
				OR string LIKE '%sputum%'
				OR string LIKE '%rsbi%'
				OR string LIKE '%tube%'
				OR string LIKE '%ett%'
				OR string LIKE '%endotracheal%'
				OR string LIKE '%tracheal suctioning%'
				OR string LIKE '%tracheostomy%'
				OR string LIKE '%reintubation%'
				OR string LIKE '%assist controlled%'
				OR string LIKE '%volume controlled%'
				OR string LIKE '%pressure controlled%'
				OR string LIKE '%trach collar%'
			THEN 4

			-- Noninvasive ventilation
			WHEN
				string IN (
					'bi-pap',
					'ambubag'
				)
				OR string LIKE '%ipap%'
				OR string LIKE '%niv%'
				OR string LIKE '%epap%'
				OR string LIKE '%mask leak%'
				OR string LIKE '%volume assured%'
				OR string LIKE '%non-invasive ventilation%'
				OR string LIKE '%cpap%'
			THEN 3
			
			-- HFNC:
			WHEN
				string IN (
					'hfnc',
					'hhfnc',
					'high flow nasal canula'
				)
-- 					'high flow',
-- 					'hi flow',
-- 					'hiflow',
				OR string LIKE '%high flow nasal cannula%'
			THEN 666

			-- Either invasive or noninvasive ventilation:
			WHEN
				string IN (
					'flowtrigger',
					'peep',
					'tv/kg ibw',
					'mean airway pressure',
					'peak insp. pressure',
					'exhaled mv',
					'exhaled tv (machine)',
					'exhaled tv (patient)',
					'flow sensitivity',
					'peak flow',
					'f total',
					'pressure to trigger ps',
					'adult con setting set rr',
					'adult con setting set vt',
					'vti',
					'exhaled vt',
					'adult con alarms hi press alarm',
					'mve',
					'respiratory phase',
					'inspiratory pressure, set',
					'a1: high exhaled vt',
					'set fraction of inspired oxygen (fio2)',
					'insp flow (l/min)',
					'adult con setting spont exp vt',
					'spont tv',
					'pulse ox results vt',
					'vt spontaneous (ml)',
					'peak pressure',
					'ltv1200',
					'tc'
				)
				OR (
					string LIKE '%vent%'
					AND NOT string LIKE '%hyperventilat%'
				)
				OR string LIKE '%tidal%'
				OR string LIKE '%flow rate%'
				OR string LIKE '%minute volume%'
				OR string LIKE '%leak%'
				OR string LIKE '%pressure support%'
				OR string LIKE '%peep%'
				OR string LIKE '%tidal volume%'
			THEN 2

			-- Supplemental oxygen:
			WHEN
				string IN (
					't-piece',
					'blow-by',
					'oxyhood',
					'nc',
					'oxymizer',
					'hfnc',
					'oximizer',
					'high flow',
					'oxymask',
					'nch',
					'hi flow',
					'hiflow',
					'hhfnc',
					'nasal canula',
					'face tent',
					'high flow mask',
					'aerosol mask',
					'venturi mask',
					'cool aerosol mask',
					'simple mask',
					'face mask'
				)
				OR string LIKE '%nasal cannula%'
				OR string LIKE '%non-rebreather%'
				OR string LIKE '%nasal mask%'
				OR string LIKE '%face tent%'
			THEN 1

			-- Oxygen therapy but unknown what type:
			WHEN
				string IN (
					'pressure support',
					'rr spont',
					'ps',
					'insp cycle off (%)',
					'trach mask/collar'
				)
				OR string LIKE '%spontaneous%'
				OR string LIKE '%oxygen therapy%'
			THEN 0

			-- Supplemental oxygen therapy,
			-- i.e. more oxygen than in room air is administered.
			WHEN
				string IN (
					'lpm o2'
				)
			THEN -1

			ELSE NULL

		END AS oxygen_therapy_type
		, activeUponDischarge
	FROM (

		SELECT patientunitstayid
			, nursingChartOffset AS charttime
			, LOWER(nursingchartvalue) AS string
			, NULL AS activeUponDischarge
		FROM nursechart

		UNION ALL

		SELECT patientunitstayid
			, respchartoffset AS charttime
			, LOWER(respchartvaluelabel) AS string
			, NULL AS activeUponDischarge
		FROM respchart

		UNION ALL

		-- Oxygen device from respchart
		SELECT patientunitstayid
			, respchartoffset AS charttime
			, LOWER(respchartvalue) AS string
			, NULL AS activeUponDischarge
		FROM respchart
		WHERE LOWER(respchartvaluelabel) IN (
			'o2 device',
			'respiratory device',
			'ventilator type',
			'oxygen delivery method'
    	)

    	UNION ALL

    	-- The treatment table also contains info on oxygen therapy.
    	SELECT patientunitstayid
			, treatmentoffset AS charttime
			, LOWER(treatmentstring) AS string
			, activeUponDischarge
		FROM eicuii.treatment
	)
	WHERE charttime >= -60

	UNION ALL

	-- The following indicates oxygen therapy but unclear what type.
	SELECT patientunitstayid AS icustay_id
		, nursingchartoffset AS charttime
		, -1 AS oxygen_therapy_type
		, NULL AS activeUponDischarge
	FROM nursechart
	WHERE nursingchartoffset >= -60
		AND nursingchartcelltypevallabel = 'O2 L/%'
  		AND CASE
        		WHEN nursingchartvalue ~ '^[0-9]+$' THEN CAST(nursingchartvalue AS INTEGER)
        		ELSE NULL
      		END IS NOT NULL
  		AND CAST(nursingchartvalue AS INTEGER) > 0
  		AND CAST(nursingchartvalue AS INTEGER) <= 100

	UNION ALL

	-- fraction of inspired oxygen (fiO2) outside of [.2, .22] and [20, 22]
	-- indicates oxygen therapy.
	SELECT 
    patientunitstayid AS icustay_id,
    respchartoffset AS charttime,
    CASE
        WHEN respchartvalue ~ '^[0-9]*(\.[0-9]+)?%?$' THEN
            CASE
                WHEN CAST(REGEXP_REPLACE(respchartvalue, '[^0-9.]', '', 'g') AS FLOAT) <= 1 AND 
                     CAST(REGEXP_REPLACE(respchartvalue, '[^0-9.]', '', 'g') AS FLOAT) > 0.22 THEN -1
                WHEN CAST(REGEXP_REPLACE(respchartvalue, '[^0-9.]', '', 'g') AS FLOAT) > 22 THEN -1
                ELSE 0
            END
        ELSE NULL
    END AS oxygen_therapy_type,
    NULL AS activeUponDischarge
	FROM 
    	respchart
	WHERE 
    	respchartoffset >= -60
    	AND LOWER(respchartvaluelabel) IN ('fio2', 'fio2 (%)')

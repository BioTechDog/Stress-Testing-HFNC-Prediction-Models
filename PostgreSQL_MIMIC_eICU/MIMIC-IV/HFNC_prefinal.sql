DROP TABLE IF EXISTS hfnc.pre_final; 
CREATE TABLE hfnc.pre_final AS

WITH resp AS (
	SELECT 
    pi.age, 
    pi.height, 
    pi.weight,
	pi.subject_id,
	vs.*
FROM 
    hfnc.hfnc_bg_vs vs
LEFT JOIN 
    hfnc.hfnc_id pi
ON 
    vs.stay_id = pi.stay_id
),

death as (SELECT 
    stay_id,
    MAX(EXTRACT(DAY FROM (dod - admittime))) AS survive_days_from_admit,
    MAX(EXTRACT(DAY FROM (dod - icu_intime))) AS survive_days_from_icu
FROM 
    mimiciv_derived.icustay_detail
GROUP BY 
    stay_id)

SELECT 
     resp.*,
	 status.intubation_status,
	 d.survive_days_from_admit,
	 d.survive_days_from_icu
-- DISTINCT stay_id
FROM 
    resp
LEFT JOIN 
    hfnc.intubation_status status
ON 
    resp.stay_id = status.stay_id

LEFT JOIN death d ON d.stay_id = resp.stay_id
WHERE 
	resp.bg_charttime BETWEEN resp.starttime AND (resp.starttime + INTERVAL '4 hours'); --(0, 2.5)



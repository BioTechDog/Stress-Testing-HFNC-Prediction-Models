-- DROP TABLE IF EXISTS hfnc.lab_data; 
-- CREATE TABLE hfnc.lab_data AS

WITH ranked_therapy AS (
    SELECT icustay_id, vent_start, vent_end, oxygen_therapy_type,
           ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY vent_start) AS rn,
           LAST_VALUE(oxygen_therapy_type) OVER (PARTITION BY icustay_id ORDER BY vent_start
                                                  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_oxygen_therapy_type
    FROM hfnc.oxygen_therapy WHERE icustay_id in (SELECT icustay_id FROM hfnc.hfnc_id)
),

first_occurrence AS (
    SELECT t1.icustay_id, t1.vent_start, t1.vent_end, t1.oxygen_therapy_type, 
	t2.oxygen_therapy_type AS second_oxygen_therapy_type,
	t1.last_oxygen_therapy_type
    FROM 
    	ranked_therapy t1
	LEFT JOIN 
    	ranked_therapy t2 ON t1.icustay_id = t2.icustay_id AND t2.rn = 2
	WHERE 
    	t1.rn = 1
),

lab_data_t0 AS (
    SELECT
        fo.icustay_id,
        fo.vent_start,
        fo.vent_end,
        fo.oxygen_therapy_type,
		fo.second_oxygen_therapy_type,
		fo.last_oxygen_therapy_type,
        labresultoffset AS labtime_t0,
        MAX(CASE WHEN lab.labname = 'paO2' THEN lab.labresult END) AS paO2_t0,
        MAX(CASE WHEN lab.labname = 'FiO2' THEN lab.labresult END) AS FiO2_t0,
        MAX(CASE WHEN lab.labname = 'paCO2' THEN lab.labresult END) AS paCO2_t0,
        MAX(CASE WHEN lab.labname = 'pH' THEN lab.labresult END) AS pH_t0,
        MAX(CASE WHEN lab.labname = 'O2 Sat (%)' THEN lab.labresult END) AS O2Sat_t0
    FROM eicuii.lab lab
    JOIN first_occurrence fo ON lab.patientunitstayid = fo.icustay_id
    WHERE lab.labresultoffset < fo.vent_start + 0.1*60
      AND lab.labresultoffset > fo.vent_start - 6*60
      AND lab.labname IN ('paO2', 'FiO2', 'paCO2', 'pH', 'O2 Sat (%)')
    GROUP BY fo.icustay_id, fo.vent_start,fo.vent_end,fo.oxygen_therapy_type,labtime_t0,second_oxygen_therapy_type, last_oxygen_therapy_type
),

lab_data_t1 AS (
    SELECT
        fo.icustay_id,
		labresultoffset AS labtime_t1,
        MAX(CASE WHEN lab.labname = 'paO2' THEN lab.labresult END) AS paO2_t1,
        MAX(CASE WHEN lab.labname = 'FiO2' THEN lab.labresult END) AS FiO2_t1,
        MAX(CASE WHEN lab.labname = 'paCO2' THEN lab.labresult END) AS paCO2_t1,
        MAX(CASE WHEN lab.labname = 'pH' THEN lab.labresult END) AS pH_t1,
        MAX(CASE WHEN lab.labname = 'O2 Sat (%)' THEN lab.labresult END) AS O2Sat_t1
    FROM eicuii.lab lab
    JOIN first_occurrence fo ON lab.patientunitstayid = fo.icustay_id
    WHERE lab.labresultoffset < fo.vent_start + 3*60
      AND lab.labresultoffset > fo.vent_start + 1*60
      AND lab.labname IN ('paO2', 'FiO2', 'paCO2', 'pH', 'O2 Sat (%)')
    GROUP BY fo.icustay_id, labtime_t1
),

nursecharting_data_t0 AS (
    SELECT
        fo.icustay_id,
		nursingchartoffset AS nursingcharttime_t0,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Heart Rate' THEN nc.nursingchartvalue END) AS HR_t0,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Respiratory Rate' THEN nc.nursingchartvalue END) AS RR_t0
    FROM eicuii.nursecharting nc
    JOIN first_occurrence fo ON nc.patientunitstayid = fo.icustay_id
    WHERE nc.nursingchartoffset < fo.vent_start + 0.1*60 
      AND nc.nursingchartoffset > fo.vent_start - 6*60 
      AND nc.nursingchartcelltypevallabel IN ('Heart Rate', 'Respiratory Rate')
    GROUP BY fo.icustay_id,nursingcharttime_t0
),

nursecharting_data_t1 AS (
    SELECT
        fo.icustay_id,
		nursingchartoffset AS nursingcharttime_t1,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Heart Rate' THEN nc.nursingchartvalue END) AS HR_t1,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Respiratory Rate' THEN nc.nursingchartvalue END) AS RR_t1
    FROM eicuii.nursecharting nc
    JOIN first_occurrence fo ON nc.patientunitstayid = fo.icustay_id
    WHERE nc.nursingchartoffset < fo.vent_start + 3*60 
      AND nc.nursingchartoffset > fo.vent_start + 1*60 
      AND nc.nursingchartcelltypevallabel IN ('Heart Rate', 'Respiratory Rate')
    GROUP BY fo.icustay_id, nursingcharttime_t1
)

SELECT 
    t0.icustay_id, t0.vent_start, t0.vent_end, t0.oxygen_therapy_type,
    t0.labtime_t0, t0.paO2_t0, t0.FiO2_t0, t0.paCO2_t0, t0.pH_t0, t0.O2Sat_t0,
    t1.labtime_t1, t1.paO2_t1, t1.FiO2_t1, t1.paCO2_t1, t1.pH_t1, t1.O2Sat_t1, 
	nct0.nursingcharttime_t0, nct0.HR_t0, nct0.RR_t0, nct1.nursingcharttime_t1, nct1.HR_t1, nct1.RR_t1, 
	t0.second_oxygen_therapy_type, t0.last_oxygen_therapy_type
FROM lab_data_t0 t0
JOIN lab_data_t1 t1 ON t0.icustay_id = t1.icustay_id
JOIN nursecharting_data_t0 nct0 ON t0.icustay_id = nct0.icustay_id
JOIN nursecharting_data_t1 nct1 ON t0.icustay_id = nct1.icustay_id
WHERE t0.paO2_t0 IS NOT NULL 
  AND t0.FiO2_t0 IS NOT NULL
  AND t1.paO2_t1 IS NOT NULL
  AND t1.FiO2_t1 IS NOT NULL
  AND nct0.RR_t0 IS NOT NULL
  AND nct1.RR_t1 IS NOT NULL;
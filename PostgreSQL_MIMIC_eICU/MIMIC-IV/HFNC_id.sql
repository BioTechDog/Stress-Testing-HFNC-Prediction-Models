DROP TABLE IF EXISTS HFNC.hfnc_id; 
CREATE TABLE HFNC.hfnc_id AS

WITH ventilation_id AS (
  SELECT f.stay_id, d.subject_id 
  FROM mimiciv_derived.ventilation f
  JOIN mimiciv_derived.first_day_bg d ON f.stay_id = d.stay_id
) 
SELECT 
  v.stay_id,
  v.subject_id,
  MAX(s.age) as age,
  MAX(he.height) as height,
  AVG(we.weight) as weight,
  MAX(s.admittime) as admittime
  
FROM mimiciv_derived.ventilation ve 
RIGHT JOIN ventilation_id v ON v.stay_id = ve.stay_id
LEFT JOIN mimiciv_derived.age s ON v.subject_id = s.subject_id
LEFT JOIN mimiciv_derived.height he ON v.stay_id = he.stay_id
LEFT JOIN mimiciv_derived.weight_durations we ON v.stay_id = we.stay_id and we.weight_type = 'daily'
WHERE v.stay_id IN (SELECT DISTINCT stay_id FROM mimiciv_derived.ventilation WHERE ventilation_status IN ('HFNC'))
GROUP BY v.stay_id, v.subject_id
ORDER BY v.stay_id, v.subject_id



-- Create the new table with stay_id and intubation_status
DROP TABLE IF EXISTS hfnc.intubation_status; 
CREATE TABLE hfnc.intubation_status AS
WITH intubation_status AS (
    SELECT
        stay_id,
        CASE
            WHEN MAX(CASE WHEN ventilation_status IN ('Tracheostomy', 'InvasiveVent', 'NonInvasiveVent') THEN 1 ELSE 0 END) OVER (PARTITION BY stay_id) = 1
            THEN 'intubated'
            ELSE 'not intubated'
        END AS intubation_status
    FROM
        (SELECT
        stay_id,
        starttime,
        ventilation_status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
		FROM  mimiciv_derived.ventilation ve where ve.stay_id IN (SELECT stay_id FROM hfnc.hfnc_nopriointu_id))
)
SELECT DISTINCT
    stay_id,
    intubation_status
FROM
    intubation_status;
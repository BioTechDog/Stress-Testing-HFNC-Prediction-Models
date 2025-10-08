DROP TABLE IF EXISTS hfnc.hfnc_nopriointu_id; 
CREATE TABLE hfnc.hfnc_nopriointu_id AS

WITH VentilationStatus AS (
    SELECT stay_id, starttime, ventilation_status,
           ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY starttime) AS row_num
    FROM mimiciv_derived.ventilation
),
HFNCVent AS (
    SELECT stay_id, row_num
    FROM VentilationStatus
    WHERE ventilation_status = 'HFNC'
),
PriorStatus AS (
    SELECT DISTINCT v1.stay_id
    FROM VentilationStatus v1
    JOIN HFNCVent nv ON v1.stay_id = nv.stay_id AND v1.row_num < nv.row_num
    WHERE v1.ventilation_status IN ('Tracheostomy', 'InvasiveVent', 'NonInvasiveVent')
)
SELECT DISTINCT nv.stay_id
FROM HFNCVent nv
WHERE nv.stay_id NOT IN (SELECT stay_id FROM PriorStatus);


DROP TABLE IF EXISTS hfnc.hfnc_id; 
CREATE TABLE hfnc.hfnc_id AS

WITH oxygen_therapy_666 AS (
    SELECT
        *
    FROM
        hfnc.oxygen_therapy
    WHERE
        oxygen_therapy_type = 666
),
exclusion_list AS (
    SELECT
        DISTINCT t1.icustay_id
    FROM
        hfnc.oxygen_therapy t1
    JOIN
        oxygen_therapy_666 t2
    ON
        t1.icustay_id = t2.icustay_id
    WHERE
        t1.oxygen_therapy_type IN (2, 3, 4)
        AND t1.vent_start < t2.vent_start
)
SELECT 
    *
FROM
    oxygen_therapy_666
WHERE
    icustay_id NOT IN (SELECT icustay_id FROM exclusion_list);
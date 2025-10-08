DROP TABLE IF EXISTS hfnc.t1; 
CREATE TABLE hfnc.t1 AS

WITH filtered_data AS (
    SELECT 
        p.*
    FROM 
        hfnc.pre_final p
    WHERE 
        bg_charttime BETWEEN (starttime + INTERVAL '1 hour') AND (starttime + INTERVAL '3 hours')
        AND po2 IS NOT NULL
        AND pco2 IS NOT NULL
--         AND peep_value IS NOT NULL
--         AND psv IS NOT NULL
),
ranked_data AS (
    SELECT
        fd.*,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY bg_charttime) AS row_num
    FROM
        filtered_data fd
)

SELECT 
    rd.*
FROM 
    ranked_data rd
WHERE 
    rd.row_num = 1;
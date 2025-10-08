DROP TABLE IF EXISTS hfnc.hfnc_bg_vs; 
CREATE TABLE hfnc.hfnc_bg_vs AS

SELECT DISTINCT
    bg.*,
    vitalsign.charttime AS vitalsign_charttime,
    vitalsign.heart_rate,
    vitalsign.resp_rate,
    vitalsign.spo2
FROM 
    (SELECT DISTINCT *
     FROM hfnc.hfnc_bg 
     WHERE ventilation_status = 'HFNC') bg
--      AND psv IS NOT NULL) bgpsv
LEFT JOIN 
    mimiciv_derived.vitalsign vitalsign
ON 
    bg.stay_id = vitalsign.stay_id
    AND vitalsign.charttime BETWEEN (bg.starttime-INTERVAL '2 hour') AND bg.endtime
	
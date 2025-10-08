DROP TABLE IF EXISTS hfnc.hfnc_bg; 
CREATE TABLE hfnc.hfnc_bg AS

SELECT DISTINCT
    v.stay_id, 
    v.ventilation_status, 
    v.starttime, 
    v.endtime,
	bg.charttime AS bg_charttime,
    bg.so2,
    bg.po2,
    bg.pco2,
    bg.fio2,
    bg.fio2_chartevents,
    bg.pao2fio2ratio,
    bg.ph,
    bg.baseexcess,
    bg.hemoglobin,
    bg.lactate
FROM 
    hfnc.hfnc_id hf
LEFT JOIN 
    mimiciv_derived.bg bg
ON 
    hf.subject_id = bg.subject_id
LEFT JOIN
	mimiciv_derived.ventilation v
ON 
    v.stay_id = hf.stay_id AND v.stay_id IN (SELECT stay_id FROM hfnc.hfnc_nopriointu_id)
WHERE 
    v.stay_id IN (SELECT stay_id FROM hfnc.hfnc_nopriointu_id)
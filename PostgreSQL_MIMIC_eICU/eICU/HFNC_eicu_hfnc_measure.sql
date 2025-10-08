WITH selected AS (
    SELECT icustay_id, vent_start, vent_end, oxygen_therapy_type,
           ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY vent_start) AS rn,
           LAST_VALUE(oxygen_therapy_type) OVER (PARTITION BY icustay_id ORDER BY vent_start
                                                  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_oxygen_therapy_type
    FROM hfnc.oxygen_therapy WHERE icustay_id in (SELECT icustay_id FROM hfnc.hfnc_id)
),

first_occurrence AS (
    SELECT icustay_id, vent_start, vent_end, oxygen_therapy_type, last_oxygen_therapy_type
    FROM selected
    WHERE rn = 1
),

lab_data AS (
    SELECT
        fo.icustay_id,
        fo.vent_start,
        fo.vent_end,
        fo.oxygen_therapy_type,
        fo.last_oxygen_therapy_type,
        labresultoffset AS labtime,
        labname,
        labresult
    FROM eicuii.lab lab
    JOIN first_occurrence fo ON lab.patientunitstayid = fo.icustay_id
    WHERE lab.labresultoffset < fo.vent_start + 0.5*60
        AND lab.labresultoffset > fo.vent_start - 1*60
        AND lab.labname IN ('paO2', 'paCO2', 'pH', 'FiO2', 'O2 Sat (%)', 'creatinine', 'lactate')
		AND lab.patientunitstayid IN (SELECT icustay_id from selected)
),

nursecharting_data AS (
    SELECT
        fo.icustay_id,
        fo.vent_start,
        fo.vent_end,
        fo.oxygen_therapy_type,
        fo.last_oxygen_therapy_type,
        nc.nursingchartoffset AS charttime,
        nc.nursingchartcelltypevallabel AS celllabel,
        nc.nursingchartvalue AS cellvalue
    FROM eicuii.nursecharting nc
    JOIN first_occurrence fo ON nc.patientunitstayid = fo.icustay_id
    WHERE nc.nursingchartoffset < fo.vent_start + 0.5*60
        AND nc.nursingchartoffset > fo.vent_start - 1*60
        AND nc.nursingchartcelltypecat IN ('Respiratory', 'Vital Signs')
),

respiratorycharting_data AS (
    SELECT
        fo.icustay_id,
        fo.vent_start,
        fo.vent_end,
        fo.oxygen_therapy_type,
        fo.last_oxygen_therapy_type,
        rc.respchartvaluelabel AS resplabel,
        rc.respchartvalue AS respvalue,
        rc.respchartoffset AS respcharttime
    FROM eicuii.respiratorycharting rc
    JOIN first_occurrence fo ON rc.patientunitstayid = fo.icustay_id
    WHERE rc.respchartoffset < fo.vent_start + 0.5*60
        AND rc.respchartoffset > fo.vent_start - 1*60
		AND rc.respchartvaluelabel IN ('FiO2', 'Vent Rate')
),

ranked_data AS (
    SELECT
        ld.icustay_id,
        ld.vent_start,
        ld.vent_end,
        ld.oxygen_therapy_type,
        ld.labtime,
        ld.last_oxygen_therapy_type,
        MAX(CASE WHEN labname = 'paO2' THEN labresult END) AS paO2,
        MAX(CASE WHEN labname = 'paCO2' THEN labresult END) AS paCO2,
        MAX(CASE WHEN labname = 'pH' THEN labresult END) AS pH,
        MAX(CASE WHEN labname = 'FiO2' THEN labresult END) AS FiO2,
        MAX(CASE WHEN labname = 'O2 Sat (%)' THEN labresult END) AS SpO2,
        MAX(CASE WHEN labname = 'creatinine' THEN labresult END) AS creatinine,
        MAX(CASE WHEN labname = 'lactate' THEN labresult END) AS lactate,
        nc.charttime AS nursingcharttime,
        MAX(CASE WHEN nc.celllabel = 'Respiratory Rate' THEN nc.cellvalue END) AS respiratory_rate,
        MAX(CASE WHEN nc.celllabel = 'Heart Rate' THEN nc.cellvalue END) AS heart_rate,
        MAX(CASE WHEN nc.celllabel LIKE 'O2 L/%' THEN nc.cellvalue END) AS o2_lpm,
        rc.respcharttime AS respcharttime,
        MAX(CASE WHEN rc.resplabel = 'FiO2' THEN rc.respvalue END) AS resp_fio2,
        MAX(CASE WHEN rc.resplabel = 'Vent Rate' THEN rc.respvalue END) AS vent_rate,
        apr.actualhospitalmortality,
        pat.hospitaldischargeoffset AS time_of_death,
        CASE
            WHEN apr.actualhospitalmortality = 'EXPIRED' THEN (pat.hospitaldischargeoffset - ld.vent_start) / 60
            ELSE NULL
        END AS death_hours,
        RANK() OVER (PARTITION BY ld.icustay_id ORDER BY ABS(nc.charttime - ld.vent_start)) AS charttime_rank,
        RANK() OVER (PARTITION BY ld.icustay_id ORDER BY ABS(rc.respcharttime - ld.vent_start)) AS respcharttime_rank
    FROM lab_data ld
    JOIN nursecharting_data nc ON ld.icustay_id = nc.icustay_id
        AND ld.vent_start = nc.vent_start
        AND ld.vent_end = nc.vent_end
        AND ld.oxygen_therapy_type = nc.oxygen_therapy_type
        AND ld.last_oxygen_therapy_type = nc.last_oxygen_therapy_type
    JOIN respiratorycharting_data rc ON ld.icustay_id = rc.icustay_id
        AND ld.vent_start = rc.vent_start
        AND ld.vent_end = rc.vent_end
        AND ld.oxygen_therapy_type = rc.oxygen_therapy_type
        AND ld.last_oxygen_therapy_type = rc.last_oxygen_therapy_type
    JOIN eicuii.apachePatientResult apr ON ld.icustay_id = apr.patientunitstayid
    JOIN eicuii.patient pat ON ld.icustay_id = pat.patientunitstayid
    GROUP BY ld.icustay_id, ld.vent_start, ld.vent_end, ld.oxygen_therapy_type, ld.labtime, ld.last_oxygen_therapy_type, apr.actualhospitalmortality, pat.hospitaldischargeoffset, nc.charttime, rc.respcharttime
)
SELECT
    icustay_id,
    vent_start,
    vent_end,
    oxygen_therapy_type,
    labtime,
    last_oxygen_therapy_type,
    paO2,
    paCO2,
    pH,
    FiO2,
    SpO2,
    creatinine,
    lactate,
    nursingcharttime,
    respiratory_rate,
    heart_rate,
    o2_lpm,
    respcharttime,
    resp_fio2,
    vent_rate,
    actualhospitalmortality,
    time_of_death,
    death_hours
FROM ranked_data
WHERE charttime_rank = 1 AND respcharttime_rank = 1;
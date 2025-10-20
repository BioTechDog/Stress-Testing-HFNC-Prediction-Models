WITH bg_filtered AS (
    SELECT
        t1.stay_id,
        t1.subject_id,
        t1.starttime,
        bg.charttime AS bg_charttime,
        bg.po2,
		bg.pco2,
        bg.fio2,
        bg.fio2_chartevents,
        bg.pao2fio2ratio,
        bg.ph,
        bg.baseexcess,
        t1.intubation_status,
        ABS(EXTRACT(EPOCH FROM (bg.charttime - t1.starttime))) AS time_diff
    FROM
        mimiciv_derived.bg bg
    JOIN
        hfnc.t1 t1
    ON
        bg.subject_id = t1.subject_id
    WHERE
        bg.charttime BETWEEN (t1.starttime - INTERVAL '6 hour') AND (t1.starttime + INTERVAL '0.1 hour')
),
vitalsign_filtered AS (
    SELECT
        vs.stay_id,
        vs.charttime AS vitalsign_charttime,
        vs.heart_rate,
        vs.resp_rate,
        vs.spo2,
        t1.starttime,
        t1.intubation_status,
		t1.subject_id,
        ABS(EXTRACT(EPOCH FROM (vs.charttime - t1.starttime))) AS time_diff
    FROM
        mimiciv_derived.vitalsign vs
    JOIN
        hfnc.t1 t1
    ON
        vs.stay_id = t1.stay_id
    WHERE
        vs.charttime BETWEEN (t1.starttime - INTERVAL '6 hour') AND (t1.starttime + INTERVAL '0.1 hour')
),
bg_closest AS (
    SELECT
        stay_id,
		subject_id,
        bg_charttime,
        po2,
		pco2,
        fio2,
        fio2_chartevents,
        pao2fio2ratio,
        ph,
        baseexcess,
        starttime,
        intubation_status
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY time_diff) AS rn
        FROM
            bg_filtered
    ) sub
    WHERE
        rn = 1
),
vitalsign_closest AS (
    SELECT
        stay_id,
        vitalsign_charttime,
        heart_rate,
        resp_rate,
        spo2,
        starttime,
        intubation_status
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY time_diff) AS rn
        FROM
            vitalsign_filtered
    ) sub
    WHERE
        rn = 1
),

sapsii AS (
    SELECT
        t1.stay_id,
        t1.subject_id,
        t1.starttime,
        sap.starttime sapsii_start,
		sap.endtime sapsii_end,
		sap.sapsii
    FROM
        mimiciv_derived.sapsii sap
    JOIN
        niv.t1 t1
    ON
        sap.subject_id = t1.subject_id
    WHERE
        sap.starttime BETWEEN (t1.starttime - INTERVAL '6 hour') AND (t1.starttime + INTERVAL '0.1 hour') AND sap.subject_id = 18588165
),

flow AS (
    SELECT DISTINCT
    t1.stay_id,
    t1.subject_id,
    t1.starttime,
    f.flow_rate
	FROM
		mimiciv_derived.ventilator_setting f
	JOIN
		niv.t1 t1
	ON
		f.subject_id = t1.subject_id
	WHERE
		f.flow_rate IS NOT NULL
-- 		AND f.charttime BETWEEN (t1.starttime - INTERVAL '6 hour') AND (t1.starttime + INTERVAL '0.1 hour')
)

SELECT
DISTINCT
    bg_closest.stay_id,
    bg_closest.starttime,
    bg_closest.bg_charttime,
    bg_closest.po2,
	bg_closest.pco2,
    bg_closest.fio2,
    bg_closest.fio2_chartevents,
    bg_closest.pao2fio2ratio,
    bg_closest.ph,
    bg_closest.baseexcess,
    vitalsign_closest.vitalsign_charttime,
    vitalsign_closest.heart_rate,
    vitalsign_closest.resp_rate,
    vitalsign_closest.spo2,
    bg_closest.intubation_status,
	sapsii.sapsii_start,
	sapsii.sapsii,
	flow.flow_rate
FROM
    bg_closest
LEFT JOIN
    vitalsign_closest ON bg_closest.stay_id = vitalsign_closest.stay_id AND bg_closest.starttime = vitalsign_closest.starttime
LEfT JOIN
	sapsii ON bg_closest.stay_id = sapsii.stay_id
LEFT JOIN
	flow ON bg_closest.stay_id = flow.stay_id AND bg_closest.subject_id = flow.subject_id
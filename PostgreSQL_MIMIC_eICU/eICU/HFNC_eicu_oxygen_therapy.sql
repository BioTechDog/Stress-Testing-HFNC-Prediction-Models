DROP TABLE IF EXISTS hfnc.oxygen_therapy; 
CREATE TABLE hfnc.oxygen_therapy AS

WITH respchart AS (
	SELECT *
	FROM eicuii.respiratorycharting
)

, nursechart AS (
	SELECT *
	FROM eicuii.nursecharting
)

, pat AS (
	SELECT *
	FROM eicuii.patient
)


-- Extract the type of oxygen therapy.
-- The categories are invasive ventilation,
-- noninvasive ventilation, and supplemental oxygen.
-- `oxygen_therapy_type = -1` indicates oxygen therapy,
-- i.e. more oxygen than in room air is administered.
, ventsettings0 AS (
	SELECT * FROM hfnc.ventsettings0
)


-- Ensure charttime is unique
, ventsettings AS (
	SELECT icustay_id
		, charttime
		, MAX(oxygen_therapy_type) AS oxygen_therapy_type
		, MAX(activeUponDischarge) AS activeUponDischarge
		, COUNT(CASE WHEN oxygen_therapy_type = -1 THEN 1 END) > 0 AS supp_oxygen
	FROM ventsettings0
	-- If oxygen_therapy_type is NULL,
	-- then the record does not correspond with oxygen therapy.
	WHERE oxygen_therapy_type IS NOT NULL
	GROUP BY icustay_id, charttime
)


, vd0 as
(
  select
    *
    -- this carries over the previous charttime which had an oxygen therapy event
    , LAG(CHARTTIME, 1) OVER (partition by icustay_id order by charttime)
	as charttime_lag
  from ventsettings 
)
, vd1 as
(
  select
      icustay_id
      , charttime
      , oxygen_therapy_type
      , activeUponDischarge
      , supp_oxygen

      -- If the time since the last oxygen therapy event is more than 24 hours,
	-- we consider that ventilation had ended in between.
	-- That is, the next ventilation record corresponds to a new ventilation session.
      , CASE
		WHEN charttime - charttime_lag > 24*60 THEN 1
		WHEN charttime_lag IS NULL THEN 1 -- No lag can be computed for the very first record
		ELSE 0
	END AS newvent
  -- use the staging table with only oxygen therapy records from chart events
  FROM vd0
)
, vd2 as
(
  select vd1.*
  -- create a cumulative sum of the instances of new ventilation
  -- this results in a monotonic integer assigned to each instance of ventilation
  , SUM( newvent )
      OVER ( partition by icustay_id order by charttime )
    as ventnum
  from vd1
)

--- now we convert CHARTTIME of ventilator settings into durations
-- create the durations for each oxygen therapy instance
-- We only keep the first oxygen therapy instance
, vd3 AS
(
	SELECT 
    icustay_id,
    ventnum,
		CASE
			-- If activeUponDischarge, then the unit discharge time is vent_end
			WHEN (
				MAX(CASE WHEN activeUponDischarge = 'True' THEN 1 ELSE 0 END) = 1
				-- vent_end cannot be later than the unit discharge time.
				-- However, unitdischargeoffset often seems too low.
				-- So, we only use it if it yields an extension of the
				-- ventilation time from ventsettings.
				AND MAX(charttime) + 60 < MAX(pat.unitdischargeoffset::int)
			)
			THEN MAX(pat.unitdischargeoffset::int)
			-- End time is currently a charting time
			-- Since these are usually recorded hourly, ventilation is actually longer.
			-- We therefore add 60 minutes to the last time.
			ELSE MAX(charttime) + 60
		END AS vent_end,
		MIN(charttime) AS vent_start,
		MAX(CAST(oxygen_therapy_type AS INT)) AS oxygen_therapy_type,
		MAX(CAST(supp_oxygen AS INT)) AS supp_oxygen
	FROM 
		vd2
	LEFT JOIN 
		pat ON vd2.icustay_id = pat.patientunitstayid
	GROUP BY 
		icustay_id, ventnum
)


select vd3.*
	-- vent_duration is in hours.
	, (vent_end - vent_start) / 60 AS vent_duration
	, MIN(vent_start) OVER(PARTITION BY icustay_id) AS vent_start_first
from vd3

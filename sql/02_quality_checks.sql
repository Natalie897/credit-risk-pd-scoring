SELECT COUNT(*) AS row_count,
       COUNT(DISTINCT application_id) AS unique_applications,
       SUM(CASE WHEN default_12m NOT IN (0,1) OR default_12m IS NULL THEN 1 ELSE 0 END) AS invalid_targets,
       SUM(CASE WHEN debt_to_income NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS invalid_dti,
       ROUND(AVG(default_12m),4) AS default_rate
FROM feature_store_applications;

SELECT SUBSTR(application_date,1,4) AS application_year,
       customer_segment,
       COUNT(*) AS applications,
       ROUND(AVG(default_12m),4) AS default_rate
FROM feature_store_applications
GROUP BY application_year, customer_segment
ORDER BY application_year, customer_segment;

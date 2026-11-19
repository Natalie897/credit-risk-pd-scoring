-- All predictors are known at the time of the application. No target leakage.
DROP TABLE IF EXISTS feature_store_applications;
CREATE TABLE feature_store_applications AS
SELECT
    application_id, application_date, age, annual_income, employment_years,
    debt_to_income, credit_utilization, delinquencies_12m, loan_amount,
    loan_term_months, region, channel, customer_segment,
    CASE WHEN annual_income > 0 THEN loan_amount / annual_income ELSE NULL END AS loan_to_income,
    CASE WHEN delinquencies_12m > 0 THEN 1 ELSE 0 END AS has_delinquency,
    default_12m
FROM raw_applications;
CREATE UNIQUE INDEX idx_feature_application ON feature_store_applications(application_id);
CREATE INDEX idx_feature_date ON feature_store_applications(application_date);

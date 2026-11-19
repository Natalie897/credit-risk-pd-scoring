"""Generate synthetic loan-application snapshots with a 12-month default label."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_applications(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    if n < 100:
        raise ValueError("n must be >= 100")
    rng = np.random.default_rng(seed)
    application_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1096, size=n), unit="D")
    income = np.rint(rng.lognormal(np.log(88000), .53, n)).astype(float)
    income = np.clip(income, 18000, 650000)
    age = rng.integers(21, 66, n)
    experience = np.maximum(0, age - rng.integers(19, 37, n)).astype(float)
    dti = np.clip(rng.beta(2.1, 5.4, n) * 1.12, .01, .94)
    utilization = np.clip(rng.beta(2.2, 2.6, n), .01, .99)
    delinquencies = rng.poisson(.35 + 1.4 * (dti > .5), n)
    loan_amount = np.rint(rng.uniform(15000, 180000, n)).astype(float)
    loan_term = rng.choice([12, 24, 36, 48, 60], n, p=[.12, .23, .35, .18, .12])
    region = rng.choice(['Central','North','South','East'], n, p=[.32,.22,.27,.19])
    channel = rng.choice(['online','branch','partner'], n, p=[.55,.30,.15])
    segment = rng.choice(['individual','small_business'], n, p=[.8,.2])
    # Data-generating mechanism is synthetic, not an empirical banking PD model.
    logit = (-2.65 + 2.7 * (dti - .3) + 1.8 * (utilization - .5)
             + .58 * delinquencies + .22 * (loan_amount / np.maximum(income, 1) - 1)
             - .012 * experience + .19 * (channel == 'partner')
             + .12 * (segment == 'small_business'))
    probability = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
    label = rng.binomial(1, probability)
    data = pd.DataFrame({
        'application_id': [f'APP{i:07d}' for i in range(1, n+1)],
        'application_date': application_date.strftime('%Y-%m-%d'),
        'age': age, 'annual_income': income, 'employment_years': experience,
        'debt_to_income': dti, 'credit_utilization': utilization,
        'delinquencies_12m': delinquencies, 'loan_amount': loan_amount,
        'loan_term_months': loan_term, 'region': region, 'channel': channel,
        'customer_segment': segment, 'default_12m': label,
    })
    # Missing predictor values, never in labels or application IDs.
    for column, rate in [('annual_income', .035), ('employment_years', .05), ('credit_utilization', .025)]:
        data.loc[rng.random(n) < rate, column] = np.nan
    return data.sort_values(['application_date','application_id']).reset_index(drop=True)


def save_applications(path: str | Path, n: int = 6000, seed: int = 42) -> pd.DataFrame:
    frame = generate_applications(n, seed)
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame

"""Exploratory statistics and basic feature drift on synthetic loan applications."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def population_stability_index(reference, current, bins=10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]; cur = cur[np.isfinite(cur)]
    if len(ref)<bins or len(cur)<bins:
        raise ValueError('Not enough samples for drift estimate')
    edges = np.unique(np.quantile(ref, np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0] = -np.inf; edges[-1] = np.inf
    a=np.histogram(ref,bins=edges)[0]; b=np.histogram(cur,bins=edges)[0]
    p=np.maximum(a/len(ref),1e-5);q=np.maximum(b/len(cur),1e-5)
    return float(np.sum((p-q)*np.log(p/q)))


def describe_data(frame:pd.DataFrame) -> dict:
    train=frame[frame.application_date<'2024-01-01']
    later=frame[frame.application_date>='2025-01-01']
    income=frame[['annual_income','default_12m']].dropna()
    yes=income.loc[income.default_12m==1,'annual_income']
    no=income.loc[income.default_12m==0,'annual_income']
    stat,p_value=mannwhitneyu(yes,no,alternative='two-sided')
    rho,rho_p=spearmanr(frame['delinquencies_12m'],frame['default_12m'])
    return {
      'synthetic_data':True,
      'n_applications':len(frame),
      'default_rate':float(frame.default_12m.mean()),
      'missing_rates':{column:float(frame[column].isna().mean()) for column in ['annual_income','employment_years','credit_utilization']},
      'mann_whitney_income_by_target':{'statistic':float(stat),'p_value':float(p_value)},
      'spearman_delinquencies_default':{'rho':float(rho),'p_value':float(rho_p)},
      'psi_2023_to_2025':{
          'debt_to_income':population_stability_index(train.debt_to_income,later.debt_to_income),
          'credit_utilization':population_stability_index(train.credit_utilization.dropna(),later.credit_utilization.dropna())
      },
      'notes':'These tests characterize simulated data. p-values and PSI do not validate real bank relationships.'
    }

import sqlite3
import numpy as np
import pandas as pd
import pytest
from credit_risk.generate import generate_applications
from credit_risk.mart import build_mart
from credit_risk.model import split_by_time, metrics, run_experiment,FEATURES

@pytest.fixture(scope='module')
def sample(): return generate_applications(900,42)

def test_deterministic(sample):
    pd.testing.assert_frame_equal(sample,generate_applications(900,42))

def test_unique_ids(sample): assert sample.application_id.is_unique

def test_binary_label(sample): assert set(sample.default_12m.unique())=={0,1}

def test_dates(sample): assert sample.application_date.min()>='2023-01-01' and sample.application_date.max()<='2025-12-31'

def test_years(sample):
    assert set(sample.application_date.str[:4])=={'2023','2024','2025'}

def test_positive_income(sample): assert sample.annual_income.dropna().gt(0).all()

def test_valid_dti(sample): assert sample.debt_to_income.between(0,1).all()

def test_missing_predictors(sample): assert sample.annual_income.isna().any()

def test_mart_reconciliation(tmp_path,sample):
    mart=build_mart(sample,tmp_path/'a.sqlite')
    assert len(mart)==len(sample) and mart.default_12m.sum()==sample.default_12m.sum()

def test_mart_query(tmp_path,sample):
    db=tmp_path/'a.sqlite';build_mart(sample,db)
    with sqlite3.connect(db) as c:
        assert c.execute('select count(*) from feature_store_applications').fetchone()[0]==len(sample)

def test_derived_features(tmp_path,sample):
    m=build_mart(sample,tmp_path/'a.sqlite')
    assert set(m.has_delinquency.unique())<={0,1}
    row=m[m.annual_income.notna()].iloc[0]
    assert abs(row.loan_to_income-row.loan_amount/row.annual_income)<1e-10

def test_duplicates_error(tmp_path,sample):
    d=pd.concat([sample,sample.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='duplicate'):build_mart(d,tmp_path/'x.sqlite')

def test_target_error(tmp_path,sample):
    d=sample.copy(); d.loc[0,'default_12m']=4
    with pytest.raises(ValueError,match='default_12m'):build_mart(d,tmp_path/'x.sqlite')

def test_train_val_test_order(tmp_path,sample):
    m=build_mart(sample,tmp_path/'a.sqlite'); a,b,c=split_by_time(m)
    assert a.application_date.max()<'2024-01-01'
    assert b.application_date.min()>='2024-01-01' and b.application_date.max()<'2025-01-01'
    assert c.application_date.min()>='2025-01-01'
    assert len(a)+len(b)+len(c)==len(sample)

def test_feature_leakage_prevented():
    assert 'default_12m' not in FEATURES and 'application_id' not in FEATURES

def test_roc_gini_identity():
    m=metrics([0,0,1,1],[.2,.4,.6,.8])
    assert m['roc_auc']==1 and m['gini']==1

def test_brier_perfect(): assert metrics([0,1],[0,1])['brier']<1e-12

def test_metrics_invalid():
    with pytest.raises(ValueError):metrics([1,1],[.3,.4])

def test_small_n_error():
    with pytest.raises(ValueError):generate_applications(10)

def test_full_experiment(tmp_path,sample):
    m=build_mart(sample,tmp_path/'a.sqlite')
    result,scored,model,cal=run_experiment(m,include_xgboost=False)
    assert result['selected_model']=='logistic_regression'
    assert result['splits']['test']==len(scored)
    assert scored.pd_calibrated.between(0,1).all()
    assert all(c in scored for c in ['pd_raw','risk_band','default_12m'])

from credit_risk.statistics import population_stability_index, describe_data

def test_psi_identical():
    vals=np.linspace(0,1,1000)
    assert abs(population_stability_index(vals,vals))<1e-10

def test_psi_shift():
    vals=np.linspace(0,1,1000)
    assert population_stability_index(vals,vals+.4)>0.1

def test_descriptive_stats(sample):
    s=describe_data(sample)
    assert s['n_applications']==900
    assert 0<s['default_rate']<1
    assert 'credit_utilization' in s['psi_2023_to_2025']

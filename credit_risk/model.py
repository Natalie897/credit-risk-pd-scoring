"""Classical PD models and time-based validation. Not a production risk system."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression

NUMERIC = ['age','annual_income','employment_years','debt_to_income','credit_utilization',
           'delinquencies_12m','loan_amount','loan_term_months','loan_to_income','has_delinquency']
CATEGORICAL = ['region','channel','customer_segment']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'default_12m'


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Historical samples 2023 train, 2024 validation, 2025 held-out test.
    dates = pd.to_datetime(frame['application_date'])
    train = frame[dates < '2024-01-01'].copy()
    valid = frame[(dates >= '2024-01-01') & (dates < '2025-01-01')].copy()
    test = frame[dates >= '2025-01-01'].copy()
    if any(len(x)==0 or x[TARGET].nunique()!=2 for x in (train,valid,test)):
        raise ValueError('Each split must be nonempty and contain both target classes')
    return train, valid, test


def preprocessing() -> ColumnTransformer:
    number = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=False)),
                       ('scale', StandardScaler())])
    category = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric',number,NUMERIC),('category',category,CATEGORICAL)])


def candidates(seed: int=42, include_xgboost: bool=True) -> dict[str, Pipeline]:
    models = {'logistic_regression': Pipeline([('prep',preprocessing()),('model',LogisticRegression(max_iter=650,random_state=seed))])}
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError('Install optional xgboost dependency or pass --no-xgboost')
        models['xgboost'] = Pipeline([('prep',preprocessing()),('model',XGBClassifier(
            n_estimators=110, max_depth=3, learning_rate=.055, subsample=.87, colsample_bytree=.9,
            objective='binary:logistic', eval_metric='logloss', random_state=seed,n_jobs=2))])
    return models


def metrics(y, scores, threshold:float=.5) -> dict[str,float]:
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(scores,dtype=float),1e-8,1-1e-8)
    if not (len(y)==len(p) and len(y)>0): raise ValueError('unequal or empty arrays')
    if len(np.unique(y))<2: raise ValueError('Metrics require both target classes')
    return {k:float(v) for k,v in {
       'roc_auc':roc_auc_score(y,p), 'gini':2*roc_auc_score(y,p)-1,
       'pr_auc':average_precision_score(y,p), 'brier':brier_score_loss(y,p),
       'log_loss':log_loss(y,p), 'precision_at_0_5':precision_score(y,p>=threshold,zero_division=0),
       'recall_at_0_5':recall_score(y,p>=threshold,zero_division=0),
       'observed_default_rate':float(y.mean()), 'mean_predicted_pd':float(p.mean()),
    }.items()}


def run_experiment(frame:pd.DataFrame, include_xgboost:bool=True, seed:int=42):
    train,valid,test=split_by_time(frame)
    x_train,y_train=train[FEATURES],train[TARGET]
    x_valid,y_valid=valid[FEATURES],valid[TARGET]
    x_test,y_test=test[FEATURES],test[TARGET]
    scoreboard={}; fitted={}
    for name,model in candidates(seed,include_xgboost).items():
        model.fit(x_train,y_train)
        scores=model.predict_proba(x_valid)[:,1]
        scoreboard[name]=metrics(y_valid,scores)
        fitted[name]=model
    best=max(scoreboard, key=lambda name:scoreboard[name]['roc_auc'])
    chosen=fitted[best]
    valid_predictions=chosen.predict_proba(x_valid)[:,1]
    test_predictions=chosen.predict_proba(x_test)[:,1]
    # Hold-out calibration is for demonstration; in production use independent calibration data.
    calibrator=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
    calibrator.fit(valid_predictions,y_valid)
    calibrated_test=calibrator.predict(test_predictions)
    test_scores=metrics(y_test,calibrated_test)
    test_uncalibrated=metrics(y_test,test_predictions)
    scored=test[['application_id','application_date','customer_segment',TARGET]].copy()
    scored['pd_raw']=test_predictions
    scored['pd_calibrated']=calibrated_test
    scored['risk_band']=pd.cut(calibrated_test,bins=[-1,.08,.20,1.00001],labels=['low','medium','high']).astype(str)
    result={'dataset':'synthetic_educational_only', 'splits':{'train':len(train),'validation':len(valid),'test':len(test)},
            'validation_models':scoreboard, 'selected_by':'validation roc_auc','selected_model':best,
            'test_raw':test_uncalibrated,'test_isotonic_calibrated':test_scores,
            'notes':'Model selection and calibration use 2024 validation; 2025 holdout is evaluated once. No customer decisions.'}
    return result, scored, chosen, calibrator

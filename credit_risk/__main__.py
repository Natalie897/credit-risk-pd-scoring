import argparse
import json
from pathlib import Path
import pandas as pd
from .generate import save_applications
from .mart import build_mart
from .model import run_experiment
from .report import save_report
from .statistics import describe_data


def main():
    parser=argparse.ArgumentParser(description='Educational PD risk scoring pipeline')
    parser.add_argument('action', choices=['demo','model'], help='demo creates synthetic data; model reuses your local CSV')
    parser.add_argument('--n',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--data',default='data/raw/applications.csv')
    parser.add_argument('--no-xgboost',action='store_true')
    args=parser.parse_args()
    if args.action=='demo':
        frame=save_applications(args.data,args.n,args.seed)
    else:
        frame=pd.read_csv(args.data)
    mart=build_mart(frame,'data/credit_features.sqlite')
    result, scored,_,_=run_experiment(mart,include_xgboost=not args.no_xgboost,seed=args.seed)
    save_report(result,scored,Path('reports'))
    (Path('reports')/'eda_statistics.json').write_text(json.dumps(describe_data(mart),ensure_ascii=False,indent=2),encoding='utf-8')
    m=result['test_isotonic_calibrated']
    print(f"Model: {result['selected_model']} | ROC-AUC: {m['roc_auc']:.3f} | Gini: {m['gini']:.3f} | Brier: {m['brier']:.3f}")
    print('Outputs: data/credit_features.sqlite and reports/{metrics.json,test_pd_scores.csv,risk_segments.csv,roc_curve.png,calibration.png}')


if __name__=='__main__': main()

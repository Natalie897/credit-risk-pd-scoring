"""Save reproducible experimental artifacts and graphics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


def save_report(result:dict, scored:pd.DataFrame, outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    scored.to_csv(outdir/'test_pd_scores.csv',index=False)
    y=scored.default_12m.to_numpy()
    p=scored.pd_calibrated.to_numpy()
    fpr,tpr,_=roc_curve(y,p)
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    ax.plot(fpr,tpr,label=f"PD model · AUC {result['test_isotonic_calibrated']['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],linestyle='--',label='Random baseline')
    ax.set(title='ROC curve · held-out 2025 cohort',xlabel='False positive rate',ylabel='True positive rate')
    ax.legend();fig.tight_layout();fig.savefig(outdir/'roc_curve.png',dpi=145);plt.close(fig)
    observed,predicted=calibration_curve(y,p,n_bins=8,strategy='quantile')
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    ax.plot(predicted,observed,'o-',label='Model')
    ax.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
    ax.set(xlabel='Mean predicted PD',ylabel='Observed default frequency',title='Calibration · held-out 2025 cohort',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(outdir/'calibration.png',dpi=145);plt.close(fig)
    by_band=scored.groupby('risk_band',observed=True).agg(n=('default_12m','size'),observed_rate=('default_12m','mean'),average_pd=('pd_calibrated','mean')).reset_index()
    by_band.to_csv(outdir/'risk_segments.csv',index=False)

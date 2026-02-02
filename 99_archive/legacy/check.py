python -c "import pandas as pd; df=pd.read_parquet(r'datasets_blind_v1\step2_features_v1p4\gate_samples_v1p4.parquet'); 
cols=[c for c in df.columns if ('rvol_log_96' in c or 'rvol_z_192' in c or 'nr_48' in c or 'brs_48' in c)]; 
print('n_cols',len(cols)); 
print(df[cols].std(numeric_only=True).sort_values().head(20).to_string());"

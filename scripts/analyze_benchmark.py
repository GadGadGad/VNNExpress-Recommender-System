
import pandas as pd
import io

data = """Type        Model Protocol  Recall@10  NDCG@10      AUC
  CB       bge-m3     cold   0.061958 0.029704 0.659792
  CB     vn-sbert     cold   0.049566 0.024571 0.621024
  CB        tfidf     cold   0.049566 0.026243 0.581515
  CB       bge-m3     full   0.054546 0.028066 0.689575
  CB        tfidf     full   0.048748 0.027035 0.628283
  CB     vn-sbert     full   0.047256 0.024389 0.647051
  CB        tfidf   loo100   0.388016 0.205936 0.000000
  CB       bge-m3   loo100   0.330425 0.188102 0.000000
  CB     vn-sbert   loo100   0.291449 0.162670 0.000000
  CF    simgcl_G2     cold   0.320632 0.231625 0.000000
  CF    simgcl_G3     cold   0.314230 0.229445 0.000000
  CF   xsimgcl_G3     cold   0.295023 0.185639 0.000000
  CF   xsimgcl_G2     cold   0.282838 0.181610 0.000000
  CF  lightgcl_G2     cold   0.061958 0.032432 0.000000
  CF   xsimgcl_G1     cold   0.058860 0.032476 0.000000
  CF  lightgcl_G1     cold   0.056588 0.029438 0.000000
  CF  lightgcl_G3     cold   0.055762 0.029731 0.000000
  CF    hetgnn_G1     cold   0.041099 0.019029 0.000000
  CF    ma-hcl_G2     cold   0.038414 0.021147 0.000000
  CF    hetgnn_G2     cold   0.030979 0.016893 0.000000
  CF    hetgnn_G3     cold   0.030772 0.015291 0.000000
  CF    ma-hcl_G1     cold   0.030772 0.014613 0.000000
  CF    simgcl_G1     cold   0.030359 0.015828 0.000000
  CF    ma-hcl_G3     cold   0.025816 0.015331 0.000000
  CF    ma_hgn_G3     cold   0.020756 0.010296 0.000000
  CF    ma_hgn_G2     cold   0.019000 0.008363 0.000000
  CF sim-mahgn_G2     cold   0.017348 0.009119 0.000000
  CF    ma_hgn_G1     cold   0.015903 0.008039 0.000000
  CF sim-mahgn_G1     cold   0.014663 0.007588 0.000000
  CF sim-mahgn_G3     cold   0.005576 0.002091 0.000000
  CF    simgcl_G3     full   0.236744 0.176540 0.000000
  CF    simgcl_G2     full   0.234960 0.178415 0.000000
  CF   xsimgcl_G3     full   0.228539 0.150610 0.000000
  CF   xsimgcl_G2     full   0.225417 0.153181 0.000000
  CF  lightgcl_G2     full   0.064333 0.035657 0.000000
  CF   xsimgcl_G1     full   0.062467 0.035661 0.000000
  CF  lightgcl_G3     full   0.060425 0.034036 0.000000
  CF  lightgcl_G1     full   0.060198 0.036290 0.000000
  CF    ma-hcl_G2     full   0.044994 0.025065 0.000000
  CF    ma-hcl_G3     full   0.044659 0.028273 0.000000
  CF    ma-hcl_G1     full   0.043998 0.025346 0.000000
  CF    hetgnn_G2     full   0.038405 0.018888 0.000000
  CF    hetgnn_G1     full   0.036061 0.018521 0.000000
  CF    hetgnn_G3     full   0.035246 0.019440 0.000000
  CF    ma_hgn_G3     full   0.028228 0.013967 0.000000
  CF    ma_hgn_G1     full   0.027269 0.014814 0.000000
  CF    simgcl_G1     full   0.023193 0.012742 0.000000
  CF    ma_hgn_G2     full   0.022836 0.011386 0.000000
  CF sim-mahgn_G2     full   0.010067 0.005417 0.000000
  CF sim-mahgn_G1     full   0.009589 0.004792 0.000000
  CF sim-mahgn_G3     full   0.007040 0.003703 0.000000
  CF   xsimgcl_G2   loo100   0.601206 0.459361 0.000000
  CF   xsimgcl_G3   loo100   0.591055 0.457118 0.000000
  CF    simgcl_G3   loo100   0.502644 0.395773 0.000000
  CF    simgcl_G2   loo100   0.496464 0.396062 0.000000
  CF    hetgnn_G2   loo100   0.350975 0.208274 0.000000
  CF    hetgnn_G1   loo100   0.341402 0.201708 0.000000
  CF  lightgcl_G1   loo100   0.340239 0.221901 0.000000
  CF  lightgcl_G2   loo100   0.339167 0.224757 0.000000
  CF    hetgnn_G3   loo100   0.335426 0.196841 0.000000
  CF  lightgcl_G3   loo100   0.334053 0.206183 0.000000
  CF   xsimgcl_G1   loo100   0.315457 0.201893 0.000000
  CF    ma_hgn_G3   loo100   0.304741 0.170625 0.000000
  CF    ma_hgn_G1   loo100   0.303175 0.172781 0.000000
  CF    ma_hgn_G2   loo100   0.284447 0.164893 0.000000
  CF    ma-hcl_G2   loo100   0.250089 0.158419 0.000000
  CF    ma-hcl_G1   loo100   0.239973 0.158053 0.000000
  CF    ma-hcl_G3   loo100   0.238483 0.154780 0.000000
  CF    simgcl_G1   loo100   0.203699 0.121342 0.000000
  CF sim-mahgn_G3   loo100   0.129641 0.071282 0.000000
  CF sim-mahgn_G1   loo100   0.120336 0.063306 0.000000
  CF sim-mahgn_G2   loo100   0.116602 0.058511 0.000000"""

# Parse data
rows = [line.split() for line in data.strip().split('\n')]
headers = rows[0]
df = pd.DataFrame(rows[1:], columns=headers)

# Dynamically convert all metric columns to numeric
metric_cols = [c for c in df.columns if '@' in c or c == 'AUC' or c == 'MRR' or c.lower().startswith('hitrate')]
for col in metric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Best overall model (based on first metric column, usually Recall@10 or Recall@50)
main_metric = metric_cols[0] if metric_cols else 'Recall@10'
best_model = df.loc[df[main_metric].idxmax()]

# 2. Compare G1, G2, G3 for the top model
models_to_compare = ['ma-hcl', 'simgcl', 'xsimgcl', 'hetgnn']
comparison_results = {}
for m_name in models_to_compare:
    m_df = df[df['Model'].str.contains(m_name, case=False)]
    if not m_df.empty:
        try:
            comparison_results[m_name] = m_df.pivot(index='Protocol', columns='Model', values=main_metric)
        except:
            pass

# 3. Compare protocols (Full vs Cold vs LOO)
protocols = df.groupby(['Protocol', 'Type'])[metric_cols].mean().reset_index()

# 4. Compare CF vs CB
cf_vs_cb = df.groupby('Type')[metric_cols].mean()

print("TOP MODEL:")
print(best_model)

print("\nModel Comparisons (G1 vs G2 vs G3):")
for m_name, pivot_df in comparison_results.items():
    print(f"\n--- {m_name.upper()} ---")
    print(pivot_df)

print("\nAverage Metrics per Protocol:")
print(protocols)

print("\nCF vs CB Averages:")
print(cf_vs_cb)

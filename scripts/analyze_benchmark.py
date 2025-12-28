
import pandas as pd
import io

data = """Type        Model Protocol  Recall@10  NDCG@10      AUC
  CB     vn-sbert     cold   0.178419 0.095190 0.656474
  CB        tfidf     cold   0.159188 0.082045 0.624098
  CB       bge-m3     cold   0.158120 0.083515 0.697097
  CB     vn-sbert     full   0.195023 0.101132 0.692853
  CB       bge-m3     full   0.183333 0.101652 0.727015
  CB        tfidf     full   0.168100 0.089590 0.655712
  CB        tfidf   loo100   0.425339 0.210580 0.000000
  CB       bge-m3   loo100   0.352941 0.210144 0.000000
  CB     vn-sbert   loo100   0.303167 0.170360 0.000000
  CF    ma-hcl_G1     cold   0.869919 0.691448 0.000000
  CF    ma-hcl_G3     cold   0.852033 0.644018 0.000000
  CF    ma-hcl_G2     cold   0.843089 0.659985 0.000000
  CF   xsimgcl_G2     cold   0.436179 0.315610 0.000000
  CF   xsimgcl_G3     cold   0.406911 0.283255 0.000000
  CF    simgcl_G2     cold   0.239837 0.144010 0.000000
  CF    simgcl_G3     cold   0.235366 0.136330 0.000000
  CF  lightgcl_G3     cold   0.199187 0.097512 0.000000
  CF  lightgcl_G1     cold   0.193496 0.094426 0.000000
  CF  lightgcl_G2     cold   0.189431 0.101174 0.000000
  CF   xsimgcl_G1     cold   0.099187 0.058617 0.000000
  CF sim-mahgn_G3     cold   0.073171 0.041530 0.000000
  CF sim-mahgn_G1     cold   0.060163 0.026286 0.000000
  CF    simgcl_G1     cold   0.056098 0.030534 0.000000
  CF sim-mahgn_G2     cold   0.045122 0.022840 0.000000
  CF    ma-hcl_G1     full   0.890748 0.797491 0.000000
  CF    ma-hcl_G2     full   0.875712 0.786840 0.000000
  CF    ma-hcl_G3     full   0.872528 0.779056 0.000000
  CF   xsimgcl_G2     full   0.585572 0.478817 0.000000
  CF   xsimgcl_G3     full   0.581017 0.476836 0.000000
  CF    simgcl_G2     full   0.457479 0.330302 0.000000
  CF    simgcl_G3     full   0.446661 0.333808 0.000000
  CF  lightgcl_G2     full   0.124353 0.061796 0.000000
  CF  lightgcl_G3     full   0.123771 0.058176 0.000000
  CF   xsimgcl_G1     full   0.120859 0.064873 0.000000
  CF sim-mahgn_G2     full   0.109472 0.057779 0.000000
  CF  lightgcl_G1     full   0.109446 0.051301 0.000000
  CF sim-mahgn_G1     full   0.099508 0.048864 0.000000
  CF sim-mahgn_G3     full   0.085391 0.045997 0.000000
  CF    simgcl_G1     full   0.060093 0.030706 0.000000
  CF    ma-hcl_G2   loo100   0.938263 0.844172 0.000000
  CF    ma-hcl_G3   loo100   0.935520 0.856783 0.000000
  CF    ma-hcl_G1   loo100   0.934239 0.862142 0.000000
  CF   xsimgcl_G2   loo100   0.695562 0.567809 0.000000
  CF   xsimgcl_G3   loo100   0.691058 0.570125 0.000000
  CF    simgcl_G2   loo100   0.574068 0.442503 0.000000
  CF    simgcl_G3   loo100   0.574055 0.436897 0.000000
  CF  lightgcl_G2   loo100   0.269966 0.157657 0.000000
  CF  lightgcl_G1   loo100   0.255292 0.141888 0.000000
  CF   xsimgcl_G1   loo100   0.247748 0.140946 0.000000
  CF sim-mahgn_G1   loo100   0.245626 0.133812 0.000000
  CF sim-mahgn_G3   loo100   0.243517 0.124391 0.000000
  CF  lightgcl_G3   loo100   0.238975 0.134364 0.000000
  CF sim-mahgn_G2   loo100   0.224094 0.124506 0.000000
  CF    simgcl_G1   loo100   0.171946 0.085404 0.000000"""

# Parse data
rows = [line.split() for line in data.strip().split('\n')]
headers = rows[0]
df = pd.DataFrame(rows[1:], columns=headers)
df['Recall@10'] = pd.to_numeric(df['Recall@10'])
df['NDCG@10'] = pd.to_numeric(df['NDCG@10'])

# 1. Best overall model
best_model = df.loc[df['Recall@10'].idxmax()]

# 2. Compare G1, G2, G3 for the top model (MA-HCL seems dominant)
mahcl = df[df['Model'].str.contains('ma-hcl')]
mahcl_pivoted = mahcl.pivot(index='Protocol', columns='Model', values='Recall@10')

# 3. Compare protocols (Full vs Cold vs LOO)
protocols = df.groupby(['Protocol', 'Type'])['Recall@10'].mean().reset_index()

# 4. Compare CF vs CB
cf_vs_cb = df.groupby('Type')['Recall@10'].mean()

print("TOP MODEL:")
print(best_model)
print("\nMA-HCL Comparison (G1 vs G2 vs G3):")
print(mahcl_pivoted)
print("\nAverage Recall per Protocol:")
print(protocols)

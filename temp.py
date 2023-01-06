from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

import pandas as pd
df_data = pd.DataFrame(breast_cancer_data.data)
df_labels = pd.DataFrame(breast_cancer_data.target)

print(df_data)
print(df_labels.head())
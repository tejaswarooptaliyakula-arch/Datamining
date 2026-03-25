from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

data = pd.DataFrame({
    'Milk': [1,1,0,1],
    'Bread': [1,0,1,1],
    'Butter': [0,1,1,1]
})

frequent = fpgrowth(data, min_support=0.5, use_colnames=True)

print(frequent)

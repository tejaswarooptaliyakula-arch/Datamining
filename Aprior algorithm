from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
#rule generation using  apriori algorithm
# Example dataset
data = pd.DataFrame({
    'Milk': [1,1,0,1],
    'Bread': [1,0,1,1],
    'Butter': [0,1,1,1]
})

frequent = apriori(data, min_support=0.5, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

print(rules)

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


# dataframe and row
# object instance
# iterate in range

file_path = "Market_Basket_Optimisation.csv"
data = pd.read_csv(file_path)

dataset = []

for index, row in data.head(50).iterrows():
    dataset.append(row.dropna().tolist())  

# Convert dataset to one-hot encoded format
transaction_encoder = TransactionEncoder()
transformed_array = transaction_encoder.fit(dataset).transform(dataset)
encode_matrix = pd.DataFrame(transformed_array, columns=transaction_encoder.columns_)

# Apply Apriori algorithm to find frequent itemsets

min_support = 0.01  # Adjust as need
frequent_itemsets = apriori(encode_matrix, min_support=min_support, use_colnames=True)

# Generate association rules
# suport is this

min_confidence = 0.08  # Adjust as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Print frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules["confidence"],rules["lift"])

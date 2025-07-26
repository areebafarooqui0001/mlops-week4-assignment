import pandas as pd 
from datetime import datetime 

print("Loading original iris.csv")
df = pd.read_csv('data/iris.csv')

df['iris_id'] = range(1, len(df) + 1)
df['event_timestamp'] = datetime.now()

df = df[['iris_id', 'event_timestamp', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']]

output_path = "data/iris_feast.csv"
print(f"Saving prepared data to {output_path}")
df.to_csv(output_path, index=False)

print("Preparation complete.")
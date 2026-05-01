# check_metadata.py
import pandas as pd

df = pd.read_csv('data_ready/metadata.csv')

print(f"Total filas en metadata.csv: {len(df)}")
print(f"\nDistribución por stage:")
print(df['stage'].value_counts())

print(f"\nPrimeras 5 filas:")
print(df.head())

print(f"\n¿Hay archivos sin stage?")
print(df['stage'].isna().sum())
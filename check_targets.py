# check_targets.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_ready/metadata.csv')

print("Estadísticas de targets:")
print(f"Attack - min: {df['attack'].min():.2f}, max: {df['attack'].max():.2f}, mean: {df['attack'].mean():.2f}")
print(f"Release - min: {df['release'].min():.2f}, max: {df['release'].max():.2f}, mean: {df['release'].mean():.2f}")

# Ver training vs validation
for stage in ['train', 'val']:
    subset = df[df['stage'] == stage]
    print(f"\n{stage.upper()}:")
    print(f"  Attack mean: {subset['attack'].mean():.2f}")
    print(f"  Release mean: {subset['release'].mean():.2f}")

# Graficar distribución
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(df['attack'], bins=30)
ax1.set_title('Distribución de Attack')
ax2.hist(df['release'], bins=30)
ax2.set_title('Distribución de Release')
plt.show()
import pandas as pd

#download the file from
file_path = 'Oulunsalo_25.11.2024_27.11.2024.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

# Remove extra quotes from all columns
data.columns = data.columns.str.replace('"', '')
data = data.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# Save the cleaned file back to CSV format
cleaned_file_path = 'Oulunsalo_ennuste_25.11.2024_cleaned.csv'
data.to_csv(cleaned_file_path, index=False, encoding='ISO-8859-1')

print(f"Cleaned file saved: {cleaned_file_path}")
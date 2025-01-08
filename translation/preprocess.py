import pandas as pd

# File path
file_path = "data/europarl-v7.de-en.de"

# Initialize an empty list to store the data
data = []
limit = 200000
limit_len = 30
# Open the file and read each line
with open(file_path, 'r', encoding='utf-8') as file:
    i  = 0
    for line in file:
        # Strip any leading/trailing whitespace and append to the list
        data.append([line.strip()])
        i += 1
        if i == limit:
            break

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["Text"])
#df.to_csv("data/news_de.csv", index=False)
add1 = df.loc[:180000]
add1_val = df.loc[180001:189000]

# File path
file_path = "data/europarl-v7.de-en.en"

# Initialize an empty list to store the data
data = []

# Open the file and read each line
with open(file_path, 'r', encoding='utf-8') as file:
    i = 0
    for line in file:
        # Strip any leading/trailing whitespace and append to the list
        data.append([line.strip()])
        i += 1
        if i == limit:
            break

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["Text"])
#df.to_csv("data/news_en.csv", index=False)
add2 = df.loc[:180000]
add2_val = df.loc[180001:189000]

print('done')
# Load the existing CSV file
df = pd.read_csv('data/train2.csv')

for i in range(180001):
    new_data = {'English': add2.values[i][0], 'Ger': add1.values[i][0]}
    new_df = pd.DataFrame([new_data])

# Append the new data to the existing DataFrame
    if len(add2.values[i][0]) >= 1 and len(add1.values[i][0]) >= 1 and len(add2.values[i][0].split()) <= limit_len and len(add1.values[i][0].split()) <= limit_len:
        df = pd.concat([df, new_df], ignore_index=True)

print(len(df))
# Save the updated DataFrame back to CSV
df.to_csv('data/train_30.csv', index=False)

df = pd.read_csv('data/val2.csv')

for i in range(9000):
    new_data = {'English': add2_val.values[i][0], 'Ger': add1_val.values[i][0]}
    new_df = pd.DataFrame([new_data])

# Append the new data to the existing DataFrame
    if len(add2_val.values[i][0]) >=1 and len(add1_val.values[i][0]) >=1 and len(add2_val.values[i][0].split()) <=limit_len and len(add1_val.values[i][0].split()) <=limit_len:
        df = pd.concat([df, new_df], ignore_index=True)

# Save the updated DataFrame back to CSV
df.to_csv('data/val_30.csv', index=False)

max_len = 0

df = pd.read_csv('data/train_30.csv')
data = df['Ger']

for i in range(60000):
    l = len(data.values[i].split())
    if l > max_len:
        max_len = l
        print(i)

data = df['English']

for i in range(60000):
    l = len(data.values[i].split())
    if l > max_len:
        max_len = l
        print(i)

df = pd.read_csv('data/val_30.csv')
data = df['Ger']

for i in range(2000):
    l = len(data.values[i].split())
    if l > max_len:
        max_len = l

data = df['English']

for i in range(2000):
    l = len(data.values[i].split())
    if l > max_len:
        max_len = l

print(max_len)
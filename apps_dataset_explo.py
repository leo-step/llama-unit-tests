from datasets import load_dataset
import pandas as pd

ds = load_dataset("codeparrot/apps", split="train")

samples = []

for sample in ds:
    samples.append(sample)
    
df = pd.DataFrame(samples)

def string_length_stats(df, percentiles=[0.95, 0.99]):
    stats = {}
    for column in df.select_dtypes(include=['object']).columns:
        lengths = df[column].str.len()
        stats[column] = lengths.describe(percentiles=percentiles)
    
    return pd.DataFrame(stats)

length_stats = string_length_stats(df)
print(length_stats)

# keep only 99%, top 1% by length skip over when creating dataset
'''

           question     solutions  input_output   difficulty         url  starter_code
count   5000.000000  5.000000e+03  5.000000e+03  5000.000000  5000.00000   5000.000000
mean    1254.159800  1.349326e+04  5.749351e+03    10.727800    52.15740     31.997400
std      888.617558  4.653155e+04  3.347700e+05     1.433219     7.34312     56.080474
min       50.000000  1.200000e+01  0.000000e+00     9.000000    36.00000      0.000000
50%     1051.500000  4.051000e+03  1.320000e+02    12.000000    54.00000     23.000000
95%     2815.050000  4.747915e+04  1.098050e+03    12.000000    63.00000     92.000000
99%     4107.430000  1.666181e+05  3.529120e+03    12.000000    82.00000    253.020000
max    13955.000000  1.214112e+06  2.361317e+07    12.000000   108.00000   1404.000000

'''
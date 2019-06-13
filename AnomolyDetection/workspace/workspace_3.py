import pandas as pd


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/train/'

df_base = pd.read_csv('{}{}'.format(DIRECTORY, 'base.csv'))
df_base['label'] = 0

df_malicious = pd.read_csv('{}{}'.format(DIRECTORY, 'base_malicious.csv'))
df_malicious['label'] = 1

df = pd.concat([df_base, df_malicious])

df.to_csv('{}{}'.format(DIRECTORY, 'train.csv', index=False))

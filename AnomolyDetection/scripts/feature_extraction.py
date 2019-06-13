from AnomolyDetection.anomoly_lib import *
import pandas as pd


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/original/TestWT/'

directory_list = [DIRECTORY]

features = extract_directory_list_fft_features(directory_list=directory_list)

print(features)

df = pd.DataFrame(features)

df.to_csv('{}{}'.format(DIRECTORY, 'test.csv'), index=False)

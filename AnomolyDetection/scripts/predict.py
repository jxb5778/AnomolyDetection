from AnomolyDetection.anomoly_API_lib import run_pipeline_prediction
from AnomolyDetection.StrOUD import StrOUD

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import pandas as pd


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/'

p_vals = run_pipeline_prediction(
    normal_dir_list=[
        '{}{}'.format(DIRECTORY, 'original/base/ModeA/'),
        '{}{}'.format(DIRECTORY, 'original/base/ModeB/'),
        '{}{}'.format(DIRECTORY, 'original/base/ModeC/'),
        '{}{}'.format(DIRECTORY, 'original/base/ModeD/')
    ],
    test_dir=['{}{}'.format(DIRECTORY, 'original/TestWT/')],
    pipeline=Pipeline([
        ('pca', TruncatedSVD(n_components=2000))
    ]),
    clf=StrOUD(k_neighbors=5)
)

df = pd.DataFrame({'p_val': p_vals})

df.to_csv('{}{}'.format(DIRECTORY, 'prediction/prediction_2.csv'), header=None, index=False)

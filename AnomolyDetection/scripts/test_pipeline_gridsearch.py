from AnomolyDetection.anomoly_API_lib import run_pipeline_test_gridsearchcv
from AnomolyDetection.StrOUD import StrOUD

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/original/base/'

run_pipeline_test_gridsearchcv(
    normal_dir_list=[
        '{}{}'.format(DIRECTORY, 'ModeA/'),
        '{}{}'.format(DIRECTORY, 'ModeB/'),
        '{}{}'.format(DIRECTORY, 'ModeC/'),
        '{}{}'.format(DIRECTORY, 'ModeD/')
    ],
    malicious_dir_list=['{}{}'.format(DIRECTORY, 'ModeM/')],
    pipeline=Pipeline([
        ('pca', TruncatedSVD()),
        ('clf', StrOUD())
    ]),
    gridsearch_params={
        'param_grid': {
            'clf__k_neighbors': [3, 5, 7, 9],
            'clf__confidence': [0.8, 0.85, 0.9],
            'pca__n_components': [25, 50, 100, 200, 300]
        },
        'cv': 5,
        'scoring': 'f1'
    }
)

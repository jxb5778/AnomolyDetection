from AnomolyDetection.anomoly_API_lib import run_pipeline_test
from AnomolyDetection.StrOUD import StrOUD

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/original/base/'

run_pipeline_test(
    normal_dir_list=[
        '{}{}'.format(DIRECTORY, 'ModeA/'),
        '{}{}'.format(DIRECTORY, 'ModeB/'),
        '{}{}'.format(DIRECTORY, 'ModeC/'),
        '{}{}'.format(DIRECTORY, 'ModeD/')
    ],
    malicious_dir_list=['{}{}'.format(DIRECTORY, 'ModeM/')],
    pipeline=Pipeline([
        ('pca', TruncatedSVD(n_components=75))
    ]),
    clf=StrOUD(k_neighbors=5, confidence=0.75)
)

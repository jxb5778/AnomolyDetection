import numpy as np
import os
import pandas as pd

from numpy.fft import fft
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report


def extract_file_fft_features(filename, delimiter=' \t'):

    with open(filename) as f:
        for line in f:
            line_split = np.array(line.split(delimiter))[:20000]

            features = [float(feature) for feature in line_split]
            fft_features = fft(features)

    amplitude = 2.0 / 20000 * np.abs(fft_features[0:10000])

    return amplitude


def extract_directory_fft_features(directory, file_delimiter=' \t'):

    directory_features = []

    for file in os.listdir(directory):
        file_features = extract_file_fft_features(
            filename='{}{}'.format(directory, file),
            delimiter=file_delimiter
        )
        directory_features.append(file_features)

    return directory_features


def extract_directory_fft_features_predict(directory, file_delimiter=' \t'):

    directory_features = []

    for index in range(1, 500):
        file_features = extract_file_fft_features(
            filename='{}Data{}.txt'.format(directory, index),
            delimiter=file_delimiter
        )
        directory_features.append(file_features)

    return directory_features


def extract_directory_list_fft_features(directory_list, file_delimiter=' \t', extraction_type='train'):

    directory_set_features = []

    for directory in directory_list:
        print("Working on Directory: ", directory)
        if extraction_type == 'train':
            directory_features = extract_directory_fft_features(
                directory=directory,
                file_delimiter=file_delimiter
            )
        elif extraction_type == 'predict':
            directory_features = extract_directory_fft_features_predict(
                directory=directory,
                file_delimiter=file_delimiter
            )
        else:
            raise ValueError('You must use one of train or predict for extraction_type')

        directory_set_features.extend(directory_features)

    return directory_set_features


def extract_directory_inputs_predict(normal_dir_list, test_dir, context):

    context.value = {
        'normal_features': extract_directory_list_fft_features(directory_list=normal_dir_list),
        'test_features': extract_directory_list_fft_features(directory_list=test_dir, extraction_type='predict')
    }

    return


def extract_directory_inputs_test(normal_dir_list, malicious_dir_list, context):

    context.value = {
        'normal_features': extract_directory_list_fft_features(directory_list=normal_dir_list),
        'malicious_features': extract_directory_list_fft_features(directory_list=malicious_dir_list)
    }

    context.value = {
        'normal_df': pd.DataFrame(context.value['normal_features']),
        'malicious_df': pd.DataFrame(context.value['malicious_features'])
    }

    context.value['normal_df']['label'] = 0
    context.value['malicious_df']['label'] = 1

    context.value = pd.concat([context.value['normal_df'], context.value['malicious_df']])

    return


def label_split_inputs(context):

    context.value = dict(X=context.value)
    context.value['y'] = context.value['X']['label']
    context.value['X'] = context.value['X'].drop(columns=['label'])

    return


def pipeline_test_gridsearchcv(context, pipeline, gridsearch_params):

    search = GridSearchCV(pipeline, **gridsearch_params)
    search.fit(context.value['X'], context.value['y'])

    print(search.best_estimator_)
    print(search.best_params_)
    print(search.best_score_)

    return


def pipeline_train_test_split(context, pipeline, clf):

    X_train, X_test, y_train, y_test = train_test_split(
        context.value['X'], context.value['y'], test_size=0.2, random_state=42
    )

    context.value = None

    pipeline.fit(X_train, y_train)

    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)

    clf.fit(X_train, y_train)
    clf.predict(X_test)

    print(list(y_test))
    print(clf.pred)
    print(clf.p_vals)

    print(classification_report(y_true=y_test, y_pred=clf.pred))

    return


def pipeline_prediction(context, pipeline, clf):

    pipeline.fit(context.value['normal_features'])

    context.value = {
        'train': pipeline.transform(context.value['normal_features']),
        'test': pipeline.transform(context.value['test_features'])
    }

    clf.fit(context.value['train'])
    context.value['train'] = None

    clf.predict(context.value['test'])
    context.value = None

    return clf.p_vals

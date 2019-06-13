from AnomolyDetection.anomoly_lib import *
from buffer import Buffer


def run_pipeline_test_gridsearchcv(normal_dir_list, malicious_dir_list, pipeline, gridsearch_params):

    context = Buffer()

    extract_directory_inputs(
        normal_dir_list=normal_dir_list,
        malicious_dir_list=malicious_dir_list,
        context=context
    )

    label_split_inputs(context=context)

    pipeline_test_gridsearchcv(
        context=context,
        pipeline=pipeline,
        gridsearch_params=gridsearch_params
    )

    return


def run_pipeline_test(normal_dir_list, malicious_dir_list, pipeline, clf):

    context = Buffer()

    extract_directory_inputs_test(
        normal_dir_list=normal_dir_list,
        malicious_dir_list=malicious_dir_list,
        context=context
    )

    label_split_inputs(context=context)

    pipeline_train_test_split(
        context=context,
        pipeline=pipeline,
        clf=clf
    )

    return


def run_pipeline_prediction(normal_dir_list, test_dir, pipeline, clf):
    context = Buffer()

    extract_directory_inputs_predict(
        normal_dir_list=normal_dir_list,
        test_dir=test_dir,
        context=context
    )

    return pipeline_prediction(context=context, pipeline=pipeline, clf=clf)

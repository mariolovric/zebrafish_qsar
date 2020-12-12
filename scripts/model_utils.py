from scripts import *


def handle_model_index(modelset='noname', dl='dump', indexlist=[]):
    """

    :param modelset: name of the modelset files
    :param dl: dump or load model
    :param indexlist:
    :return:
    """
    import json
    if dl == 'dump':
        with open(f"../model_dir/{modelset}.txt", "w") as fp:
            json.dump(indexlist, fp)
        print('modelset save')
    elif dl == 'load':
        with open(f"../model_dir/{modelset}.txt", "r") as fp:
            loaded = json.load(fp)
        return loaded
    else:
        print('Error in index loading')


def load_predict(endpoint: str, predictive_set: str, datasets: dict, feature_list: list,
                 model_folder_path='../model_dir/'):
    """

    :param endpoint:
    :param predictive_set: predictive set of data
    :param feature_list: feature list extracted from model params
    :param model_folder_path: folder with saved models
    :return:
    """
    from joblib import load

    clf = load(f'{model_folder_path}{endpoint}.joblib')
    predicted_probabilities = clf.predict_proba(datasets[predictive_set][feature_list])
    dataframe_pred_proba = pd.DataFrame(predicted_probabilities,
                                        index=datasets[predictive_set].index, columns=['c=0', 'c=1'])

    proba_vector = dataframe_pred_proba['c=1'].round(2).squeeze()
    proba_vector.name = endpoint
    return proba_vector


def best_mod_expander_for_external(best_model, param_dict):
    """

    :param best_model:
    :return:
    """
    assert best_model in param_dict.keys(), 'Model name not in params keys'
    expanded = dict(zip(['scaling_opt', 'classifier', 'seed', 'predictive_set', 'feat_sel', 'target', 'scaling_opt',
                         'num_feat'], best_model.split('|')))

    predictive_set = expanded['predictive_set']
    endpoint = expanded['target']
    feature_list = param_dict[best_model]['feat']

    return predictive_set, endpoint, feature_list


def get_model_params(param_path: str):
    """
    Retrieve model params from a file
    :param param_path:
    :return:
    """
    import os.path
    import json
    assert os.path.isfile(param_path), 'Params path does not exist?'

    with open(param_path, 'r') as f:
        param_dict = json.load(f)
    return param_dict

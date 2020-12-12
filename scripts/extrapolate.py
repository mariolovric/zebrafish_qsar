from scripts import *
from scripts.model_utils import best_mod_expander_for_external, get_model_params, handle_model_index, load_predict
from scripts.structure_handler import calc_desc_fp, currate_smiles_columns

if __name__ == "__main__":

    best_models_indices = handle_model_index(modelset='model_indices', dl='load')[0]
    params = get_model_params('../model_dir/final_params.json')
    smiles_dataframe = pd.read_csv('../data/smiles.csv', index_col=0)
    smiles_dataframe.columns = ['smiles']

    collected_results_df = pd.DataFrame()
    smiles_dataframe = currate_smiles_columns(smiles_dataframe, 'smiles')
    fingerprints, descriptors = calc_desc_fp(smiles_dataframe, 1)
    datasets = {'fp': fingerprints, 'ds': descriptors}

    for best_model in best_models_indices:
        predictive_set, endpoint, feature_list = best_mod_expander_for_external(best_model, params)
        load_pr_ser = load_predict(endpoint, predictive_set, datasets, feature_list,
                                   model_folder_path='../model_dir/')
        collected_results_df = pd.concat([collected_results_df, load_pr_ser], axis=1)

    print(collected_results_df.head(2))
    #collected_results_df.to_csv('../results/model_output.csv')

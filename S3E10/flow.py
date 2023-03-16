from IPython.core.display import display
import pandas as pd
import numpy as np
import yaml, sys
from scipy.stats import ks_2samp
import seaborn as sns
import copy, gc, pickle, os, itertools
#data
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
#model
import optuna
import torch
from utils.helper_func import OptunaWeights, StackingAveragedModels, pickle_dump, pickle_load, DataProcessor, reduce_memory_usage, loss_func
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
if torch.cuda.is_available():
    device = 'gpu'
    tree_method = 'gpu_hist'
    import cudf
    from cuml.sklearn.svm import SVC
    from cuml.sklearn.naive_bayes import GaussianNB
    from cuml.sklearn.tree import DecisionTreeClassifier
    from cuml.sklearn.neural_network import MLPClassifier
    from cuml.sklearn.neighbors import KNeighborsClassifier
    from cuml.sklearn.calibration import CalibratedClassifierCV
    from cuml.sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier, AdaBoostClassifier ,BaggingClassifier, VotingClassifier, StackingClassifier
else:
    device = 'cpu'
    tree_method='hist'
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier, AdaBoostClassifier ,BaggingClassifier, VotingClassifier, StackingClassifier
# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_data(config):
    train_df = pd.read_csv(config['train_path'], index_col=config['index_col'],encoding="UTF-8")
    test_df = pd.read_csv(config['test_path'], index_col=config['index_col'],encoding="UTF-8")
    if config["ori_path"]:
        original_df = pd.read_csv(config['ori_path'], encoding="UTF-8")
    if config["sub_path"]:
        sub = pd.read_csv(config['sub_path'], encoding="UTF-8")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = test_df.select_dtypes(include=numerics).columns.tolist()
    cat_cols = list(set(test_df.columns.tolist()) - set(num_cols))
    use_cols = cat_cols+num_cols
    agg_col = num_cols
    agg_func = ['mean', 'std']
    group_cols = [list(comb) for comb in list(itertools.combinations(num_cols, 2))]
    config['num_cols'] = num_cols
    config['cat_cols'] = cat_cols
    #Top 30 importance
    # config['use_cols'] = ['SuperplasticizerComponentFineAggregateComponent_mean_CementComponent', 'CementComponent', 'WaterComponentAgeInDays_mean_BlastFurnaceSlag', 'CoarseAggregateComponentAgeInDays_mean_AgeInDays', 'WaterComponentAgeInDays_mean_AgeInDays', 'BlastFurnaceSlagAgeInDays_mean_SuperplasticizerComponent', 'FlyAshComponentWaterComponent_mean_CementComponent', 'WaterComponentCoarseAggregateComponent_mean_CementComponent', 'CoarseAggregateComponentAgeInDays_mean_CementComponent', 'CementComponentFlyAshComponent_mean_CementComponent', 'SuperplasticizerComponentAgeInDays_mean_CementComponent', 'FlyAshComponentSuperplasticizerComponent_mean_CementComponent', 'CementComponentAgeInDays_mean_FineAggregateComponent', 'BlastFurnaceSlagAgeInDays_mean_WaterComponent', 'BlastFurnaceSlagFineAggregateComponent_mean_SuperplasticizerComponent', 'CementComponentAgeInDays_mean_CementComponent', 'CementComponentAgeInDays_mean_AgeInDays', 'WaterComponentFineAggregateComponent_mean_BlastFurnaceSlag', 'CementComponentBlastFurnaceSlag_mean_CementComponent', 'WaterComponentFineAggregateComponent_mean_CementComponent', 'BlastFurnaceSlagFineAggregateComponent_mean_WaterComponent', 'SuperplasticizerComponentAgeInDays_mean_FlyAshComponent', 'FineAggregateComponentAgeInDays_mean_AgeInDays', 'FlyAshComponentAgeInDays_mean_SuperplasticizerComponent', 'BlastFurnaceSlagFlyAshComponent_mean_CementComponent', 'CementComponentSuperplasticizerComponent_mean_CementComponent', 'CoarseAggregateComponentAgeInDays_mean_WaterComponent', 'FlyAshComponentAgeInDays_mean_AgeInDays', 'SuperplasticizerComponentCoarseAggregateComponent_mean_WaterComponent', 'CementComponentFlyAshComponent_mean_SuperplasticizerComponent', 'CementComponentWaterComponent_mean_FlyAshComponent', 'CementComponentSuperplasticizerComponent_mean_SuperplasticizerComponent', 'BlastFurnaceSlagSuperplasticizerComponent_mean_WaterComponent', 'BlastFurnaceSlag', 'FlyAshComponentAgeInDays_mean_CementComponent', 'FlyAshComponentSuperplasticizerComponent_mean_BlastFurnaceSlag', 'FlyAshComponentFineAggregateComponent_mean_CementComponent', 'SuperplasticizerComponentAgeInDays_mean_AgeInDays', 'CementComponentWaterComponent_mean_CementComponent', 'BlastFurnaceSlagAgeInDays_mean_CementComponent', 'BlastFurnaceSlagSuperplasticizerComponent_mean_CementComponent', 'FlyAshComponentCoarseAggregateComponent_mean_SuperplasticizerComponent', 'SuperplasticizerComponentCoarseAggregateComponent_mean_SuperplasticizerComponent', 'CementComponentCoarseAggregateComponent_mean_CementComponent', 'BlastFurnaceSlagFineAggregateComponent_mean_CementComponent', 'BlastFurnaceSlagWaterComponent_mean_CementComponent', 'FlyAshComponentAgeInDays_mean_BlastFurnaceSlag', 'AgeInDays', 'BlastFurnaceSlagAgeInDays_mean_AgeInDays', 'WaterComponentAgeInDays_mean_CoarseAggregateComponent', 'SuperplasticizerComponentAgeInDays_mean_WaterComponent', 'WaterComponentCoarseAggregateComponent_mean_SuperplasticizerComponent', 'BlastFurnaceSlagCoarseAggregateComponent_mean_SuperplasticizerComponent', 'SuperplasticizerComponentAgeInDays_mean_CoarseAggregateComponent', 'WaterComponentCoarseAggregateComponent_mean_FineAggregateComponent', 'CementComponentFineAggregateComponent_mean_CementComponent']
    # config['use_cols'] += ['TotalComponentWeight', 'WCR', 'AR', 'WCPR','Cement-Age']
    #Top 10 importance
    # config['use_cols'] = ['BlastFurnaceSlag', 'CementComponentAgeInDays_mean_AgeInDays', 'SuperplasticizerComponentCoarseAggregateComponent_mean_WaterComponent', 'SuperplasticizerComponentAgeInDays_mean_FlyAshComponent', 'CoarseAggregateComponentAgeInDays_mean_AgeInDays', 'BlastFurnaceSlagCoarseAggregateComponent_mean_SuperplasticizerComponent', 'TotalComponentWeight', 'FlyAshComponentAgeInDays_mean_AgeInDays', 'FineAggregateComponentAgeInDays_mean_AgeInDays', 'AgeInDays', 'WaterComponentAgeInDays_mean_AgeInDays', 'CementComponentSuperplasticizerComponent_mean_SuperplasticizerComponent', 'BlastFurnaceSlagSuperplasticizerComponent_mean_WaterComponent', 'CementComponentWaterComponent_mean_FlyAshComponent', 'BlastFurnaceSlagAgeInDays_mean_AgeInDays', 'FlyAshComponentSuperplasticizerComponent_mean_CementComponent', 'Cement-Age', 'CoarseAggregateComponentAgeInDays_mean_WaterComponent', 'BlastFurnaceSlagAgeInDays_mean_SuperplasticizerComponent', 'FlyAshComponentCoarseAggregateComponent_mean_SuperplasticizerComponent', 'WCPR']
    config['agg_col'] = agg_col
    config['agg_func'] = agg_func
    config['group_cols'] = group_cols
    dp = DataProcessor(
                    cat_cols=cat_cols, 
                    num_cols=num_cols, 
                    use_cols=use_cols,
                    agg_col=agg_col,
                    agg_func=agg_func,
                    group_cols=group_cols,
                    comb_cat_cols=[],
                    is_agg = config['is_agg']
                    )
    train_df, test_df, original_df = dp.transform(train_df, test_df, original_df)
    if config["ori_path"]:
        train_df = pd.concat([train_df, original_df], axis=0)
    train_df = train_df.reset_index(drop=True)
    if config['drop_duplicate']:
        train_df = train_df.drop_duplicates()
    if config['use_cols']==[]:#All FE
        config['use_cols'] = test_df.columns.tolist()
    X_train = reduce_memory_usage(train_df[config['use_cols']])
    y_train = train_df[config['target']]
    X_test = reduce_memory_usage(copy.deepcopy(test_df[config['use_cols']]))
    if config['device']=='gpu':
        X_train = cudf.from_pandas(X_train.astype("float32"))
        y_train = cudf.from_pandas(y_train.astype('int32'))
        X_test = cudf.from_pandas(X_test.astype("float32"))
    gc.collect()
    return X_train, y_train, X_test, sub, config

def get_model(config):
    lgb_model = lgb.LGBMClassifier(**config['lgb'], device=device)
    xgb_model = xgb.XGBClassifier(**config['xgb'], tree_method=tree_method)
    cat_model = cat.CatBoostClassifier(**config['cat'], task_type=device.upper())
    Rf = RandomForestClassifier(**config['Rf'])
    Gb = GradientBoostingClassifier(**config['Gb'])
    Dt = DecisionTreeClassifier(**config['Dt'])
    Ada = AdaBoostClassifier(**config['Ada'], base_estimator=Dt)
    L_svc = SVC(**config['L_svc'])
    RBF_svc = SVC(**config['RBF_svc'])
    Gnb = GaussianNB()
    mlp = MLPClassifier(**config['mlp'])
    Kn = KNeighborsClassifier(**config['Kn'])
    Cc = CalibratedClassifierCV(**config['Cc'])

    meta_model_dict = {
        'L_svc':L_svc,
        'RBF_svc':RBF_svc,
        'lgb':lgb_model,
        'xgb':xgb_model,
        # 'Cc':Cc
    }
    stacking_model_dict = {
        'lgb':lgb_model,
        'xgb':xgb_model,
        'cat':cat_model,
        'Rf':Rf,
        'Gb':Gb,
        'Ada':Ada,
        'Dt':Dt,
        'L_svc':L_svc,
        'RBF_svc':RBF_svc,
        'mlp':mlp,
        'Kn':Kn,
        'Cc':Cc
    }
    return stacking_model_dict, meta_model_dict

def main(config):
    ################################

    #### Init

    ################################
    X_train, y_train, X_test, sub ,config= get_data(config)
    # X_train.describe().to_csv('describe.csv', index=False)
    stacking_model_dict, meta_model_dict = get_model(config)
    print(X_train.shape)
    ################################

    #### Stacking Model

    ################################
    gc.collect()
    print("##"*20)
    print("Start Stacking Model")
    print("##"*20)
    stacked_averaged_models = StackingAveragedModels(base_models = stacking_model_dict,
                                                    meta_models = meta_model_dict,
                                                    config=config,
                                                    n_folds=config['n_folds'])
    # X_train = X_train.iloc[:100]
    # y_train = y_train.iloc[:100]
    _, oof_df,score_dict, f_im_df = stacked_averaged_models.fit(X_train, y_train)
    gc.collect()
    print("##"*20)
    print("Score : ")
    print(score_dict)
    print("##"*20)
    current_dir = os.getcwd() 
    oof_dir = os.path.join(current_dir, 'results\\oof', config['save_dir'])
    test_dir = os.path.join(current_dir, 'results\\test', config['save_dir'])
    if not os.path.exists(oof_dir):
        os.makedirs(oof_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if config['save_f_im']:
        f_im_df.to_csv('f_im_df.csv')
    print("Save f_im_df")
    preds_df = stacked_averaged_models.predict(X_test)
    gc.collect()
    pickle_dump(score_dict, path=os.path.join(oof_dir,'score_dict.pickle'))
    print("Save score_dict")
    oof_df.to_csv(os.path.join(oof_dir,'oof_df.csv'), index=False)
    print("Save oof_df")
    preds_df.to_csv(os.path.join(test_dir,'preds_df.csv'), index=False)
    print("Save preds_df")
    # oof_df = pd.read_csv('./results/oof/oof_df.csv', encoding="UTF-8")
    ################################

    #### Optuna Weights

    ################################
    print("##"*20)
    print("Start Optuna Weight")
    print("##"*20)
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=10)
    val_pred = optweights.fit_predict(y_train.values, oof_df.values.T, n_trials=config['optuna_trial'], direction=config['score_direction'])
    # np.save(os.path.join(oof_dir,"weighted_oof.npy"),val_pred)
    oof_df["weighted"] = val_pred
    # print(optweights.weights)
    weighted_score = loss_func(y_train, val_pred)
    pickle_dump({"score":weighted_score, "weights":optweights.weights}, path = os.path.join(oof_dir,'weighted_score.pickle'))
    print("Optuna Weights Score : ", {weighted_score})
    opt_pred = optweights.predict(preds_df.values.T)
    ################################

    #### Save Results

    ################################
    score_dict["weighted"] = weighted_score
    preds_df["weighted"] = opt_pred
    if config['sub_path']:
        sub[config['target']] = opt_pred
        sub.to_csv(os.path.join(test_dir,'weighted_sub.csv'), index=False)
        print("Save weighted_sub")
    score_values = np.array(list(score_dict.values()))
    score_keys = np.array(list(score_dict.keys()))
    if config['score_direction']=='minimize':
        highest_score = score_values.min()
        highest_key = score_keys[score_values.argmin()]
        lowest_score = score_values.max()
        lowest_key = score_keys[score_values.argmax()]
    else:
        highest_score = score_values.max()
        highest_key = score_keys[score_values.argmax()]
        lowest_score = score_values.min()
        lowest_key = score_keys[score_values.argmin()]
    if config['sub_path']:
        sub[config['target']] = preds_df[highest_key].values
        sub.to_csv(os.path.join(test_dir, 'highest_sub.csv'), index=False)
        print("Save Highest Score sub")
        sub[config['target']] = preds_df[lowest_key].values
        sub.to_csv(os.path.join(test_dir, 'lowest_sub.csv'), index=False)
        print("Save Lowest Score sub")
    print("Highest Score :", highest_key, highest_score)
    print("Lowest Score :", lowest_key, lowest_score)
    print("Std : {} Mean : {}".format(score_values.std(),score_values.mean()))
    f = open(os.path.join(oof_dir,"results.txt"), "w")
    for k, v in config.items():
        f.write(f"'{k}':")
        f.write(str(v))
        f.write("\n")
    f.write("Highest Score : {} {}".format(highest_key, highest_score))
    f.write("\n")
    f.write("Lowest Score : {} {}".format(lowest_key, lowest_score))
    f.write("\n")
    f.write("Std : {} Mean : {}".format(score_values.std(),score_values.mean()))
    f.write("\n")
    f.close()
if __name__=='__main__':
    try:
        with open('flow.yaml', encoding="Shift-JIS") as file:
            config = yaml.safe_load(file)
            print("##"*20)
            print("Config : ")
            print(config)
            print("##"*20)
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    config['device'] = device
    main(config)
    if config['submit']:
        os.system(f'kaggle competitions submit -c playground-series-s3e9 -f ./results/test/{config["save_dir"]}/highest_sub.csv -m test')

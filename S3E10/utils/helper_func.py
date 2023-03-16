from IPython.core.display import display
import pandas as pd
import numpy as np
import yaml, sys, pickle
from scipy.stats import ks_2samp
import seaborn as sns
#data
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, log_loss
#model
import optuna
from functools import partial
# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_models, config ,n_folds=5):
        self.base_models = base_models
        self.meta_models = meta_models
        self.n_folds = n_folds
        self.config = config
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = {name:list() for name,_ in self.base_models.items()}
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.config['random_state'])
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        oof_df = pd.DataFrame(columns=self.base_models.keys())
        f_im_df = pd.DataFrame()
        oof_array = np.zeros((X.shape[0], ))
        for name, model in self.base_models.items():
            print(name)
            f_im_list = []
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[name].append(instance)
                if "lgb" in name or "xgb" in name or "cat" in name:
                    instance.fit(X.iloc[train_index], y.iloc[train_index],
                            eval_set=[(X.iloc[holdout_index],y.iloc[holdout_index])],
                            early_stopping_rounds=self.config['early_stopping_rounds'],
                            verbose=self.config['verbose']
                            )
                    y_pred = instance.predict_proba(X.iloc[holdout_index])[:,1]
                    if self.config['save_f_im']:
                        f_im_list.append(instance.feature_importances_)
                    # print(instance.feature_importances_)
                else:
                    instance.fit(X.iloc[train_index], y.iloc[train_index])
                    y_pred = instance.predict_proba(X.iloc[holdout_index])[:,1]
                oof_array[holdout_index] = y_pred
            oof_df[name] = oof_array
            if ("lgb" in name or "xgb" in name or "cat" in name) and self.config['save_f_im']: 
                f_im_df[name] = np.array(f_im_list).mean(0)
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        # self.meta_model_.fit(out_of_fold_predictions, y)
        if self.config['save_f_im']:
            f_im_df.set_axis(X.columns.tolist(), inplace=True)
        _, oof_df,score_dict = self.fit_meta(oof_df, y)
        return self, oof_df, score_dict, f_im_df
    def fit_meta(self, df, y):
        dfs = self.get_fit_columns(df)
        self.meta_models_ = {name:[clone(model) for _ in range(self.choice_model_len)] 
                            for name, model in self.meta_models.items()}
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        score_dict = {}
        oof_df = pd.DataFrame()
        for name, meta_models in self.meta_models_.items():
            for k in range(self.choice_model_len):
                oof_array = np.zeros((dfs[k].shape[0], ))
                for trn_idx, val_idx in kfold.split(dfs[k], y):
                    x_train, y_train = dfs[k].iloc[trn_idx], y.iloc[trn_idx]
                    x_val = dfs[k].iloc[val_idx]
                    meta_models[k].fit(x_train, y_train)
                    oof_array[val_idx] = meta_models[k].predict_proba(x_val)[:,1]
                oof_df[f"meta:{name}, base:"+str(self.fit_cols[k])] = oof_array
                score_dict[f"meta:{name}, base:"+str(self.fit_cols[k])] = loss_func(y, oof_array)
        return self, oof_df,score_dict
    
    def get_fit_columns(self, df):
        corr_df = df.corr()     
        corr_df = corr_df[corr_df<0.95].applymap(lambda x : 0 if np.isnan(float(x)) else 1)
        cols = df.columns
        for c1 in cols:
            for c2 in cols:
                if not c1 ==c2:
                    statistic = ks_2samp(df[c1], df[c2]).statistic
                    if not statistic>=0.05:
                        corr_df[c1][c2] *=0
        # corr<0.95 and ks>=0.05‚Ì‚ÝŽg—p
        fit_dfs = []
        self.fit_cols = []
        for col in cols:
            use_cols = corr_df.columns[corr_df[col].values.astype(np.bool)].tolist()
            use_cols.append(col)
            self.fit_cols.append(use_cols)
            fit_dfs.append(df.loc[:,use_cols])
        self.base_models_ = [{name:self.base_models_[name] for name in col } for col in self.fit_cols]
        self.choice_model_len = len(self.base_models_)
        return fit_dfs
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        pred_df = pd.DataFrame()
        for name, meta_model in self.meta_models_.items():
            for k in range(self.choice_model_len):
                meta_features = np.column_stack([
                    np.column_stack([model.predict_proba(X)[:,1] for model in models]).mean(axis=1)
                    for models in self.base_models_[k].values()])
                pred = meta_model[k].predict_proba(meta_features)[:,1]
                pred_df[f"meta:{name}, base:"+str(self.fit_cols[k])] = pred
        return pred_df
class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(y_preds.shape[0])]
        # Calculate the weighted prediction
        # print(y_preds.shape)
        # print(len(weights))
        weighted_pred = np.average(y_preds, axis=0, weights=weights)

        # Calculate the ROC AUC score for the weighted prediction
        score = loss_func(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=300, direction='minimize'):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction=direction)
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(y_preds.shape[0])]
    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(y_preds, axis=0, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=300):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights

class DataProcessor():
    def __init__(self, num_cols, cat_cols, use_cols, agg_col, agg_func, group_cols, comb_cat_cols, is_agg=False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.use_cols = use_cols
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.group_cols = group_cols
        self.comb_cat_cols = comb_cat_cols
        self.en_num = RobustScaler()
        self.en_cat = LabelEncoder()
        self.is_agg = is_agg
    def extra_fe(self, df):
        return df
    def _fe(self, df):
        df = self.extra_fe(df)
        df[self.num_cols] = self.en_num.fit_transform(df[self.num_cols])
        return df
    def transform(self, trn, tst, ori):
        if self.is_agg:
            agg_train, agg_ori, agg_test = [], [], []
            for group_col in self.group_cols:
                agg_extractor = AggFeatureExtractor(group_col=group_col, agg_col=self.agg_col, agg_func=self.agg_func)
                agg_extractor.fit(pd.concat([trn[self.use_cols], ori[self.use_cols] , tst], axis=0).reset_index(drop=True))
                agg_train.append(agg_extractor.transform(trn[self.use_cols]))
                agg_ori.append(agg_extractor.transform(ori[self.use_cols]))
                agg_test.append(agg_extractor.transform(tst))
            trn = pd.concat([trn] + agg_train, axis=1)
            ori = pd.concat([ori] + agg_ori, axis=1)
            tst = pd.concat([tst] + agg_test, axis=1)
            
        trn = self.remove_col(trn)
        tst = self.remove_col(tst)
        ori = self.remove_col(ori)
        trn = self._fe(trn)
        tst = self._fe(tst)
        ori = self._fe(ori)
        return trn, tst, ori
    def remove_col(self,df):
        bool_df = ((df==0).sum() == len(df.index.tolist()))
        df = df.loc[:,-bool_df==True]
        df = df.loc[:, -df.isna().any()]
        return df

class AggFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, agg_col, agg_func):
        self.group_col = group_col
        self.group_col_name = ''
        for col in group_col:
            self.group_col_name += col
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.agg_df = None
        self.medians = None
        
    def fit(self, X, y=None):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        
        self.agg_df = X.groupby(group_col)[agg_col].agg(agg_func)
        self.agg_df.columns = [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]
        self.medians = X[agg_col].median()
        
        return self
    
    def transform(self, X):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        agg_df = self.agg_df
        medians = self.medians
        
        X_merged = pd.merge(X, agg_df, left_on=group_col, right_index=True, how='left')
        X_merged.fillna(medians, inplace=True)
        X_agg = X_merged.loc[:, [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]]
        
        return X_agg
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_agg = self.transform(X)
        return X_agg

def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def loss_func(y_true, y_pred):
    loss = log_loss(y_true, y_pred)
    return loss

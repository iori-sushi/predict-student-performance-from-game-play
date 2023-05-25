# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Preparation

# ## Loading packages & Setting variables

# +
import pandas as pd
import numpy as np
import warnings
import gc
import os
import random
import pickle
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import plotly.express as px
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.simplefilter("ignore")
pd.options.display.max_columns=1000
torch.autograd.set_detect_anomaly(True)

class CFG:
    # hyper parameters
    EPOCHS = 50
    PRETRAIN_EPOCHS = 100
    THRESHOLD = .5
    BATCH_SIZE = 2**6
    WARM_UP = min(20, EPOCHS//10) # do not stop early
    EARLY_STOPPING_ROUNDS = max(EPOCHS//20, WARM_UP)
    LEARNING_RATE = .01
    SEED = 1
    
    # utils
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VERBOSE = True # or False
    PREDICT_ALL = False # True or False
    
    SUBMISSION_MODE = False
    CHECKPOINT = False
    RESUME = True
    PRETRAINING_OMIT = True
    TRAINING_OMIT = False
    if SUBMISSION_MODE:
        CHECKPOINT = False
        RESUME = False
        PRETRAINING_OMIT = True
        TRAINING_OMIT = True
        VERBOSE = False
    elif CHECKPOINT:
        RESUME = False
        PRETRAINING_OMIT = False
        TRAINING_OMIT = False
    else:
        RESUME = True
        if TRAINING_OMIT:
            PRETRAINING_OMIT = True
    INPUT = '../input'
    CHECKPOINT_PATH = '../checkpoint/SAINT'

if CFG.SUBMISSION_MODE:
    import jo_wilder_310
    
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
torch.cuda.manual_seed(CFG.SEED)

lq_dict = {
    "0-4":["q"+str(i) for i in range(1,4)],
    "5-12":["q"+str(i) for i in range(4,14)],
    "13-22":["q"+str(i) for i in range(14,19)]
}
questions = ['q'+str(i) for i in range(1,19)]

# %env CUDA_LAUNCH_BLOCKING=1
# #%env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# -

# ## Functions & Classes

# ### Data Processing Class

class DataProcessing():
    
    def __init__(self):
        self.agg_units = ["session_id", "level_group"]
    
    def read_table(self, usecols, inc_agg_units=True):
        if inc_agg_units:
            base_cols = self.agg_units + ["level"]
            usecols += base_cols
        if self.path:
            data = pd.read_csv(self.path, usecols=usecols)
        elif not self.data is None:
            data = self.data[usecols]
                
        return data

    def agg_by_level(self, data, agg_func):
        data = data.copy()
        if type(agg_func) == dict:
            data = data.groupby(self.agg_units + ["level"]).agg(**agg_func)
        elif type(agg_func) == list:
            data = data.groupby(self.agg_units + ["level"]).agg(agg_func)
            data.columns = [col[0]+"_"+col[1] for col in data.columns]
        else:
            data = data.groupby(self.agg_units + ["level"]).agg(agg_func)
        """
        tmp = data.diff().fillna(0)
        tmp.columns = [col+"_diff" for col in data.columns]
        data = pd.concat([data, tmp], axis=1)
        del tmp
        gc.collect()
        """
        data = data.reset_index()
        data = data.pivot(index=self.agg_units, columns="level")
        data.columns = [f"{col[0]}_level{col[1]}" for col in data.columns]

        return data.sort_index()

    def agg_by_level_group(self, data, agg_func):
        data = data.copy()
        if CFG.SUBMISSION_MODE:
            if type(agg_func) == dict:
                data = data.groupby(self.agg_units).agg(**agg_func)
            elif type(agg_func) == list:
                data = data.groupby(self.agg_units).agg(agg_func)
                data.columns = [col[0]+"_"+col[1] for col in data.columns]
            else:
                data = data.groupby(self.agg_units).agg(agg_func)
        else:        
            data2 = data.query("level_group!='13-22'")
            data2.level_group = "5-12"
            data1 = data.query("level_group=='0-4'")
            data1.level_group = "0-4"
            data.level_group = "13-22"
            if type(agg_func) == dict:
                data = data.groupby(self.agg_units).agg(**agg_func)
                data2 = data2.groupby(self.agg_units).agg(**agg_func)
                data1 = data1.groupby(self.agg_units).agg(**agg_func)
            elif type(agg_func) == list:
                cols = [col for col in data.columns if col != "level"]
                data = data[cols].groupby(self.agg_units).agg(agg_func)
                data.columns = [col[0]+"_"+col[1] for col in data.columns]
                data2 = data2[cols].groupby(self.agg_units).agg(agg_func)
                data2.columns = [col[0]+"_"+col[1] for col in data2.columns]
                data1 = data1[cols].groupby(self.agg_units).agg(agg_func)
                data1.columns = [col[0]+"_"+col[1] for col in data1.columns]
            else:        
                cols = [col for col in data.columns if col != "level"]
                data = data[cols].groupby(self.agg_units).agg(agg_func)
                data2 = data2[cols].groupby(self.agg_units).agg(agg_func)
                data1 = data1[cols].groupby(self.agg_units).agg(agg_func)

            data = pd.concat([data1, data2, data], axis=0).fillna(float("inf"))
        
            del data1, data2
            gc.collect()
        
        return data.sort_index()
    
    def fit_transform(self, path=None, data=None):
        self.fit = True
        if path:
            self.path = path
            self.data = None
        elif not data is None:
            self.path = None
            self.data = data
        else:
            assert (self.path is None and self.data is None, "both path and data is None")
        
        data = self.transform(path)
        self.fit = False
        
        return data 
    
    def transform(self, path=None, data=None, decided_level_group=None):
        if CFG.SUBMISSION_MODE:
            self.data = data
            self.path = None
            self.agg_units = ["session_id"]
        else:
            if path:
                self.path = path
                self.data = None
            elif not data is None:
                self.path = None
                self.data = data
            else:
                assert (self.path is None and self.data is None, "both path and data is None")

        self.level_group = decided_level_group

        datas = []
        
        """elapsed_time"""
        datas.append(
            self.col_processing(
                ["elapsed_time"],
                agg_func = dict(
                    elapsed_time_max=("elapsed_time", "max"),
                    elapsed_time_mean=("elapsed_time", "mean"),
                    elapsed_time_median=("elapsed_time", "median"),
                    elapsed_time_75percentile=("elapsed_time", lambda x: np.percentile(x, 75)),
                    elapsed_time_25percentile=("elapsed_time", lambda x: np.percentile(x, 25)),
                    elapsed_time_std=("elapsed_time", "std"),
                    logs=("elapsed_time", "count"),
                )
            )
        )
        if CFG.VERBOSE:
            print("elapsed_time processing has finished!")

        """fqids"""
        #datas.append(self.col_processing(["fqid"], "nunique"))
        #if CFG.VERBOSE:
        #    print("fqid processing has finished!")
        
        #datas.append(self.col_processing(["room_fqid"], "nunique"))
        #if CFG.VERBOSE:
        #    print("rfqid processing has finished!")
        
        #datas.append(self.col_processing(["text_fqid"], "nunique"))
        #if CFG.VERBOSE:
        #    print("tfqid processing has finished!")
        
        """coor"""
        datas.append(self.col_processing(
            ["screen_coor_x", "screen_coor_y", "room_coor_x", "room_coor_y"],
            ["mean", "std", "max", "min", "median"]))
        if CFG.VERBOSE:
            print("coor processing has finished!")
        
        """text"""
        datas.append(self.text_processing())
        if CFG.VERBOSE:
            print("text processing has finished!")
        
        """name"""
        datas.append(self.str_col_processing("name"))
        if CFG.VERBOSE:
            print("name processing has finished!")
        
        """event_name"""
        datas.append(self.str_col_processing("event_name", "_"))
        if CFG.VERBOSE:
            print("event_name processing has finished!")
            
        """others"""
        datas.append(
            self.col_processing(["page", "hover_duration"],
            dict(
                page_max=("page", "max"),
                hover_duration_max=("hover_duration", "max"),
            )
        ))
        if CFG.VERBOSE:
            print("others processing has finished!")

        """concate data"""
        datas = pd.concat(datas, axis=1)
        
        """final process"""
        datas = self.make_cols_same(datas)
        datas = self.standardizing_and_masking(datas)
        datas = self.feature_elimination(datas, pca_components=10)
        if CFG.VERBOSE:
            print("final processing has finished!\n")
            
        return datas
    
    def str_col_processing(self, col, split_str=None, agg_func=["sum", "mean"]):
        base_data = self.read_table([col])
        data = pd.DataFrame({col: base_data[col].unique()})
        
        if split_str:
            split = data[col].str.split(split_str, expand=True)
            cols = [col+str(i+1) for i in range(split.shape[1])]
            data[cols] = split
            del split
            gc.collect()
            data = pd.get_dummies(data.set_index(col), sparse=False).reset_index()
        else:
            tmp = pd.get_dummies(data[col], sparse=False)
            data = pd.concat([data, tmp], axis=1)
        
        data = pd.merge(base_data, data, how="left", on=col)
        
        del base_data
        gc.collect()
        
        data = self.agg_by_level_group(data, agg_func)
        return data
    
    def col_processing(self, cols, agg_func=["sum", "mean"], also_by_level=True):
        data = self.read_table(cols)
        if also_by_level:
            datas = []
            datas.append(self.agg_by_level_group(data, agg_func))
            datas.append(self.agg_by_level(data, agg_func))
            data = pd.concat(datas, axis=1)
            del datas
            gc.collect()
        else:   
            data = self.agg_by_level_group(data, agg_func)
        return data
    
    def text_processing(self, also_by_level=True):
        data = self.read_table(["text"])
        data["text_byte"] = data.text.str.len().tolist()
        data["text_len"] = data.text.str.split(" ").apply(lambda x: len(x) if type(x)==list else 0).tolist()
        data.text = data.text.notnull().astype(int).tolist()
        
        agg_func = dict(
            text_len_max = ("text_len", "max"),
            text_len_mean = ("text_len", "mean"),
            text_len_std = ("text_len", "std"),
            text_byte_max = ("text_byte", "max"),
            text_byte_mean = ("text_byte", "mean"),
            text_byte_std = ("text_byte", "std"),
            text_count_mean = ("text", "mean"),
            text_count_std = ("text", "std"),
            text_count = ("text", "sum"),
        )
        
        if also_by_level:
            datas = []
            datas.append(self.agg_by_level_group(data, agg_func))
            datas.append(self.agg_by_level(data, agg_func))
            data = pd.concat(datas, axis=1)
            del datas
            gc.collect()
        else:
            data = self.agg_by_level_group(data, agg_dict)
        
        return data
    
    def make_cols_same(self, data):
        if not self.fit:
            cols = self.columns
        
            if self.level_group:
                lg = self.level_group
                drop_cols = []
                for col in data.columns:
                    if not col in cols[lg]:
                        drop_cols.append(col)
                data = data.drop(drop_cols, axis=1)

                for col in cols[lg]:
                    if not col in data.columns:
                        data[col] = 0

                return data[cols[lg]]

            else:
                datas = {}
                for lg in lq_dict.keys():
                    dlg = data.query(f"level_group=='{lg}'")
                    
                    drop_cols = []
                    for col in dlg.columns:
                        if not col in cols[lg]:
                            drop_cols.append(col)
                    dlg = dlg.drop(drop_cols, axis=1)

                    for col in cols[lg]:
                        if not col in dlg.columns:
                            dlg[col] = 0
                            
                    datas[lg] = dlg
                return datas
            
        else:
            self.columns = {}
            datas = {}
            useless_from_lgb = []

            for lg in lq_dict.keys():
                dlg = data.query(f"level_group=='{lg}'")
                
                all_one_value = dlg.nunique()
                all_one_value = all_one_value[all_one_value<2].index
                dlg = dlg.drop(all_one_value, axis=1)
                dlg = dlg.drop([col for col in useless_from_lgb if col in dlg.columns], axis=1)
                
                print(f"{lg}: all_one_value COLS ARE {all_one_value}")
                
                datas[lg] = dlg
                self.columns[lg] = dlg.columns
            return datas
 
    def feature_elimination(self, datas, pca_components=10):
        top30_useful_lgb = {
            "0-4":[
                'basic_sum',
                'elapsed_time_std_level4',
                'elapsed_time_std_level2',
                'logs',
                'screen_coor_x_median_level4',
                'event_name1_notification_mean',
                'room_coor_y_median_level4',
                'elapsed_time_std_level1',
                'room_coor_y_max',
                'room_coor_y_max_level3',
                'screen_coor_x_median_level0',
                'screen_coor_x_std_level2',
                'room_coor_x_mean_level4',
                'room_coor_x_min_level4',
                'screen_coor_y_median_level3',
                'text_len_mean_level3',
                'room_coor_y_min_level0',
                'room_coor_x_std',
                'text_byte_std_level2',
                'room_coor_y_std_level0',
                'screen_coor_x_std_level1',
                'room_coor_x_median_level3',
                'room_coor_x_mean_level3',
                'room_coor_y_std',
                'room_coor_y_min_level3',
                'text_len_std_level2',
                'event_name2_click_mean',
                'text_len_mean_level2',
                'room_coor_x_mean_level1',
                'screen_coor_y_max_level0'
            ],
            "5-12":[
                'basic_sum',
                'logs',
                'logs_level11',
                'event_name1_object_sum',
                'elapsed_time_std_level8',
                'elapsed_time_std_level12',
                'logs_level9',
                'event_name1_notification_mean',
                'hover_duration_max_level11',
                'event_name2_click_sum',
                'text_count',
                'text_byte_mean_level6',
                'hover_duration_max_level9',
                'elapsed_time_std_level10',
                'text_len_std_level6',
                'room_coor_x_min_level11',
                'room_coor_y_max',
                'room_coor_x_min_level9',
                'room_coor_x_min',
                'room_coor_x_max_level9',
                'room_coor_x_max_level8',
                'room_coor_x_max_level7',
                'hover_duration_max_level8',
                'elapsed_time_std_level5',
                'hover_duration_max_level7',
                'screen_coor_x_min_level7',
                'room_coor_x_max_level5',
                'room_coor_x_min_level10',
                'elapsed_time_std_level11',
                'room_coor_x_max_level6'
            ],
            "13-22":[
                'event_name1_checkpoint_mean',
                'room_coor_y_max_level15',
                'logs',
                'room_coor_y_mean_level22',
                'text_byte_mean_level21',
                'elapsed_time_std',
                'screen_coor_x_median_level22',
                'elapsed_time_std_level21',
                'hover_duration_max_level20',
                'screen_coor_x_min_level20',
                'room_coor_x_max_level22',
                'room_coor_x_max_level17',
                'screen_coor_x_std_level17',
                'screen_coor_x_mean_level21',
                'elapsed_time_median',
                'room_coor_x_max',
                'elapsed_time_std_level22',
                'text_byte_std',
                'room_coor_y_min_level21',
                'hover_duration_max_level15',
                'screen_coor_x_max_level15',
                'elapsed_time_std_level20',
                'basic_sum',
                'room_coor_y_std_level22',
                'room_coor_x_std_level18',
                'text_count_level20',
                'room_coor_y_min_level18',
                'screen_coor_x_median_level17',
                'room_coor_x_min',
                'elapsed_time_std_level19'
            ]
        }
        
        if self.level_group:
            level_group = self.level_group
            data, data_mask = datas
            tmp = self.pca[level_group].transform(data)
            tmp = pd.DataFrame(tmp, columns=[f"pca_{i}" for i in range(pca_components)])
            data = data[top30_useful_lgb[level_group]]
            data_mask = data_mask[top30_useful_lgb[level_group]]
            for col in tmp.columns:
                data[col] = tmp[col].values
                data_mask[col] = 1
            
            del tmp
            gc.collect()
            
            res = (data, data_mask)
        
        else:
            res = {}

            if self.fit:
                self.pca = {}

            for level_group in lq_dict.keys():
                data, data_mask = datas[level_group]

                if not self.fit:
                    tmp = self.pca[level_group].transform(data)
                    tmp = pd.DataFrame(tmp, columns=[f"pca_{i}" for i in range(pca_components)])

                else:
                    pca = PCA(n_components=pca_components, random_state=CFG.SEED)
                    tmp = pca.fit_transform(data)
                    tmp = pd.DataFrame(tmp, columns=[f"pca_{i}" for i in range(pca_components)])
                    self.pca[level_group] = pca

                data = data[top30_useful_lgb[level_group]]
                data_mask = data_mask[top30_useful_lgb[level_group]]
                for col in tmp.columns:
                    data[col] = tmp[col].values
                    data_mask[col] = 1

                del tmp
                gc.collect()

                res[level_group] = (data, data_mask)

            self.num_features = len(data.columns)
        
        return res
        
    def standardizing_and_masking(self, datas):
        if self.level_group:
            data_mask = datas.notnull().astype(int)
            data = (datas - self.means[self.level_group]) / self.stds[self.level_group]
            datas = (data.fillna(0), data_mask)
            
        else:
            if self.fit:
                self.means = {}
                self.stds = {}

            for k, data in datas.items():
                if self.fit:
                    self.means[k] = data.apply(lambda x: x.mean(axis=0))
                    self.stds[k] = data.apply(lambda x: x.std(axis=0))

                data_mask = data.notnull().astype(int)
                #data_mask["CLS"] = 1
                #data_mask = data_mask[["CLS"] + [col for col in data_mask.columns if col != "CLS"]]

                data = (data - self.means[k]) / self.stds[k]
                #data["CLS"] = 0
                #data = data[["CLS"] + [col for col in data.columns if col != "CLS"]]

                datas[k] = (data.fillna(0), data_mask)
        return datas


# ### Dataloader Creating Functions

# +
def create_loader_train(X, X_mask, y, level_group=None, train_rate=.9, predict_all=False):
    if level_group:
        X = X.query(f'level_group=="{level_group}"')
        X = X.reset_index().drop("level_group", axis=1).set_index("session_id")
        X_mask = X_mask.query(f'level_group=="{level_group}"')
        X_mask = X_mask.reset_index().drop("level_group", axis=1).set_index("session_id")
    
    if predict_all:
        pass
    else:
        y = y[lq_dict[level_group]]
        
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-train_rate, random_state=CFG.SEED)
    res = msss.split(X.values, y.values)
    for train_index, val_index in res:
        train_index = train_index.tolist()
        val_index = val_index.tolist()

    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(X.iloc[train_index, :].values),
            torch.Tensor(X_mask.iloc[train_index, :].values).type(torch.int),
            torch.Tensor(y.iloc[train_index, :].values),
        ),
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(X.iloc[val_index, :].values),
            torch.Tensor(X_mask.iloc[val_index, :].values).type(torch.int),
            torch.Tensor(y.iloc[val_index, :].values),
        ),
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    del train_index
    gc.collect()

    return train_loader, val_loader

def create_loader_test(X, X_mask, level_group=None):
    if CFG.SUBMISSION_MODE:
        pass
    elif level_group:
        X = X.query(f'level_group=="{level_group}"')
        X = X.reset_index().drop("level_group", axis=1).set_index("session_id")
        X_mask = X_mask.query(f'level_group=="{level_group}"')
        X_mask = X_mask.reset_index().drop("level_group", axis=1).set_index("session_id")

    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(X.values),
            torch.Tensor(X_mask.values).type(torch.int),
        ),
        batch_size=1000, shuffle=False
    )
    
    return test_loader


# -

# ### MultiOutputsModel

# +
class MultiOutputsModel(nn.Module):
    def __init__(self, num_features=10, num_outputs=10,
                 units=[64, 128, 64, 64, 32],
                 units_sub=[16, 32, 16, 8, 1],
                 pretrained_model=None,
                 dim=None
                ):
        super(MultiOutputsModel, self).__init__()
        self.name = "MultiOutputsModel"
        self.num_features = num_features
        self.dim = dim
        self.num_outputs = num_outputs
        
        if pretrained_model:
            self.embedding = pretrained_model.embedding
            self.saint = pretrained_model.saint
            
            self.each_dim_models =  nn.ModuleList([
                EachSAINTDimModel(
                    num_features=num_features,
                    num_outputs=num_outputs,
                    units=units,
                    units_sub=units_sub,
                    dim=dim
                )
                for _ in range(dim)
            ])
            
            self.last_output_layers = nn.ModuleList([
                SigmoidLayer(dim, 1) for _ in range(num_outputs)
            ])
        
    def register_pretrained_model(self, pretrained_model):
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.__init__(
            num_features=pretrained_model.embedding.num_features,
            dim=pretrained_model.dim,
            num_outputs=self.num_outputs,
            pretrained_model=pretrained_model,
        )
        
    def forward(self, x, x_mask):
        x = self.embedding(x.clone(), x_mask.clone())
        x = self.saint(x)
        #x = torch.mean(x, dim=1) #x[:,0,:]

        dim_outputs = []
        for i_dim, l in enumerate(self.each_dim_models):
            dim_outputs.append(l(x[:,:,i_dim].clone()))
        x = torch.stack(dim_outputs)
        x = torch.transpose(x, 0, 1)
        
        last_outputs = []
        for i_output, l in enumerate(self.last_output_layers):
            last_outputs.append(l(x[:,:,i_output].clone()))
            
        x = torch.concat(last_outputs, dim=1)
        return x

class EachSAINTDimModel(nn.Module):
    def __init__(self, num_features=10, num_outputs=10,
                 units=[128, 256, 128, 64, 32],
                 units_sub=[64, 16, 32, 16, 1],
                 dim=None
                ):
        super(EachSAINTDimModel, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.units = [self.num_features] + units
        self.units_sub = [self.units[-1]] + units_sub
        
        self.l1_bn = nn.BatchNorm1d(self.units[0])
        self.l1 = nn.Linear(self.units[0], self.units[1])
        nn.init.xavier_normal_(self.l1.weight)
        
        self.ls = nn.ModuleList([
            LeakyReLULayer(self.units[i+1], self.units[i+2])
            if i%2 == 1 else ResidualBlock(self.units[i+1], self.units[i+2])
            for i in range(len(self.units)-2)
        ])
        self.num_outputs = num_outputs
        
        self.subs = nn.ModuleList([
            MultiOutputsModelSub(self.units[-1])
            for _ in range(num_outputs)
        ])
        
    def forward(self, x):
        x = F.leaky_relu(self.l1(self.l1_bn(x)))
        x = F.dropout(x, .1, training=self.training)
        for l in self.ls:
            x = l(x)

        res = []
        for sub in self.subs:
            res.append(sub(x.clone()))            
        x = torch.concat(res, dim=1)
        
        return x
    
class MultiOutputsModelSub(nn.Module):
    def __init__(self, num_inputs, units_sub=[64, 16, 32, 16, 1]):
        super(MultiOutputsModelSub, self).__init__()
        self.geglu_layer = GeGLULayer(num_inputs, units_sub[0])
        self.units_sub = units_sub
        self.ls_sub =  nn.ModuleList([
            LeakyReLULayer(self.units_sub[i], self.units_sub[i+1])
            if i < len(self.units_sub)-1 else
            SigmoidLayer(self.units_sub[i], self.units_sub[i+1])
            for i in range(len(self.units_sub)-1)
        ])
                
    def forward(self, x):
        x = self.geglu_layer(x)
        for i, l in enumerate(self.ls_sub):
            if i+1 < len(self.units_sub)-1:
                x = l(x)
            else:
                x = F.dropout(x, .2, training=self.training)
                x = l(x)
        return x

class LeakyReLULayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LeakyReLULayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)
        self.linear = nn.Linear(num_inputs, num_outputs)
                
    def forward(self, x):
        x = self.linear(self.bn(x))
        x = F.leaky_relu(x)
        return x

class SigmoidLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SigmoidLayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)
        self.linear = nn.Linear(num_inputs, num_outputs)
                
    def forward(self, x):
        x = self.linear(self.bn(x))
        x = F.sigmoid(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ResidualBlock, self).__init__()
        self.layer1 = LeakyReLULayer(num_inputs, 32)
        self.layer2 = LeakyReLULayer(32, 16)
        self.layer3 = LeakyReLULayer(16, num_inputs)
        self.geglu = GeGLULayer(num_inputs, num_outputs)
                
    def forward(self, x):
        x_mlp = self.layer1(x.clone())
        x_mlp = self.layer2(x_mlp)
        x += self.layer3(x_mlp)
        x = self.geglu(x)
        return x
    
class GeGLULayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GeGLULayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)
        self.linear = nn.Linear(num_inputs, num_outputs*2)
        self.geglu = GEGLU()
                
    def forward(self, x):
        x = self.linear(self.bn(x))
        x = self.geglu(x)
        return x


# -

# ### SAINT

# +
class MLP(nn.Module):
    def __init__(self, three_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(three_dim[0], three_dim[1]),
            nn.ReLU(),
            nn.Linear(three_dim[1], three_dim[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, num_features, dim=32):
        super(EmbeddingLayer, self).__init__()
        self.dim = dim
        self.num_features = num_features
        self.MLPs = nn.ModuleList([MLP([1, 100, dim]) for _ in range(num_features)])
        mask_offset = F.pad(
            torch.Tensor(num_features).fill_(2).type(torch.int8),
            (1,0),
            value=0
        ).cumsum(dim=-1)[:-1].to(CFG.DEVICE)
        self.register_buffer('mask_offset', mask_offset)
        self.embedding_mask = nn.Embedding(num_features*2, dim)
        
    def forward(self, x, x_mask):
        x_enc = torch.empty(*x.shape, self.dim).to(CFG.DEVICE)
        for i in range(self.num_features):
            x_enc[:,i,:] = self.MLPs[i](x[:,i])
        x_enc[x_mask==0] = self.embedding_mask(x_mask+self.mask_offset.type_as(x_mask))[x_mask==0]
        return x_enc
    
class SAINT(nn.Module):
    def __init__(self, num_features, dim=32, heads=8, dim_head=16, attention_dropout=.1, ff_dropout=.1):
        super(SAINT, self).__init__()
        self.msa = Attention(dim, heads, dim_head, attention_dropout)
        self.ff1 = FeedForward(dim, dropout=ff_dropout)
        self.misa = Attention(dim*num_features, heads, 64, attention_dropout)
        self.ff2 = FeedForward(dim*num_features, dropout=ff_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim*num_features)
        self.norm4 = nn.LayerNorm(dim*num_features)
        
    def forward(self, x):
        shape = x.shape
        
        x = self.norm1(x)
        x += self.msa(x.clone())
        
        x = self.norm2(x)
        x += self.ff1(x.clone())
        x = x.view(1, shape[0], -1)
        
        x = self.norm3(x)
        x += self.misa(x.clone())
        
        x = self.norm4(x)
        x += self.ff2(x.clone())
        x = x.view(*shape)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim=32, heads=8, dim_head=16, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.reshaping = lambda x, shape: torch.permute(x, (0,2,1)).reshape(shape[0], heads, shape[1], -1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim =-1)
        shape = q.shape
        q, k, v = self.reshaping(q, shape), self.reshaping(k, shape), self.reshaping(v, shape)
        sim = torch.einsum('b h i d, b h j d -> b h i j', (q,k)) * self.scale
        attention = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out_shape = out.shape
        out = out.view(out_shape[0], out_shape[2], -1)
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim=32, mult=4, dropout=.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, dim*mult*2)
        self.linear2 = nn.Linear(dim*mult, dim)
        self.dropout = dropout
        self.geglu = GEGLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.geglu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x*F.gelu(gates)
    
class Constrastive(nn.Module):
    def __init__(self, num_features, dim=32, temperature=0.7):
        super(Constrastive, self).__init__()
        units = [
            dim*num_features,
            6*dim*num_features//5,
            dim*num_features//2
        ]
        self.projection_head_true = MLP(units)
        self.projection_head_false = MLP(units)
        self.reshaping = lambda x: (x / x.norm(dim=-1, keepdim=True)).flatten(1,2)
        self.temperature = temperature
        
    def forward(self, x_true, x_false):
        x_true, x_false = self.reshaping(x_true), self.reshaping(x_false)
        x_true = F.normalize(self.projection_head_true(x_true), dim=-1).flatten(1)
        x_false = F.normalize(self.projection_head_false(x_false), dim=-1).flatten(1)  
        logits = x_true @ x_false.t() / self.temperature
        logits_ =  x_false @ x_true.t() / self.temperature
        targets = torch.arange(logits.size(0)).to(CFG.DEVICE)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss_ = F.cross_entropy(logits_, targets, reduction="mean")
        loss = (loss + loss_) / 2
        return loss
    
class Denoising(nn.Module):
    def __init__(self, num_features, dim=32):
        super(Denoising, self).__init__()
        self.num_features = num_features
        self.MLPs = nn.ModuleList([
            MLP([dim, dim*5, 1])
            for i in range(num_features)
        ])
        
    def forward(self, x, x_original):
        x = [
            self.MLPs[i](x[:,i,:])
            for i in range(1, self.num_features) # except for CLS
        ]
        x = torch.cat(x,dim=1)
        loss = F.mse_loss(x, x_original[:,1:], reduction="mean")
        return loss


# -

# ### Pretraining Class

class PretrainingSAINT(nn.Module):
    def __init__(self, num_features, dim=32):
        super(PretrainingSAINT, self).__init__()
        self.dim = dim
        self.embedding = EmbeddingLayer(num_features, dim)
        self.saint = SAINT(num_features, dim)
        self.contrastive = Constrastive(num_features, dim)
        self.denoising = Denoising(num_features, dim)
    
    def cutmix(self, x, m=.1):
        x = x.clone()
        x_shuffle = x[torch.randperm(x.shape[0]),:]
        random_choice = torch.from_numpy(np.random.choice(2,(x.shape),p=[m,1-m]))
        x[random_choice==0] = x_shuffle[random_choice==0]
        return x

    def mixup(self, x_enc, alpha=.3):
        index = torch.randperm(x_enc.shape[0])
        x_enc = alpha*x_enc + (1-alpha)*x_enc[index, :]
        return x_enc
    
    def forward(self, x, x_mask):
        x_true = self.embedding(x.clone(), x_mask.clone())
        x_true = self.saint(x_true)
        
        x_false = self.cutmix(x.clone())
        x_false = self.embedding(x_false, x_mask.clone())
        x_false = self.mixup(x_false)
        x_false = self.saint(x_false)
        
        contrastive_loss = self.contrastive(x_true.clone(), x_false.clone())
        denoising_loss = self.denoising(x_false.clone(), x.clone())
        
        return contrastive_loss, denoising_loss


# ### Pretraining Function

def pretraining(train_loader, level_group, epochs=CFG.EPOCHS, lambda1=.5, lambda2=10, omit=CFG.PRETRAINING_OMIT):
    num_features = train_loader.dataset.tensors[0].shape[1]
    model = PretrainingSAINT(num_features).to(CFG.DEVICE)
    checkpoint_path = f"{CFG.CHECKPOINT_PATH}/pretrain_params_{level_group}.pth"
    
    if omit:
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
        losses = [10**10]
        model.train()

        print("Pretraining is starting...") 
        for epoch in tqdm(range(epochs)):
            loss_sum = 0.0
            contrastive_loss_sum = 0.0
            denoising_loss_sum = 0.0
            for i, (X, X_mask, _) in enumerate(train_loader):
                X, X_mask = X.to(CFG.DEVICE), X_mask.to(CFG.DEVICE)
                contrastive_loss, denoising_loss = model(X, X_mask)
                loss = lambda1 * contrastive_loss + lambda2 * denoising_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                contrastive_loss_sum += (lambda1 * contrastive_loss).item()
                denoising_loss_sum += (lambda2 * denoising_loss).item()
                loss_sum += loss.item()

            epoch_loss = loss_sum/i
            if epoch_loss < min(losses):
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            losses.append(epoch_loss)

            print(
                f"Epoch {epoch+1}/{epochs}: loss {epoch_loss: .4f} (contrastive_loss {contrastive_loss_sum/i: .4f}, denoising_loss {denoising_loss_sum/i: .4f})"
            )

        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Pretraining has finished.\nBest Epoch is {best_epoch} with loss {min(losses)} !")
    
    return model


# ### Other Functions

# +
def calc_weight(data_loader):
    count1 = torch.sum(data_loader.dataset.tensors[2], dim=0)
    count_all = data_loader.dataset.tensors[2].shape[0]
    count0 = count_all - count1
    weight = count0 / count1
    
    del count1, count_all, count0
    gc.collect()
    
    return weight

def calc_matrix(x):
    x = torch.where((x > CFG.THRESHOLD), 1, -1)
    x_t = x.T
    length = x_t.shape[0]
    res = []

    for i in range(length):
        res.append(x *  x.T[i].unsqueeze(-1))
    res = torch.concat(res, dim=1)
    res = res.view(-1, length, length)
    res = torch.where(res==-1, 0.0, 1.0)
    res = torch.mean(res, axis=0).to(CFG.DEVICE)
        
    return res

def explore_threshold(model, val_loader):
    preds, true_values = [], []
    for i, (x, x_mask, t) in enumerate(val_loader):
        x, x_mask, t = x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE), t.to(CFG.DEVICE)
        y = model(x, x_mask).view(-1)
        preds += y
        true_values += t

    preds = torch.stack(preds).view(-1).detach().cpu().numpy()
    true_values = torch.stack(true_values).view(-1).detach().cpu().numpy()
    px.box(x=true_values, y=preds, points="all").show()

    f1s = []
    for th in range(0, 101, 1):
        f1 = f1_score(true_values, (preds > (th/100)).astype(int), average="macro")
        f1s.append(f1)
    f1s = pd.DataFrame({"threshold":[i/100 for i in range(0, 101, 1)], "f1":f1s}).set_index("threshold")
    best_threshold = f1s[f1s.f1.apply(lambda x: x//.01)==f1s.f1.apply(lambda x: x//.01).max()].index.max()
    px.line(f1s, title=f"Best Threshold is {best_threshold}.").show()

    return best_threshold


# -

# ### Training Function

def training(train_loader, val_loader, model_class, pretrained_model, omit=CFG.TRAINING_OMIT):
    num_features = train_loader.dataset.tensors[0].shape[1]
    num_outputs = val_loader.dataset.tensors[2].shape[1]
    level_group = '0-4' if num_outputs == 3 else '5-12' if num_outputs == 10 else '13-22' if num_outputs == 5 else None
    model = model_class(num_features, num_outputs)
    model.register_pretrained_model(pretrained_model)
    model = model.to(CFG.DEVICE)
    checkpoint_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_bestmodel{'_'+level_group if level_group else ''}{'_all' if CFG.PREDICT_ALL else ''}.pth"
    
    if omit:
        checkpoint_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_bestmodel{'_'+level_group if level_group else ''}{'_all' if CFG.PREDICT_ALL else ''}.pth"
        model.load_state_dict(torch.load(checkpoint_path))
        try:
            with open(f"{CFG.CHECKPOINT_PATH}/{model.name}_thresholds{'_all' if CFG.PREDICT_ALL else ''}.pickle", "rb") as f:
                best_thresholds = pickle.load(f)[level_group]
        except Exception as e:
            print(e)
            best_thresholds = None
        return model, best_thresholds
    else:
        print("Training is starting...") 
        
        weight_ratio = calc_weight(train_loader).to(CFG.DEVICE)
        early_stopping_count = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=.005)
        loss_trains, loss_vals, f1_trains, f1_vals = [], [], [], []
        loss_func = nn.BCELoss(reduction='none')

        for epoch in range(CFG.EPOCHS):
            #"""
            #train
            #"""
            model.train()
            loss_train = 0
            preds, true_values = [], []
            for i, (x, x_mask, t) in enumerate(train_loader):
                x, x_mask, t = x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x, x_mask)
                preds += y
                true_values += t
                
                weight = (lambda w: torch.where(w==0, 1, w))(t*weight_ratio)
                loss = torch.mean(loss_func(y, t)*weight) / num_outputs
                loss_train += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            loss_train /= i
            loss_trains.append(loss_train)
            preds = torch.stack(preds).view(-1)
            true_values = torch.stack(true_values).view(-1).detach().cpu().numpy()
            f1_train = f1_score(true_values, (preds > CFG.THRESHOLD).int().detach().cpu().numpy(), average="macro")
            f1_trains.append(f1_train)

            #"""
            #validation
            #"""
            model.eval()
            loss_val = 0
            preds, true_values = [], []
            for i, (x, x_mask, t) in enumerate(val_loader):
                x, x_mask, t = x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x, x_mask)
                preds += y
                true_values += t

                weight = (lambda w: torch.where(w==0, 1, w))(t*weight_ratio)
                try:
                    loss = torch.mean(loss_func(y, t)*weight) / num_outputs
                except Exception as e:
                    print(y)
                    return (x, x_mask, y,t,weight, loss_func, model), None
                loss_val += loss.item()

            loss_val /= i
            loss_vals.append(loss_val)
            preds = torch.stack(preds).view(-1)
            true_values = torch.stack(true_values).view(-1).detach().cpu().numpy()
            f1_val = f1_score(true_values, (preds > CFG.THRESHOLD).int().detach().cpu().numpy(), average="macro")
            f1_vals.append(f1_val)

            print(f"Epoch {epoch+1}/{CFG.EPOCHS}: loss_train {loss_train:.4f}, loss_val {loss_val:.4f}, f1_train {f1_train:.4f} f1_val - {f1_val:.4f}")

            #"""
            #early stopping
            #"""
            if epoch+1 < CFG.WARM_UP: # warm up: do not stop early
                best_score_before = None
                continue
            elif epoch+1 == CFG.WARM_UP:
                best_score_before = min(loss_vals[CFG.WARM_UP-1:])
            else:
                best_score_before = min(loss_vals[CFG.WARM_UP-1:-1])

            if (min(loss_vals[CFG.WARM_UP-1:]) == loss_val) and (best_score_before != loss_val):
                early_stopping_count = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                early_stopping_count += 1

            if early_stopping_count == CFG.EARLY_STOPPING_ROUNDS:
                best_epoch = loss_vals.index(min(loss_vals[CFG.WARM_UP-1:]))
                early_stopping_message =\
                f"Best Epoch: {best_epoch+1}"\
                + f", TrainLoss: {loss_trains[best_epoch]:.4f}" + f", ValLoss: {loss_vals[best_epoch]:.4f}"\
                + f", TrainF1: {f1_trains[best_epoch]:.4f}" + f", ValF1: {f1_vals[best_epoch]:.4f}"
                print("\n!!!Early Stopping !!!")
                print(early_stopping_message)
                try:
                    model.load_state_dict(torch.load(checkpoint_path))
                except Exception as e:
                    print(e)
                break
            else:
                best_epoch = epoch + 1

        torch.save(model.state_dict(), checkpoint_path)

        #"""
        #checking train result
        #"""
        result = pd.DataFrame(
            {
                "value": loss_trains + loss_vals + f1_trains + f1_vals,
                "metric": ["loss"] * (len(loss_trains)+len(loss_vals)) + ["f1_score"] * (len(loss_trains)+len(loss_vals)),
                "epoch": ([i+1 for i in range(len(loss_trains))] + [i+1 for i in range(len(loss_vals))])*2,
                "train/val": (["train" for _ in range(len(loss_trains))] + ["val" for _ in range(len(loss_vals))])*2
            }
        )
        px.line(result.query("metric=='loss'"), x="epoch", y="value",
                color="train/val", height=250, title="loss").show()
        px.line(result.query("metric=='f1_score'"), x="epoch", y="value",
                color="train/val", height=250, title="f1_score").show()

        #"""
        #exploring best threshold
        #"""
        preds, true_values = [], []
        for i, (x, x_mask, t) in enumerate(val_loader):
            x, x_mask, t = x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE), t.to(CFG.DEVICE)
            y = model(x, x_mask).view(-1)
            preds += y
            true_values += t

        preds = torch.stack(preds).view(-1).detach().cpu().numpy()
        true_values = torch.stack(true_values).view(-1).detach().cpu().numpy()
        px.box(x=true_values, y=preds, points="all").show()

        f1s = []
        for th in range(0, 101, 1):
            f1 = f1_score(true_values, (preds > (th/100)).astype(int), average="macro")
            f1s.append(f1)
        f1s = pd.DataFrame({"threshold":[i/100 for i in range(0, 101, 1)], "f1":f1s}).set_index("threshold")
        best_threshold = f1s[f1s.f1.apply(lambda x: x//.01)==f1s.f1.apply(lambda x: x//.01).max()].index.max()
        px.line(f1s, title=f"Best Threshold is {best_threshold}.").show()

        #"""
        #fine tuning with validation data
        #"""
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE*.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20*.5, eta_min=.005)

        loss_trains, f1_trains = [], []
        loss_func = nn.BCELoss(reduction='none')
        model.train()

        epochs = max(int(best_epoch*.1), 10)
        for epoch in range(epochs):
            loss_train = 0
            preds, true_values = [], []
            for i, (x, x_mask, t) in enumerate(val_loader):
                x, x_mask, t = x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x, x_mask)
                preds += y
                true_values += t

                weight = (lambda w: torch.where(w==0, 1, w))(t*weight_ratio)
                loss = torch.mean(loss_func(y, t)*weight) / num_outputs
                loss_train += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            loss_train /= i
            loss_trains.append(loss_train)
            preds = torch.stack(preds).view(-1)
            true_values = torch.stack(true_values).view(-1).detach().cpu().numpy()
            f1_train = f1_score(true_values, (preds > best_threshold).int().detach().cpu().numpy(), average="macro")
            f1_trains.append(f1_train)

            print(f"Epoch {epoch+1}: loss_train {loss_train:.4f}, f1_train {f1_train:.4f}")

            if epoch == 0:
                best_score_before = loss_train
                if f1_train != 0:
                    torch.save(model.state_dict(), checkpoint_path)                  
            else:
                best_score_before = min(loss_trains)

            if (min(loss_trains) == loss_train) and (best_score_before != loss_train):
                if f1_train != 0:
                    torch.save(model.state_dict(), checkpoint_path)                  

        model.load_state_dict(torch.load(checkpoint_path))

        #"""
        #checking train result
        #"""
        result = pd.DataFrame(
            {
                "value": loss_trains + f1_trains,
                "metric": ["loss"] * len(loss_trains) + ["f1_score"] * len(loss_trains),
                "epoch": [i+1 for i in range(len(loss_trains))]*2,
            }
        )
        px.line(result.query("metric=='loss'"), x="epoch", y="value",
                height=250, title="FineTuning: loss").show()
        px.line(result.query("metric=='f1_score'"), x="epoch", y="value",
                height=250, title="FineTuning: f1_score").show()

    return model, best_threshold


# ### Predicting Function

def predict(test, model, level_group=None, threshold=None, question=None):
    
    if CFG.SUBMISSION_MODE:
        test_idx = test[0].index.unique()
        if question:
            cols = [question]
        else:
            if level_group and model.num_outputs != 18:
                cols = lq_dict[level_group]
            else:
                cols = questions
    elif level_group:
        #test = test.query(f"level_group=='{level_group}'")
        test_idx = [i[0] for i in test[0].index.unique()]
        if threshold is None:
            threshold = CFG.THRESHOLD
        
        if question:
            cols = [question]
        else:
            if model.num_outputs == 18:
                cols = questions
            else:
                cols = lq_dict[level_group]
    else:
        test_idx = test[0].index.unique()
        cols = questions
        if threshold is None:
            threshold = CFG.THRESHOLD
    
    test_loader = create_loader_test(test[0], test[1], level_group)
    preds = []
    model.eval()

    for x, x_mask in test_loader:
        y = model(x.to(CFG.DEVICE), x_mask.to(CFG.DEVICE))
        preds += y
    preds = torch.stack(preds)
    
    submission = pd.DataFrame(
        (preds > threshold).int().detach().cpu().numpy(),
        columns = cols,
    )
    
    submission = submission[cols]
    submission["session_id"] = test_idx
    submission["session_id"] = submission["session_id"].astype(str)
    submission = pd.melt(submission, id_vars="session_id", var_name="question", value_name="correct")
    submission["session_id"] += "_" + submission.question
    submission = submission[["session_id", "correct"]]
    
    del preds, test_idx
    gc.collect()
    
    return submission


# ### LightGBM for feature elimination

def lgb_training(X, y, params=None, train_rate=.9):
    
    lgb_models = {}
    lgb_preds = {}
    lgb_scores = {}
    fi_df = []
    
    if params is None:
        params = dict(
            objective='binary',
            metric='binary_logloss',
            verbosity=1,
            early_stopping_round=100,
            random_state=CFG.SEED,
            is_unbalance=True,
            num_iterations=2000,
            num_leaves=500,
            lambda_l1=.2,
            lambda_l2=.2,
            bagging_freq=10,
            bagging_seed=CFG.SEED,
            force_col_wise=True
        )
    
    for level_group in lq_dict.keys():
        for q in lq_dict[level_group]:
            X_train, X_val, y_train, y_val = tts(
                X[level_group][0],
                y[q],
                train_size=train_rate,
                random_state=CFG.SEED,
                stratify=y[q]
            )
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val)

            del X_train, y_train
            gc.collect()

            lgb_model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=1000,
                valid_sets=(lgb_train, lgb_val),
                callbacks=None,
                verbose_eval=100
            )

            lgb_models[q] = lgb_model
            lgb_preds[q] = lgb_model.predict(X_val)
            lgb_scores[q] = f1_score(y_val,
                                     [int(p > CFG.THRESHOLD) for p in lgb_preds[q]],
                                     average="macro")

            fi_df.append(
                pd.DataFrame(
                    lgb_model.feature_importance(importance_type="gain"),
                    index=X[level_group][0].columns,
                    columns=[q]
                )
            )

    fi_df = pd.concat(fi_df, axis=1)
    
    feature_elimination_dict = {}
    for level_group in lq_dict.keys():
        fe_cols = fi_df[lq_dict[level_group]].mean(axis=1).sort_values(ascending=False)
        fe_cols = fe_cols.iloc[:30].index.tolist()
        feature_elimination_dict[level_group] = fe_cols
    
    return lgb_models, lgb_preds, lgb_scores, fi_df, feature_elimination_dict


# # Execution

# ## Preparing Objects

# +
# %%time

if CFG.SUBMISSION_MODE:
    with open(f"{CFG.CHECKPOINT_PATH}/dp.pickle", "rb") as f:
        dp = pickle.load(f)

    models = {}    
    for level_group in lq_dict.keys():
        num_features = dp.num_features
        pretrained_model = PretrainingSAINT(num_features).to(CFG.DEVICE)
        model = MultiOutputsModel(num_features, 18 if CFG.PREDICT_ALL else len(lq_dict[level_group]))
        model.register_pretrained_model(pretrained_model)
        checkpoint_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_bestmodel{'_'+level_group if level_group else ''}{'_all' if CFG.PREDICT_ALL else ''}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        models[level_group] = model
        
    with open(f"{CFG.CHECKPOINT_PATH}/{model.name}_thresholds{'_all' if CFG.PREDICT_ALL else ''}.pickle", "rb") as f:
        thresholds = pickle.load(f)
        
else:
    if CFG.RESUME:
        with open(f"{CFG.CHECKPOINT_PATH}/dp.pickle", "rb") as f:
            dp = pickle.load(f)
        
        with open(f"{CFG.CHECKPOINT_PATH}/train.pickle", "rb") as f:
            train = pickle.load(f)
        
        test = dp.transform(f"{CFG.INPUT}/test.csv")
        
        with open(f"{CFG.CHECKPOINT_PATH}/cooccurence_rate.pickle", "rb") as f:
            cooccurence_rate = pickle.load(f)
        
        train_labels = pd.read_csv(f"{CFG.INPUT}/train_labels.csv", engine='python')
        train_labels[["session_id", "question"]] = train_labels.session_id.str.split("_", expand=True)
        train_labels = train_labels[["session_id", "question", "correct"]]
        train_labels["session_id"] = train_labels["session_id"].astype(int)
        train_labels = train_labels.pivot(index="session_id", columns="question", values="correct").sort_index()
        train_labels = train_labels.reindex(columns=questions)
    else:
        dp = DataProcessing()

        train = dp.fit_transform(f"{CFG.INPUT}/train.csv")
        test = dp.transform(f"{CFG.INPUT}/test.csv")

        if CFG.CHECKPOINT:
            with open(f"{CFG.CHECKPOINT_PATH}/dp.pickle", "wb") as f:
                pickle.dump(dp, f)
            with open(f"{CFG.CHECKPOINT_PATH}/train.pickle", "wb") as f:
                pickle.dump(train, f)

        del dp
        gc.collect()

        train_labels = pd.read_csv(f"{CFG.INPUT}/train_labels.csv", engine='python')
        train_labels[["session_id", "question"]] = train_labels.session_id.str.split("_", expand=True)
        train_labels = train_labels[["session_id", "question", "correct"]]
        train_labels["session_id"] = train_labels["session_id"].astype(int)
        train_labels = train_labels.pivot(index="session_id", columns="question", values="correct").sort_index()
        train_labels = train_labels.reindex(columns=questions)

        check = train_labels.copy()
        res = {}
        for session_id, row in check[questions].iterrows():
            tmp = []
            for i1, q1 in enumerate(row):
                for i2, q2 in enumerate(row):
                    tmp.append((questions[i1], questions[i2], int(q1==q2)))
            tmp = pd.DataFrame(tmp)
            tmp = pd.pivot(data=tmp, index=0, columns=1, values=2)
            res[session_id] = np.array(tmp.reindex(index=questions, columns=questions))

        cooccurence_rate = np.mean(list(res.values()), axis=0)
        cooccurence_rate = pd.DataFrame(data=cooccurence_rate, columns=questions)
        cooccurence_rate.index = questions

        if CFG.CHECKPOINT:
            with open(f"{CFG.CHECKPOINT_PATH}/cooccurence_rate.pickle", "wb") as f:
                pickle.dump(cooccurence_rate, f)
# -

# ## Deep Learning

# +
# %%time

if CFG.SUBMISSION_MODE:    
    env = jo_wilder_310.make_env()
    iter_test = env.iter_test()

    storage = []
    for x, sample in iter_test:
        level_group = x.level_group.values[0]
        if level_group == "0-4":
            storage.append(x)
            x = dp.transform(data=x, decided_level_group=level_group)
        elif level_group == "5-12":
            storage.append(x)
            x = pd.concat(storage, axis=0, ignore_index=True)
            x.level_group = level_group
            x = dp.transform(data=x, decided_level_group=level_group)
        else: # "13-22"
            x = pd.concat(storage + [x], axis=0, ignore_index=True)
            x.level_group = level_group
            x = dp.transform(data=x, decided_level_group=level_group)
            storage = []
        
        submission = predict(x, models[level_group], level_group, thresholds[level_group])
        submission = submission.set_index("session_id").reindex(sample.session_id).reset_index()
        env.predict(submission)                       

else:
    thresholds = {}
    for level_group in lq_dict.keys():
        print(f"\nlevel_group {level_group} model...")
        
        # create data loader
        train_loader, val_loader = create_loader_train(
                            train[level_group][0],
                            train[level_group][1],
                            train_labels,
                            level_group,
                            predict_all=CFG.PREDICT_ALL
                        )
        # pretraining
        pretrained_model = pretraining(
            train_loader,
            level_group,
            epochs=CFG.PRETRAIN_EPOCHS,
            omit=CFG.PRETRAINING_OMIT
        )
            
        # training
        model, best_threshold = training(
            train_loader,
            val_loader,
            MultiOutputsModel,
            pretrained_model,
            omit=True#CFG.TRAINING_OMIT
        )
        
        best_threshold = explore_threshold(model, val_loader)
            
        # predicting
        submission = predict(test[level_group], model, level_group, best_threshold)
        display(submission)
        submission.to_csv("submission.csv", index=False)
        
        thresholds[level_group] = best_threshold
        
        torch.cuda.empty_cache()
    
    thresholds_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_thresholds{'_all' if CFG.PREDICT_ALL else ''}.pickle"
    with open(thresholds_path, "wb") as f:
        pickle.dump(thresholds, f)
# -


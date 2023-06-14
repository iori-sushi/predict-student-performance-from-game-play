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
from torch.utils.data.dataset import Dataset
import lightgbm as lgb
import plotly.express as px
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.simplefilter("ignore")
pd.options.display.max_columns=1000
torch.autograd.set_detect_anomaly(True)

class CFG:
    # hyper parameters
    EPOCHS = 200
    PRETRAIN_EPOCHS = 11
    THRESHOLD = .5
    PRETRAIN_BATCH_SIZE = 2**10
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
    PRETRAIN_OMIT = False
    TRAINING_OMIT = False
    if SUBMISSION_MODE:
        CHECKPOINT = False
        RESUME = False
        PRETRAIN_OMIT = True
        TRAINING_OMIT = True
        VERBOSE = False
    elif CHECKPOINT:
        RESUME = False
        PRETRAIN_OMIT = False
        TRAINING_OMIT = False
    else:
        RESUME = True
        if TRAINING_OMIT:
            PRETRAIN_OMIT = True
            
    INPUT = '../input'
    CHECKPOINT_PATH = '../checkpoint/PretrainFromDynamicData'
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

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
# -

# ## Functions & Classes

# ### Data Processing Class

class DataProcessing():
    def __init__(
        self,
        level_group:str="0-4",
        stride:int=1,
        filter_size:int=10
    ):
        self.level_group=level_group
        self.stride=stride
        self.filter_size=filter_size
    
    def get_labels(self, path=f"{CFG.INPUT}/train_labels.csv"):
        train_labels = pd.read_csv(path, engine='python')
        train_labels[["session_id", "question"]] = train_labels.session_id.str.split("_", expand=True)
        train_labels = train_labels[["session_id", "question", "correct"]]
        train_labels["session_id"] = train_labels["session_id"].astype(int)
        train_labels = train_labels.pivot(index="session_id", columns="question", values="correct").sort_index()
        train_labels = train_labels.reindex(columns=questions)
        return train_labels
    
    def fit_transform(self, path:str=None, df:pd.DataFrame=None):
        return self.transform(path=path, df=df, fit=True)
    
    def transform(self, path:str=None, df:pd.DataFrame=None, fit=False):
        if path:
            self.df = pd.read_csv(path)
            self.df = self.df.query(f"level_group=='{self.level_group}'").reset_index(drop=True)
        if df is not None:
            self.df = df.copy()
            self.df = self.df.query(f"level_group=='{self.level_group}'").reset_index(drop=True)
        df = self.df.copy()
        df = df.drop(["index", "page", "text", "level_group"], axis=1)
        
        df = df.sort_values(["session_id", "elapsed_time"]).reset_index(drop=True)
        df.elapsed_time = [t if t>=0 else 0 for t in df.elapsed_time.diff()]
        df.elapsed_time = df.elapsed_time.shift(-1).fillna(0)
        
        df["distance_room"] = df.room_coor_x.diff()**2 + df.room_coor_y.diff()**2
        df["distance_room"][df.session_id.diff()!=0] = 0
        df["distance_room"] = df["distance_room"].shift(-1).fillna(0)
        df["distance_room"] = df["distance_room"].apply(lambda x: np.sqrt(x))
        df["distance_screen"] = df.screen_coor_x.diff()**2 + df.screen_coor_y.diff()**2
        df["distance_screen"][df.session_id.diff()!=0] = 0
        df["distance_screen"] = df["distance_screen"].shift(-1).fillna(0)
        df["distance_screen"] = df["distance_screen"].apply(lambda x: np.sqrt(x))
        
        df.hover_duration = df.hover_duration.fillna(0)
        df["av"] = df.fullscreen.astype(str)+df.hq.astype(str)+df.music.astype(str)
        df = df.drop([
            "room_coor_x",
            "room_coor_y",
            "screen_coor_x",
            "screen_coor_y",
            "fullscreen",
            "hq",
            "music"
        ], axis=1)
        
        self.continuous_features= [
            'elapsed_time',
            'hover_duration',
            'distance_room',
            'distance_screen'
        ]
        df = self.standardize_continuous(df, self.continuous_features, fit)

        self.categorical_features=[
            'event_name', # first one is the pretraining label
            'name',
            'level',
            'fqid',
            'room_fqid',
            'text_fqid',
            'av'            
        ]
        df = self.label_encoding(df, self.categorical_features, fit)
            
        type_dict = dict(
            session_id='int64',
            elapsed_time='float64',
            event_name='int8',
            name='int8',
            level='int8',
            hover_duration='float64',
            fqid='int16',
            room_fqid='int16',
            text_fqid='int16',
            av='int8',
            distance_room='float64',
            distance_screen='float64'
        )
        df = df[[col for col in type_dict.keys()]].astype(type_dict)
        df = self.make_columns_same(df, fit)
        
        assert df.isnull().sum().sum()==0, ("Data includes NULL.")
        
        x_categorical, x_continuous = self.tensor_processing(df, fit)
        
        session_change_index = df.session_id.diff()!=0
        session_change_index = session_change_index[session_change_index].index.to_list()
        session_change_index += [len(df)]
        translate_table = {k:df.loc[k, "session_id"] for k in session_change_index[:-1]}

        del df, self.df
        gc.collect()
        
        x_categorical, x_continuous, session_index = self.stride_data(
            x_categorical,
            x_continuous,
            session_change_index,
            translate_table
        )
        
        del session_change_index, translate_table
        gc.collect()
        
        assert len(session_index)==x_continuous.shape[0]==x_categorical.shape[0],\
        ("Length Differ Between Outputs.")
        
        session_index = [int(i) for i in session_index]
        
        return x_categorical, x_continuous, session_index
    
    def standardize_continuous(self, df:pd.DataFrame, columns:list, fit:bool=False):
        if fit:
            ss = StandardScaler()
            df[columns] = ss.fit_transform(df[columns])
            self.scaler = ss
        else:
            ss = self.scaler
            df[columns] = ss.transform(df[columns])
        return df
   
    def label_encoding(self, df:pd.DataFrame, columns:list, fit:bool=False):
        if fit:
            self.encoders = {}
            self.all_pad_tensor = []
            for col in columns:
                if col != "level":
                    df[col] = df[col].fillna("MissingValue")
                    le = LabelEncoder()
                    le.fit(df[col].unique().tolist()+["PAD"])
                    df[col] = le.transform(df[col])
                    self.encoders[col] = le
                    self.all_pad_tensor += [le.classes_.tolist().index("PAD")]
                else:
                    self.all_pad_tensor += [23]
            self.all_pad_tensor = torch.Tensor(self.all_pad_tensor).int()
        else:
            for col in columns:
                if col != "level":
                    df[col] = df[col].fillna("MissingValue")
                    le = self.encoders[col]
                    df[col] = [v if v in le.classes_ else "MissingValue" for v in df[col]]
                    df[col] = le.transform(df[col])
        return df
    
    def make_columns_same(self, df:pd.DataFrame, fit:bool=False):
        if fit:
            self.columns=df.columns
        else:
            df = df[self.columns]
        return df
    
    def tensor_processing(
        self,
        df:pd.DataFrame,
        fit:bool=False,
        num_special_tokens:int=0,
        pretrain_label_continuous:str="elapsed_time"
    ):
        if fit:
            self.num_special_tokens = num_special_tokens
            self.num_continuous = len(self.continuous_features)
            self.num_categorical = len(self.categorical_features)
            self.categories = [df[col].nunique() for col in self.categorical_features]
            self.num_features = self.num_continuous + self.num_categorical
            self.categories_offset = torch.Tensor(
                [self.num_special_tokens]+self.categories
            ).int().cumsum(dim=-1)
            self.num_tokens = self.categories_offset[-1]
            self.num_total_tokens = self.num_tokens + self.num_special_tokens
        
        x_categorical = torch.Tensor(df[self.categorical_features].values)
        x_categorical += self.categories_offset[:-1] # avoid duplication between columns
        x_categorical = x_categorical.int()
        
        x_continuous = torch.Tensor(df[self.continuous_features].values)
        
        return x_categorical, x_continuous
    
    def stride_data(
        self,
        categorical:torch.Tensor,
        continuous:torch.Tensor,
        change_index:list,
        translate_table:dict
    ):
        categorical_data = []
        continuous_data = []
        session_index = []
        all_pad_continuous = torch.zeros(continuous.shape[1])
        stride = self.stride
        filter_size = self.filter_size
        
        for i in range(len(change_index)-1):
            begining = change_index[i]
            ending = change_index[i+1]
            session = translate_table[begining]
            
            # head    
            categorical_data.append(
                torch.concat(
                    [
                        self.all_pad_tensor.repeat(stride).view(stride, -1),
                        categorical[begining:begining+filter_size-stride]
                    ], axis=0
                )
            )
            continuous_data.append(
                torch.concat(
                    [
                        all_pad_continuous.repeat(stride).view(stride, -1),
                        continuous[begining:begining+filter_size-stride]
                    ], axis=0
                )
            )
            session_index += [session]
            
            # intermediate
            for i2 in range(begining, ending-filter_size+1, stride):
                categorical_data.append(categorical[i2:i2+filter_size])
                continuous_data.append(continuous[i2:i2+filter_size])
                session_index.append(session)
                
            # tail
            continuous_data.append(
                torch.concat(
                    [
                        continuous[ending-filter_size+stride:ending],
                        all_pad_continuous.repeat(stride).view(stride, -1)
                    ], axis=0
                )
            )
            categorical_data.append(
                torch.concat(
                    [
                        categorical[ending-filter_size+stride:ending],
                        self.all_pad_tensor.repeat(stride).view(stride, -1)
                    ], axis=0
                )
            )
            session_index += [session]
            
        categorical = torch.concat(categorical_data, dim=0).view(-1, filter_size, self.num_categorical)
        continuous = torch.concat(continuous_data, dim=0).view(-1, filter_size, self.num_continuous)
            
        return categorical, continuous, session_index


# ### Dataloader Creating Functions

# +
def create_loader_pretrain(
    x_categorical:torch.Tensor,
    x_continuous:torch.Tensor,
    y:pd.DataFrame,
    session_index:list,
    level_group="0-4",
    train_rate:float=.9,
    predict_all:float=False
):
    y = y[lq_dict[level_group]]
    
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1-train_rate,
        random_state=CFG.SEED
    )
    res = msss.split(X=y.values, y=y.values)
    for train_index, val_index in res:
        y_train = y.iloc[train_index.tolist(),:]
        y_val = y.iloc[val_index.tolist(),:]
        y_train_sessions = y_train.index
        y_val_sessions = y_val.index
        
    x_train_sessions=[i if i in y_train_sessions else np.nan for i in np.array(session_index)]
    x_val_sessions=[i if i in y_val_sessions else np.nan for i in np.array(session_index)]
    
    train_index = ~np.isnan(np.array(x_train_sessions))
    val_index = ~np.isnan(np.array(x_val_sessions))
    
    x_train_loader = DataLoader(
        TensorDataset(
            x_categorical[train_index],
            x_continuous[train_index],
        ),
        batch_size=CFG.PRETRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    x_val_loader = DataLoader(
        TensorDataset(
            x_categorical[val_index],
            x_continuous[val_index],
        ),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    del train_index, val_index
    gc.collect()

    x_train_sessions=[i for i in x_train_sessions if not np.isnan(i)]
    x_val_sessions=[i for i in x_val_sessions if not np.isnan(i)]

    return x_train_loader, x_train_sessions, y_train, x_val_loader, x_val_sessions, y_val

def create_loader_train(
    x_dict_train:dict,
    y_train:pd.DataFrame,
    x_dict_val:dict,
    y_val:pd.DataFrame,
):
    x_train = [x_dict_train[i] for i in y_train.index]
    x_train = torch.stack(x_train)
    
    x_val = [x_dict_val[i] for i in y_val.index]
    x_val = torch.stack(x_val)
    
    train_loader = DataLoader(
        TensorDataset(x_train, torch.Tensor(y_train.values)),
        batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4,
    )

    val_loader = DataLoader(
        TensorDataset(x_val, torch.Tensor(y_val.values)),
        batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4,
    )

    return train_loader, val_loader

def create_loader_test(
    x_categorical:torch.Tensor,
    x_continuous:torch.Tensor,
    level_group=None
):
    if CFG.SUBMISSION_MODE:
        pass

    test_loader = DataLoader(
        x_categorical, x_continuous,
        batch_size=CFG.BATCH_SIZE, shuffle=False
    )
    
    return test_loader


# -

# ### PretrainModel

# #### Augmentation Functions

# +
def cutmix(x:torch.Tensor, m:float=.1):
    x = x.clone()
    batch_size, filter_size, *others = x.shape
    x_shuffle = x[torch.randperm(batch_size)]
    random_choice = torch.from_numpy(
        np.random.choice(
            2,
            ([batch_size, filter_size]+others),
            p=[m,1-m]
        )
    )
    x[random_choice==0] = x_shuffle[random_choice==0]
    return x

def mixup(x:torch.Tensor, alpha:float=.2):
    x = x.clone()
    index = torch.randperm(x.shape[0])
    x = alpha*x + (1-alpha)*x[index, :]
    return x


# -

# #### Pretraining Orchestrator

class PretrainModel(nn.Module):
    def __init__(
        self,
        num_total_tokens:int,
        num_continuous:int,
        categories:list,
        filter_size:int=10,
        num_features:int=11,
        dim:int=32,
        num_outputs_projection_head:int=128,
        heads:int=8,
        dim_head:int=16,
        attention_dropout:float=.05,
    ):
        super(PretrainModel, self).__init__()
        self.encoder = Encoder(
            num_total_tokens=num_total_tokens,
            num_continuous=num_continuous,
            filter_size=filter_size,
            num_features=num_features,
            dim=dim,
            num_outputs=num_outputs_projection_head,
            heads=heads,
            dim_head=dim_head,
            attention_dropout=attention_dropout,
        )
        
        self.categories = categories
        self.filter_size = filter_size
        self.num_outputs_projection_head = num_outputs_projection_head
        self.predictor_sim = nn.Sequential(
            nn.Linear(num_outputs_projection_head, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs_projection_head)
        )
        self.predictor_fve_cat = nn.Sequential(
            nn.Linear(num_features*dim, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
            nn.Sigmoid()
        )
        self.future_records_prediction = FutureRecordsPrediction(
            encoder=self.encoder,
            filter_size=filter_size,
            num_features=num_features,
            dim=dim
        )

    @torch.no_grad()
    def _make_label(self, x_categorical, category_index:int=0):
        num_classes=self.categories[category_index]+1
        mini = torch.min(x_categorical[:,:,category_index])
        label_categorical = F.one_hot(
            x_categorical[:,:,category_index].to(torch.int64)-mini,
            num_classes=num_classes
        ).to(torch.float32)
        label_categorical = label_categorical.view(-1, num_classes)
        x_categorical[:,:,category_index] = 0
        x_fve = x_categorical
        return label_categorical, x_fve
    
    def forward(
        self,
        x_categorical:torch.Tensor,
        x_continuous:torch.Tensor,
        real_space_aug_func=cutmix,
        latent_space_aug_func=mixup
    ):
        ## contrastive learning
        x = self.encoder(x_categorical.clone(), x_continuous.clone())
        x = F.normalize(x, dim=1)
        xp = self.predictor_sim(x)
        
        x_ = self.encoder(
            x_categorical.clone(),
            x_continuous.clone(),
            real_space_aug_func=real_space_aug_func,
            latent_space_aug_func=latent_space_aug_func
        )
        x_ = F.normalize(x_, dim=1)
        xp_ = self.predictor_sim(x_)
        
        loss = -F.cosine_similarity(x.detach(), xp_)
        loss_ =  -F.cosine_similarity(x_.detach(), xp)        
        loss = torch.mean((loss + loss_) / 2.0) + 1 # make loss a positive number
        
        ## feature vector estimation: fve
        label_categorical, x_fve = self._make_label(x_categorical.clone())
        
        # estimation categorical feature
        x_fve = self.encoder.embedding(x_fve, x_continuous.clone())
        x_fve = torch.concat(x_fve, dim=2)
        x_fve = self.encoder.column_attention(x_fve)
        x_fve = self.encoder.time_series_attention(x_fve)
        x_fve = x_fve.flatten(2)
        x_fve = x_fve.view(-1, x_fve.shape[-1])
        x_fve = self.predictor_fve_cat(F.normalize(x_fve.flatten(1), dim=1))
        loss_fve_cat = F.cross_entropy(x_fve, label_categorical, reduction="mean")
        
        ## future records estimation
        x_frp = self.encoder.embedding(x_categorical.clone(), x_continuous.clone())
        x_frp = torch.concat(x_frp, dim=2)
        x_frp = self.encoder.column_attention(x_frp)
        x_frp = self.encoder.time_series_attention(x_frp)
        loss_frp = self.future_records_prediction(x_frp)
                    
        return loss, loss_fve_cat, loss_frp


# #### Future Records Prediction

class FutureRecordsPrediction(nn.Module):
    def __init__(
        self,
        encoder:nn.Module,
        filter_size:int,
        num_features:int,
        dim:int
    ):
        super(FutureRecordsPrediction, self).__init__()
        self.first_half = nn.Sequential(
            nn.Linear(int(filter_size/2*num_features*dim), 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.second_half = nn.Sequential(
            nn.Linear(int(filter_size/2*num_features*dim), 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_frp:torch.Tensor):
        first_half, second_half = x_frp.chunk(2, dim=1)
        first_half, second_half = first_half.flatten(1), second_half.flatten(1)

        first_half_mlp = self.first_half(first_half.clone())
        first_half_mlp_ = self.first_half(second_half.clone())
        second_half_mlp = self.second_half(second_half.clone())
        second_half_mlp_ = self.second_half(first_half.clone())
        
        pred = self.classifier(torch.concat([first_half_mlp, second_half_mlp], dim=1))
        loss = F.binary_cross_entropy(pred, torch.ones(pred.shape).to(CFG.DEVICE))
        pred_ = self.classifier(torch.concat([first_half_mlp_, second_half_mlp_], dim=1))
        loss_ = F.binary_cross_entropy(pred_, torch.zeros(pred_.shape).to(CFG.DEVICE))
        
        return (loss + loss_) / 2.0


# #### Encoder

class Encoder(nn.Module):
    def __init__(
        self,
        num_total_tokens:int,
        num_continuous:int,
        filter_size:int=10,
        num_features:int=11,
        dim:int=32,
        num_outputs:int=128,
        heads:int=8,
        dim_head:int=16,
        attention_dropout:float=.1,
    ):
        super(Encoder, self).__init__()
        self.num_total_tokens = num_total_tokens
        self.num_continuous = num_continuous
        self.dim = dim
        
        self.embedding = Embedding(num_total_tokens, num_continuous, filter_size, dim)
        self.column_attention = Attention(dim, heads, dim_head, attention_dropout)
        self.time_series_attention = Attention(dim, heads, dim_head, attention_dropout)
        self.projection_head = ProjectionHead(filter_size, num_features, dim, num_outputs)
        
    def forward(
        self,
        x_categorical:torch.Tensor,
        x_continuous:torch.Tensor,
        real_space_aug_func=None, # cutmix?
        latent_space_aug_func=None, # mixup?
    ):        
        if real_space_aug_func:
            x_categorical = real_space_aug_func(x_categorical)
            x_continuous = real_space_aug_func(x_continuous)
        
        x = torch.concat(self.embedding(x_categorical, x_continuous), dim=2)

        if latent_space_aug_func:
            x = latent_space_aug_func(x)
    
        x = self.column_attention(x)
        x = torch.permute(x, (0,2,1,3))
        x = self.time_series_attention(x)
        x = torch.permute(x, (0,2,1,3))
        x = self.projection_head(x.flatten(1))
        
        return x


# #### Embedding

# +
class Embedding(nn.Module):
    def __init__(
        self,
        num_total_tokens:int,
        num_continuous:int,
        filter_size:int=10,
        dim:int=32
    ):
        super(Embedding, self).__init__()                
        self.dim = dim
        self.embedding_categorical = EmbeddingCategorical(num_total_tokens, dim)
        self.embedding_continuous = EmbeddingContinuous(num_continuous, filter_size, dim)
        
    def forward(self, x_categorical, x_continuous):
        x_categorical = self.embedding_categorical(x_categorical)
        x_continuous = self.embedding_continuous(x_continuous) # batch_size, filter_size, num_features, dim 
        return x_categorical, x_continuous
        
class EmbeddingCategorical(nn.Module):
    def __init__(
        self,
        num_total_tokens:int,
        dim:int=32,
    ):
        super(EmbeddingCategorical, self).__init__()
        self.dim=dim
        self.num_total_tokens=num_total_tokens
        self.embedding=nn.Embedding(self.num_total_tokens, self.dim)
        
    def forward(self, x):
        x = self.embedding(x)
        return x
        
class EmbeddingContinuous(nn.Module):
    def __init__(
        self,
        num_continuous:int,
        filter_size:int=10,
        dim:int=32
    ):
        super(EmbeddingContinuous, self).__init__()
        self.num_continuous=num_continuous
        self.dim=dim
        self.filter_size=filter_size
        self.projections = nn.ModuleList(
            [
                ProjectingContinuous(dim=dim)
                for _ in range(self.num_continuous*self.filter_size)
            ]
        )
        
    def forward(self, x):
        x_enc = torch.empty(*x.shape, self.dim).to(CFG.DEVICE) # batch_size, filter_size, num_features, dim 
        for s in range(self.filter_size):
            for i in range(self.num_continuous):
                x_enc[:,s,i,:] = self.projections[s*self.num_continuous+i](x[:,s,i])
        return x_enc
        
class ProjectingContinuous(nn.Module):
    def __init__(
        self,
        intermediate_unit=100,
        dim:int=32,
    ):
        super(ProjectingContinuous, self).__init__()
        self.linear1 = nn.Linear(1, intermediate_unit)
        self.linear2 = nn.Linear(intermediate_unit, dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# -

# #### Attention

class Attention(nn.Module):
    def __init__(
        self,
        dim:int=32,
        heads:int=8,
        dim_head:int=16,
        dropout:float=.1
    ):
        super(Attention, self).__init__()
        self.dim=dim
        self.heads=heads
        self.dim_head=dim_head
        self.key = nn.Linear(dim, dim_head*heads, bias=False)
        self.query = nn.Linear(dim, dim_head*heads, bias=False)
        self.value = nn.Linear(dim, dim_head*heads, bias=False)
        self.out = nn.Linear(dim_head*heads, dim)
        self.dropout = dropout
        
    def forward(self, x):
        # x.shape -> batch_size, filter_size, num_features, dim=32
        batch_size, filter_size, num_features, dim = x.shape
        
        # k.shape -> batch_size, filter_size, num_features, dim_head*heads=128
        key, query, value = self.key(x), self.query(x), self.value(x)

        # k.shape -> batch_size, heads, filter_size, num_features, dim_head
        reshaping = lambda t:torch.permute(t, (0,1,3,2)).reshape(
            batch_size, self.heads, filter_size, num_features, self.dim_head
        )
        key, query, value = map(reshaping, [key, query, value])
        
        attention = torch.einsum('bhsid, bhsjd -> bhsij', (query, key))
        attention /= self.dim_head**0.5
        attention = attention.softmax(dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = torch.einsum('bhsij, bhsjd -> bhsid', (attention, value))

        # attention.shape -> batch_size, filter_size, num_features, dim_head*heads=128
        attention = attention.view(batch_size, filter_size, num_features, self.dim_head*self.heads)

        # attention.shape -> batch_size, filter_size, num_features, dim=32
        attention = self.out(attention)

        return attention


# #### Projection Head

class ProjectionHead(nn.Module):
    def __init__(
        self,
        filter_size:int,
        num_features:int,
        dim:int,
        num_outputs:int=128
    ):
        super(ProjectionHead, self).__init__()
        units = [
            filter_size*num_features*dim,
            filter_size*num_features*dim*6,
            num_outputs
        ]
        self.linear1 = nn.Linear(units[0], units[1])
        self.linear2 = nn.Linear(units[1], units[2])
        
    def forward(self, x):
        # x -> batch_size, filter_size, num_features, dim
        x = x.flatten(1) # -> batch_size, -1 
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x) # -> batch_size, filter_size*num_features*dim/8
        return x


# ### MultiOutputsModel

# +
class MultiOutputsModel(nn.Module):
    def __init__(
        self,
        num_inputs:int=128,
        num_outputs:int=10,
        units:list=[64, 128, 64, 32],
        units_sub:list=[16, 32, 8, 1],
    ):
        super(MultiOutputsModel, self).__init__()
        self.name = "MultiOutputsModel"
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        units = [num_inputs] + units
        
        self.main_mlp = nn.ModuleList([
            ResidualBlock(units[i], units[i+1])
            for i in range(len(units)-1)
        ])

        self.last_output_layers = nn.ModuleList([
            MultiOutputsModelSub(units[-1], units_sub=[16, 8, 1])
            for _ in range(num_outputs)
        ])
        
    def forward(self, x):
        for l in self.main_mlp:
            x = l(x)
        
        last_outputs = []
        for l in self.last_output_layers:
            last_outputs.append(l(x.clone()))
            
        x = torch.concat(last_outputs, dim=1)
        return x
    
class MultiOutputsModelSub(nn.Module):
    def __init__(self, num_inputs, units_sub=[16, 8, 1]):
        super(MultiOutputsModelSub, self).__init__()
        self.geglu_layer = GeGLULayer(num_inputs, units_sub[0])
        self.units_sub = units_sub
        self.ls_sub =  nn.ModuleList([
            LeakyReLULayer(units_sub[i], units_sub[i+1])
            if i < len(units_sub)-2 else
            SigmoidLayer(units_sub[i], units_sub[i+1])
            for i in range(len(units_sub)-1)
        ])
                
    def forward(self, x):
        x = self.geglu_layer(x)
        for i, l in enumerate(self.ls_sub):
            if i+1 < len(self.units_sub)-2:
                x = l(x)
            else:
                x = F.dropout(x, .1, training=self.training)
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
     
class GeGLULayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GeGLULayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)
        self.linear = nn.Linear(num_inputs, num_outputs*2)
        self.geglu = GeGLU()
                
    def forward(self, x):
        x = self.linear(self.bn(x))
        x = self.geglu(x)
        return x

class GeGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x*F.gelu(gates)
    
class ResidualBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ResidualBlock, self).__init__()
        self.layer1 = LeakyReLULayer(num_inputs, 64)
        self.layer2 = LeakyReLULayer(64, 32)
        self.layer3 = LeakyReLULayer(32, num_inputs)
        self.geglu = GeGLULayer(num_inputs, num_outputs)
                
    def forward(self, x):
        x_mlp = self.layer1(x.clone())
        x_mlp = self.layer2(x_mlp)
        x += self.layer3(x_mlp)
        x = self.geglu(x)
        return x


# -

# ### Pretraining Function

def pretraining(
    pretrain_loader:DataLoader,
    dp:DataProcessing,
    level_group:str="0-4",
    epochs:int=CFG.PRETRAIN_EPOCHS,
    lambda1:float=1.0,
    lambda2:float=1.0,
    lambda3:float=1.0,
    pretrain_loader_val:DataLoader=None,
    pretrain_sessions:list=None,
    pretrain_sessions_val:list=None,
    omit:bool=False,
    omit_data_dict:bool=False,
    agg_func=torch.mean,
):
    pretrain_model = PretrainModel(
        num_total_tokens=dp.num_total_tokens,
        num_continuous=dp.num_continuous,
        categories=dp.categories,
        filter_size=dp.filter_size,
        num_features=dp.num_features,
        dim=32,
        num_outputs_projection_head=128,
        heads=8,
        dim_head=16,
        attention_dropout=.1,
    ).to(CFG.DEVICE)

    optimizer = torch.optim.AdamW(pretrain_model.parameters(),lr=0.003)
    losses = [10**10]
    checkpoint_path = f"{CFG.CHECKPOINT_PATH}/pretrain_params_{level_group}.pth"
    pretrain_model.train()

    if omit:
        pretrain_model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Pretraining is starting...") 
        for epoch in tqdm(range(CFG.PRETRAIN_EPOCHS)):
            loss_sum, loss_fve_cat_sum, loss_frp_sum = 0.0, 0.0, 0.0
            for i, (x_categorical, x_continuous) in enumerate(pretrain_loader):
                x_categorical, x_continuous = x_categorical.to(CFG.DEVICE), x_continuous.to(CFG.DEVICE)
                loss, loss_fve_cat, loss_frp = pretrain_model(x_categorical, x_continuous)
                loss = lambda1*loss + lambda2*loss_fve_cat + lambda3*loss_frp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                loss_fve_cat_sum += loss_fve_cat.item()
                loss_frp_sum += loss_frp.item()

            epoch_loss = (loss_sum + loss_fve_cat_sum + loss_frp)/i
            if epoch_loss < min(losses):
                best_epoch = epoch
                torch.save(pretrain_model.state_dict(), checkpoint_path)
            losses.append(epoch_loss)

            print(f"Epoch {epoch+1}/{CFG.PRETRAIN_EPOCHS}: loss{epoch_loss: .4f} (contrastive{loss_sum/i: .4f}, fve_cat{loss_fve_cat_sum/i: .4f}, frp{loss_frp_sum/i: .4f})")
        
        pretrain_model.load_state_dict(torch.load(checkpoint_path))
        print(f"Pretraining has finished.\nBest Epoch is {best_epoch} with loss {min(losses)} !")
    
    if pretrain_sessions:
        agg_func_str = "mean" if agg_func==torch.mean else "sum" if agg_func==torch.sum else None 
        pretrain_model.eval()
        
        if omit_data_dict:
            with open(f"{CFG.CHECKPOINT_PATH}/x_dict_train_{level_group}_{agg_func_str}.pickle", "rb") as f:
                x_dict_train = pickle.load(f)
            with open(f"{CFG.CHECKPOINT_PATH}/x_dict_val_{level_group}_{agg_func_str}.pickle", "rb") as f:
                x_dict_val = pickle.load(f)
        else:
            x_train = []
            for x_categorical, x_continuous in pretrain_loader:
                x_categorical, x_continuous = x_categorical.to(CFG.DEVICE), x_continuous.to(CFG.DEVICE)
                x_train += [pretrain_model.encoder(x_categorical, x_continuous).detach().cpu()]
            
            x_train = torch.concat(x_train, dim=0)
            session_before = pretrain_sessions[0]
            x_dict_train = {session_before:[]}
            length = len(pretrain_sessions)-1
            for i, (session, data) in enumerate(zip(pretrain_sessions, x_train)):
                if i == length:
                    x_dict_train[session] += [data]                    
                    x_dict_train[session] = torch.mean(
                        torch.stack(x_dict_train[session]),
                        dim=0
                    )*len(x_dict_train[session])**.2
                else:
                    if session_before != session:
                        x_dict_train[session_before] = agg_func(
                            torch.stack(x_dict_train[session_before]),
                            dim=0
                        )*len(x_dict_train[session_before])**.2
                        x_dict_train[session] = []
                        session_before = session

                    x_dict_train[session] += [data]
                
            x_val = []
            for x_categorical, x_continuous in pretrain_loader_val:
                x_categorical, x_continuous = x_categorical.to(CFG.DEVICE), x_continuous.to(CFG.DEVICE)
                x_val += [pretrain_model.encoder(x_categorical, x_continuous).detach().cpu()]
            
            x_val = torch.concat(x_val, dim=0)
            session_before = pretrain_sessions_val[0]
            x_dict_val = {session_before:[]}
            length = len(pretrain_sessions_val)-1
            for i, (session, data) in enumerate(zip(pretrain_sessions_val, x_val)):
                if i == length:
                    x_dict_val[session] += [data]                    
                    x_dict_val[session] = agg_func(
                        torch.stack(x_dict_val[session]),
                        dim=0
                    )*len(x_dict_val[session])**.2
                else:
                    if session_before != session:
                        x_dict_val[session_before] = torch.mean(
                            torch.stack(x_dict_val[session_before]),
                            dim=0
                        )*len(x_dict_val[session_before])**.2
                        x_dict_val[session] = []
                        session_before = session

                    x_dict_val[session] += [data]
                
            del x_train, x_val
            gc.collect()

            with open(f"{CFG.CHECKPOINT_PATH}/x_dict_train_{level_group}_{agg_func_str}.pickle", "wb") as f:
                pickle.dump(x_dict_train, f)
            with open(f"{CFG.CHECKPOINT_PATH}/x_dict_val_{level_group}_{agg_func_str}.pickle", "wb") as f:
                pickle.dump(x_dict_val, f)
        
        return pretrain_model, x_dict_train, x_dict_val
    
    else:
        return pretrain_model


# ### Training Function

def training(
    train_loader:DataLoader,
    val_loader:DataLoader,
    model_class,
    num_inputs:int,
    level_group:str="0-4",
    omit:bool=CFG.TRAINING_OMIT
):
    num_outputs = len(lq_dict[level_group])
    model = model_class(num_inputs, num_outputs).to(CFG.DEVICE)
    checkpoint_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_bestmodel{'_'+level_group if level_group else ''}{'_all' if CFG.PREDICT_ALL else ''}.pth"
    
    if omit:
        checkpoint_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_bestmodel{'_'+level_group if level_group else ''}{'_all' if CFG.PREDICT_ALL else ''}.pth"
        model.load_state_dict(torch.load(checkpoint_path))
        with open(f"{CFG.CHECKPOINT_PATH}/{model.name}_thresholds{'_all' if CFG.PREDICT_ALL else ''}.pickle", "rb") as f:
            best_thresholds = pickle.load(f)[level_group]
        return model, best_thresholds
    else:
        print("Training is starting...") 
        
        weight_ratio = calc_weight(train_loader).to(CFG.DEVICE)
        early_stopping_count = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=.005
        )
        loss_trains, loss_vals, f1_trains, f1_vals = [], [], [], []
        loss_func = nn.BCELoss(reduction='none')
        #loss_func_correct_rate = nn.MSELoss()

        for epoch in range(CFG.EPOCHS):
            model.train()
            loss_train = 0.0
            preds, true_values = [], []
            for i, (x, t) in enumerate(train_loader):
                x, t = x.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x)
                preds += y
                true_values += t
                
                weight = (lambda w: torch.where(w==0, 1, w))(t*weight_ratio)
                loss = torch.mean(loss_func(y, t)*weight) / num_outputs
                
                #correct_rate = torch.mean(torch.where(y>CFG.THRESHOLD, 1, 0), dim=1)
                #loss_correct_rate = loss_func_correct_rate(
                #    correct_rate,
                #    torch.mean(t, dim=1)
                #)
                
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
            loss_val = 0.0
            preds, true_values = [], []
            for i, (x, t) in enumerate(val_loader):
                x, t = x.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x)
                preds += y
                true_values += t
                
                weight = (lambda w: torch.where(w==0, 1, w))(t*weight_ratio)
                loss = torch.mean(loss_func(y, t)*weight) / num_outputs
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
        for i, (x, t) in enumerate(val_loader):
            x, t = x.to(CFG.DEVICE), t.to(CFG.DEVICE)
            y = model(x).view(-1)
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
            for i, (x, t) in enumerate(val_loader):
                x, t = x.to(CFG.DEVICE), t.to(CFG.DEVICE)
                y = model(x)
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

        torch.cuda.empty_cache()
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


# ### Other Functions

# +
def calc_weight(data_loader):
    count1 = torch.sum(data_loader.dataset.tensors[1], dim=0)
    count_all = data_loader.dataset.tensors[1].shape[0]
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
    for i, (x, t) in enumerate(val_loader):
        x, t = x.to(CFG.DEVICE), t.to(CFG.DEVICE)
        y = model(x).view(-1)
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

# # Execution

# ## Preparing Objects

# +
# %%time

if CFG.SUBMISSION_MODE:
    print("SUBMISSION_MODE")
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
        
elif CFG.RESUME:
    print("RESUME_MODE")
    dps = {}
    with open(f"{CFG.CHECKPOINT_PATH}/datas.pickle", "rb") as f:
        datas = pickle.load(f)
    train_labels = datas["0-4"]["dp"].get_labels()
    
elif CFG.CHECKPOINT:
    print("CHECKPOINT_MODE")
    datas = {}
    for i, level_group in enumerate(lq_dict.keys()):
        dp = DataProcessing(level_group=level_group, stride=10, filter_size=20)
        train = dp.fit_transform(f"{CFG.INPUT}/train.csv")
        test = dp.transform(f"{CFG.INPUT}/test.csv")
        # -> x_categorical, x_continuous, session_index

        if i==0:
            train_labels = dp.get_labels()

        datas[level_group] = dict(dp=dp, train=train, test=test)

    with open(f"{CFG.CHECKPOINT_PATH}/datas.pickle", "wb") as f:
        pickle.dump(datas, f)                
        
    if False:
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
        if level_group=="13-22": continue
        print(f"\nlevel_group {level_group} model...")
        
        # create pretrain loader
        x_train_loader, x_train_sessions, y_train, x_val_loader, x_val_sessions, y_val = \
            create_loader_pretrain(
                x_categorical=datas[level_group]["train"][0],
                x_continuous=datas[level_group]["train"][1],
                y=train_labels,
                session_index=datas[level_group]["train"][2],
                level_group=level_group,
            )
        
        # pretraining
        pretrain_model, x_dict_train, x_dict_val = pretraining(
            pretrain_loader=x_train_loader,
            dp=datas[level_group]["dp"],
            level_group=level_group,
            epochs=CFG.PRETRAIN_EPOCHS,
            pretrain_loader_val=x_val_loader,
            pretrain_sessions=x_train_sessions,
            pretrain_sessions_val=x_val_sessions,
            omit=CFG.PRETRAIN_OMIT,
            omit_data_dict=CFG.PRETRAIN_OMIT,
            agg_func=torch.mean
        )
        
        del x_train_loader, x_val_loader
        gc.collect()
        
        # create train loader
        train_loader, val_loader = create_loader_train(
            x_dict_train, y_train, x_dict_val, y_val
        )
                    
        # training
        model, best_threshold = training(
            train_loader,
            val_loader,
            model_class=MultiOutputsModel,
            num_inputs=pretrain_model.num_outputs_projection_head,
            level_group=level_group,
            omit=CFG.TRAINING_OMIT
        )
        
        #best_threshold = explore_threshold(model, val_loader)
            
        # predicting
        #submission = predict(test[level_group], model, level_group, best_threshold)
        #display(submission)
        #submission.to_csv("submission.csv", index=False)
        
        thresholds[level_group] = best_threshold
        
        torch.cuda.empty_cache()
    
    thresholds_path = f"{CFG.CHECKPOINT_PATH}/{model.name}_thresholds{'_all' if CFG.PREDICT_ALL else ''}.pickle"
    with open(thresholds_path, "wb") as f:
        pickle.dump(thresholds, f)
# -


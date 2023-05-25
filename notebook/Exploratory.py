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

# # Columns Explanations
# + session_id - the ID of the session the event took place in
# + index - the index of the event for the session
# + elapsed_time - how much time has passed (in milliseconds) between the start of the session and when the event was recorded
# + event_name - the name of the event type
# + name - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)
# + level - what level of the game the event occurred in (0 to 22)
# + page - the page number of the event (only for notebook-related events)
# + room_coor_x - the coordinates of the click in reference to the in-game room (only for click events)
# + room_coor_y - the coordinates of the click in reference to the in-game room (only for click events)
# + screen_coor_x - the coordinates of the click in reference to the player’s screen (only for click events)
# + screen_coor_y - the coordinates of the click in reference to the player’s screen (only for click events)
# + hover_duration - how long (in milliseconds) the hover happened for (only for hover events)
# + text - the text the player sees during this event
# + fqid - the fully qualified ID of the event
# + room_fqid - the fully qualified ID of the room the event took place in
# + text_fqid - the fully qualified ID of the
# + fullscreen - whether the player is in fullscreen mode
# + hq - whether the game is in high-quality
# + music - whether the game music is on or off
# + level_group - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)

# # Preparation

# +
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from tqdm import tqdm
import missingno

warnings.simplefilter("ignore")
pd.options.display.max_columns=200
pd.options.display.max_rows=150

class CFG:
    INPUT = "../input"


# +
# %%time

train = pd.read_csv(f"{CFG.INPUT}/train.csv")
train_labels = pd.read_csv(f"{CFG.INPUT}/train_labels.csv")
test = pd.read_csv(f"{CFG.INPUT}/test.csv")
sample_submission = pd.read_csv(f"{CFG.INPUT}/sample_submission.csv")

train_labels[["session_id", "question"]] = train_labels.session_id.str.split("_", expand=True)
train_labels = train_labels[["session_id", "question", "correct"]]

display(train)
display(train_labels)
display(test)
display(sample_submission.sort_values("session_id"))
# -

# # EDA

# ## Correct Rate

# +
questions = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18']

check = pd.pivot(data=train_labels, index="session_id", columns="question", values="correct")
check = check[questions]

display(px.bar(check.mean(axis=0), title="Correct Rate by Questions."))

check["correct_rate"] = check.mean(axis=1)
display(px.histogram(check.correct_rate, title="Correct Rate Histogram by Sessions." ))

check["relative_group"] = pd.qcut(check.correct_rate, 4, labels=[1, 2, 3, 4])
check["abs_group"] = pd.cut(check.correct_rate, 4, labels=[1, 2, 3, 4])

display(check)

# +
# %%time

res = {}

for session_id, row in check[questions].iterrows():
    tmp = []
    for i1, q1 in enumerate(row):
        for i2, q2 in enumerate(row):
            tmp.append((questions[i1], questions[i2], int(q1==q2)))
    tmp = pd.DataFrame(tmp)
    tmp = pd.pivot(data=tmp, index=0, columns=1, values=2)
    res[session_id] = np.array(tmp.reindex(index=questions, columns=questions))

cooccurence_rate = np.sum(list(res.values()), axis=0) / len(res.keys())
cooccurence_rate = pd.DataFrame(data=np.tril(cooccurence_rate), columns=questions)
cooccurence_rate.index = questions

sns.heatmap(cooccurence_rate, cmap="Blues", annot=True)
plt.title("correct/uncorrect cooccurence_rate between questions");
# -

check.relative_group.value_counts().sort_index()
check.abs_group.value_counts().sort_index()
check.correct_rate.describe()

# ## Variables

# ### train data

# #### base analysis

train.info()

print("Null Rate")
train.isnull().sum() / len(train)

missingno.matrix(train)

train.describe()

# #### create eda train dataset

# +
# %%time

tmp = check[["correct_rate", "relative_group", "abs_group"]].reset_index()
tmp.session_id = tmp.session_id.astype(int)
train4eda = pd.merge(train, tmp, how = "left", on = "session_id")

tmp = []
for i, row in tqdm(train[["fqid", "room_fqid", "text_fqid"]].fillna("").iterrows()):
    tfqid = row.text_fqid.replace(row.room_fqid, "").replace(row.fqid, "")
    tfqid = tfqid.replace("..", ".")
    tmp.append(tfqid[1:] if tfqid.startswith(".") else None)
train4eda["text_fqid"] = tmp

train4eda["room_fqid"] = [f.replace("tunic.", "") for f in train4eda.room_fqid]
train4eda[["room_fqid_1st", "room_fqid_2nd"]] = train4eda.room_fqid.str.split(".", expand=True)

train4eda[["fqid_"+i for i in ["1st", "2nd", "3rd"]]] = train4eda.fqid.str.split(".", expand=True)
train4eda["fqid_1st"] = [fqid.split("_")[0] if b else None for fqid,b in zip(train4eda["fqid_1st"], train4eda["fqid_1st"].notnull())]
train4eda["fqid_2nd"] = [fqid.split("_")[0] if b else None for fqid,b in zip(train4eda["fqid_2nd"], train4eda["fqid_2nd"].notnull())]

train4eda[["text_fqid_"+i for i in ["1st", "2nd", "3rd"]]] = train4eda.text_fqid.str.split(".", expand=True)
train4eda["text_fqid_1st"] = [text_fqid.split("_")[0] if b else None for fqid,b in zip(train4eda["text_fqid_1st"], train4eda["text_fqid_1st"].notnull())]
train4eda["text_fqid_2nd"] = [text_fqid.split("_")[0] if b else None for fqid,b in zip(train4eda["text_fqid_2nd"], train4eda["text_fqid_2nd"].notnull())]

train4eda[["event_name", "event_type"]] = train4eda.event_name.str.split("_", expand=True)
# -

train4eda.text_fqid.value_counts().sort_index()

# #### continuous variances

# #### relations between fqids

pd.options.display.max_rows=200
train.groupby(["fqid", "room_fqid", "text_fqid"])["session_id", "index"].count()

train4eda.room_fqid.str.split(".", expand=True)
train4eda.fqid.str.split(".", expand=True).rename(columns={0:"1st", 1:"2nd", 2:"3rd"}).value_counts(dropna=False).sort_index()

# +
# %%time

x="room_coor_x"
y="room_coor_y"

tmp = train4eda.groupby(["session_id", "relative_group"])[[x,y]].mean().reset_index()
sns.jointplot(data=tmp, x=x, y=y, hue="relative_group", alpha=.3)

# +
x="screen_coor_x"
y="screen_coor_y"

tmp = train4eda.groupby(["session_id", "relative_group"])[[x,y]].mean().reset_index()
sns.jointplot(data=tmp, x=x, y=y, hue="relative_group", alpha=.3)
# -

print("Number of unique sessions", train4eda.session_id.nunique())
tmp = train4eda.groupby(["session_id", "relative_group"]).agg(logs_in_a_session=("index", "count"))
tmp = tmp[tmp>0].reset_index()
px.histogram(data_frame=tmp, x="logs_in_a_session", color="relative_group", opacity=.3,
             barmode="overlay", title="number of logs in a session")

tmp = train4eda.groupby(["session_id", "relative_group"]).agg(max_elapsed_time_in_a_session=("elapsed_time", "max")).reset_index()
tmp = tmp[tmp.max_elapsed_time_in_a_session.notnull()].sort_values("relative_group")
#tmp.max_elapsed_time_in_a_session /= 1e6
px.histogram(data_frame=tmp, x="max_elapsed_time_in_a_session", color="relative_group", opacity=.3,
             barmode="overlay", title="max_elapsed_time_in_a_session", histnorm="probability", cumulative=True,
             facet_row="relative_group", height=800)

# #### categorical variances

# +
group = "relative_group"

for col in train4eda.columns:
    if train4eda[col].dtype == object:
        if len(train4eda[col].unique()) <= 40:
            tmp = [train4eda.query(f"{group}=={g}").value_counts([group, col], dropna=False, normalize=True).reset_index()
           for g in train4eda[group].unique()]
            tmp = pd.concat(tmp, axis=0, ignore_index=True).sort_values([group,col]).reset_index(drop=True)
            display(px.bar(data_frame=tmp.rename(columns={0:"rate"}), x=col, y="rate", color=group, barmode="group"))
        else:
            print(f"{col} has so much various values.")
# -

train4eda.value_counts(["event_type", "event_name", "name"], dropna=False, normalize=True).sort_index()

train4eda[["fqid_1st", "fqid_2nd", "fqid_3rd"]].value_counts(dropna=False).sort_index()

sns.jointplot(x=test.room_coor_x, y=test.room_coor_y, alpha=.3)
sns.jointplot(x=test.screen_coor_x, y=test.screen_coor_y, alpha=.3)

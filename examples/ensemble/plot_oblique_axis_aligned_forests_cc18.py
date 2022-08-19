"""
===============================================================================
Plot oblique forest and axis-aligned random forest predictions on cc18 datasets
===============================================================================

A performance comparison between oblique forest and standard axis-
aligned random forest using three datasets from OpenML benchmarking suites.

Two of these datasets, namely [WDBC](https://www.openml.org/search?type=data&sort=runs&id=1510) 
and [Phishing Website](https://www.openml.org/search?type=data&sort=runs&id=4534) datasets
consist of 31 features where the former dataset is entirely numeric
and the latter dataset is entirely norminal. The third dataset, dubbed 
[cnae-9](https://www.openml.org/search?type=data&status=active&id=1468), is a
numeric dataset that has notably large feature space of 857 features. As you
will notice, of these three datasets, the oblique forest outperforms axis-aligned
random forest on cnae-9 utilizing sparse random projection machanism. All datasets
are subsampled due to computational constraints.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import openml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ObliqueRandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate

random_state = 123456
t0 = datetime.now()
data_ids = [11, 40499]  # openml dataset id
df = pd.DataFrame()


def load_cc18(data_id):
    dat = openml.datasets.get_dataset(data_id, download_data=False)
    d_name = dat.name
    d = dat.get_data()[0]

    # Subsampling large datasets
    n = int(d.shape[0] * 0.1)
    d = d.sample(n, random_state=random_state)
    X, y = d.iloc[:, :-1], d.iloc[:, -1]

    return X, y, d_name


def get_scores(X, y, d_name="UNK", n_cv=5, n_repeats=2, random_state=1, kwargs=None):
    clfs = [
        RandomForestClassifier(**kwargs[0], random_state=random_state),
        ObliqueRandomForestClassifier(**kwargs[1], random_state=random_state),
    ]

    tmp = []

    for i, clf in enumerate(clfs):
        cv = RepeatedKFold(
            n_splits=n_cv, n_repeats=n_repeats, random_state=random_state
        )
        test_score = cross_validate(estimator=clf, X=X, y=y, cv=cv, scoring="accuracy")

        tmp.append(
            [
                d_name,
                ["RF", "OF"][i],
                test_score["test_score"],
                test_score["test_score"].mean(),
            ]
        )
        print(
            f'{d_name} mean test score for {["RF", "OF"][i]}:'
            f' {test_score["test_score"].mean()}'
        )

    df = pd.DataFrame(tmp, columns=["dataset", "model", "score", "mean"])
    df = df.explode("score")
    df["score"] = df["score"].astype(float)
    df.reset_index(inplace=True, drop=True)

    return df


def load_best_params(data_ids):
    # folder_path = "/home/jshinm/Desktop/workstation/sklearn-jms/notebook/hidden/output/"
    folder_path = None
    params = []

    if not folder_path:
        # pre-tuned hyper-parameters
        params += [
            [
                {"max_depth": 5, "max_features": "sqrt", "n_estimators": 100},
                {"max_depth": 5, "max_features": None, "n_estimators": 100},
            ],
            [
                {"max_depth": 10, "max_features": "log2", "n_estimators": 200},
                {"max_depth": 10, "max_features": 80, "n_estimators": 200},
            ],
        ]
    else:
        for data_id in data_ids:
            file_path = f"OFvsRF_grid_search_cv_results_openml_{data_id}.csv"
            df = pd.read_csv(folder_path + file_path).sort_values(
                "mean_test_score", ascending=False
            )
            tmp = []
            for clf in ["RF", "OF"]:
                tmp.append(eval(df.query(f'clf=="{clf}"')["params"].iloc[0]))
            params.append(tmp)

    return params


params = load_best_params(data_ids=data_ids)

for i, data_id in enumerate(data_ids):
    X, y, d_name = load_cc18(data_id=data_id)
    print(f"Loading [{d_name}] dataset..")
    tmp = get_scores(
        X=X, y=y, d_name=d_name, random_state=random_state, kwargs=params[i]
    )
    df = pd.concat([df, tmp])

t_d = (datetime.now() - t0).seconds
print(f"It took {t_d} seconds to run the script")

# Draw a comparison plot
d_names = df.dataset.unique()
N = d_names.shape[0]

fig, ax = plt.subplots(1, N, figsize=(6 * N, 6))

for i, name in enumerate(d_names):
    if N == 1:
        axs = ax
    else:
        axs = ax[i]
    dff = df.query(f'dataset == "{name}"')

    sns.stripplot(data=dff, x="model", y="score", ax=axs, dodge=True)
    sns.boxplot(data=dff, x="model", y="score", ax=axs, color="white")
    axs.set_title(f"{name} (#{data_ids[i]})")

    rf = dff.query('model=="RF"')["mean"].iloc[0]
    rff = f"RF (Mean Test Score: {round(rf,3)})"

    of = dff.query('model=="OF"')["mean"].iloc[0]
    off = f"OF (Mean Test Score: {round(of,3)})"

    axs.legend([rff, off], loc=4)

    if i != 0:
        axs.set_ylabel("")
    else:
        axs.set_ylabel("Accuracy")

    axs.set_xlabel("")

plt.savefig(f"plot_cc18_{t_d}s.jpg")
plt.show()

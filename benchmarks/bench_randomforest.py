"""Instructions
1. Build this PR and run:

```bash
python bench_randomforest.py bench ~/bench_results_forest pr
```

2. On main run:

```bash
python bench_randomforest.py bench ~/bench_results_forest main
```

3. Plotting

```bash
python bench_randomforest.py plot ~/bench_results_forest pr main results_image.png

# or plot size
python bench_randomforest.py plot_size ~/bench_results_forest pr main results_image.png
```
"""
import os
import tempfile
import sys
import pickle
from functools import partial
import argparse
from time import perf_counter
from statistics import mean, stdev
from itertools import product
import csv
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression, make_low_rank_matrix
import numpy as np

N_REPEATS = 10
n_jobs = -3

benchmark_config = [
    (
        RandomForestRegressor,
        list(
            product(
                ["squared_error"],
                [
                    make_regression,
                ],
                [10_000],
                ["dense"],
                ["best"],
            )
        ),
    ),
    (
        RandomForestClassifier,
        list(
            product(
                ["gini", "entropy"],
                [
                    partial(make_classification, n_informative=10, n_classes=5),
                ],
                [10_000],
                ["dense"],
                ["best"],
            )
        ),
    ),
]

def bench(args):
    bench_results, branch = args.bench_results, args.branch
    results_dir = Path(bench_results)
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / f"{branch}.csv"

    with results_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "criterion",
                "n_samples",
                "make_data",
                "container",
                "splitter",
                "n_repeat",
                "duration",
                "ram_size",
                "file_size",
            ],
        )
        writer.writeheader()

        for Klass, items in benchmark_config:

            for config in items:
                (
                    criterion,
                    make_data,
                    n_samples,
                    container,
                    splitter,
                ) = config
                if isinstance(make_data, partial):
                    make_data_str = make_data.func.__name__
                else:
                    make_data_str = make_data.__name__

                default_config = {
                    "criterion": criterion,
                    "n_samples": n_samples,
                    "make_data": make_data_str,
                    "container": container,
                    "splitter": splitter,
                }
                combine_config = " ".join(f"{k}={v}" for k, v in default_config.items())

                klass_results = []
                for n_repeat in range(N_REPEATS):
                    print(f"Running {combine_config} with {n_repeat + 1}/{N_REPEATS}")
                    X, y = make_data(
                        n_samples=n_samples,
                        n_features=20,
                        random_state=n_repeat,
                    )
                    forest = Klass(random_state=n_repeat, criterion=criterion, n_jobs=n_jobs)

                    start = perf_counter()
                    forest.fit(X, y)
                    duration = perf_counter() - start
                    klass_results.append(duration)

                    # benchmark size of object
                    ram_size = sys.getsizeof(forest)
                    with tempfile.TemporaryFile() as f:
                        pickle.dump(forest, f, -1)
                        file_size = os.path.getsize(f.name)

                    writer.writerow(
                        {
                            **default_config,
                            **{
                                "n_repeat": n_repeat,
                                "duration": duration,
                                "ram_size": ram_size,
                                "file_size": file_size,
                            },
                        }
                    )
                results_mean, results_stdev = mean(klass_results), stdev(klass_results)
                print(
                    f"{combine_config} with {results_mean:.3f} +/- {results_stdev:.3f}"
                )

def plot(args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_path = Path(args.bench_results)
    pr_path = results_path / f"{args.pr_name}.csv"
    main_path = results_path / f"{args.main_name}.csv"
    image_path = results_path / args.image_path

    df_pr = pd.read_csv(pr_path).assign(branch=args.pr_name)
    df_main = pd.read_csv(main_path).assign(branch=args.main_name)
    df_all = pd.concat((df_pr, df_main), ignore_index=True)

    df_all = df_all.assign(
        make_data=df_all["make_data"]
        .str.replace("_custom", "")
        .str.replace("make_", "")
        .str.replace("_data", "")
    )

    gb = df_all.groupby(["criterion", "make_data"])
    groups = gb.groups

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (keys, idx) in enumerate(groups.items()):
        ax = axes_flat[i]
        ax.set_title(" | ".join(keys))
        sns.boxplot(data=df_all.loc[idx], y="duration", x="branch", ax=ax)
        if i % n_cols != 0:
            ax.set_ylabel("")

    axes_flat[-1].set_visible(False)

    fig.savefig(image_path)
    print(f"Saved image to {image_path}")


def plot_size(args):
    size_id = 'file_size'

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_path = Path(args.bench_results)
    pr_path = results_path / f"{args.pr_name}.csv"
    main_path = results_path / f"{args.main_name}.csv"
    image_path = results_path / args.image_path

    df_pr = pd.read_csv(pr_path).assign(branch=args.pr_name)
    df_main = pd.read_csv(main_path).assign(branch=args.main_name)
    df_all = pd.concat((df_pr, df_main), ignore_index=True)

    df_all = df_all.assign(
        make_data=df_all["make_data"]
        .str.replace("_custom", "")
        .str.replace("make_", "")
        .str.replace("_data", "")
    )

    gb = df_all.groupby(["criterion", "make_data"])
    groups = gb.groups

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (keys, idx) in enumerate(groups.items()):
        ax = axes_flat[i]
        ax.set_title(" | ".join(keys))
        sns.boxplot(data=df_all.loc[idx], y=size_id, x="branch", ax=ax)
        if i % n_cols != 0:
            ax.set_ylabel("")

    axes_flat[-1].set_visible(False)

    fig.savefig(image_path)
    print(f"Saved image to {image_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("bench_results")
    bench_parser.add_argument("branch")
    bench_parser.set_defaults(func=bench)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("bench_results")
    plot_parser.add_argument("pr_name")
    plot_parser.add_argument("main_name")
    plot_parser.add_argument("image_path")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)

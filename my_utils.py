import pandas as pd

def get_df_info(df, thr=0.8, **kwargs):
    """
    Выводит инфу о колонках датафрейма в виде датафрейма

    Выводит информацию о типе данных, содержании
    missing values и кол-во уникальных элементов

    df: исходный датафрейм
    thr: порог для доли самого частого элемента vc_max_prop,
        если vc_max_prop > thr, то vc_max_prop может войти в trash_score

    returns: pd.DataFrame с инфой

    """
    df_info = pd.DataFrame(
        index=df.columns,
        columns=[
            "dtype",
            "nunique",
            "example1",
            "example2",
            "zero",
            "nan",
            "empty_str",
            "vc_max",
            "vc_max_prop",
            "trash_score",
        ],
    )

    for col in df.columns:
        df_info.loc[col, "dtype"] = df[col].dtype.name
        df_info.loc[col, "nunique"] = df[col].nunique(dropna=False)
        if df[col].isna().sum() == 0:
            df_info.loc[col, "nan"] = -1
        else:
            df_info.loc[col, "nan"] = "n: " + str(
                round(df[col].isna().sum() / df.shape[0], 3)
            )

        if df[df[col] == 0].shape[0] == 0:
            df_info.loc[col, "zero"] = -1
        else:
            df_info.loc[col, "zero"] = "z: " + str(
                round(df[df[col] == 0].shape[0] / df.shape[0], 3)
            )

        if df[df[col] == ""].shape[0] == 0:
            df_info.loc[col, "empty_str"] = -1
        else:
            df_info.loc[col, "empty_str"] = "e: " + str(
                round(df[df[col] == ""].shape[0] / df.shape[0], 3)
            )

        df_info["vc_max"] = df.mode().iloc[0]

        df_info.loc[col, "vc_max_prop"] = round(
            df[df[col] == df.mode().iloc[0][col]].shape[0] / df.shape[0], 3
        )

        if (
            df_info["zero"][col] == -1
            and df_info["nan"][col] == -1
            and df_info["empty_str"][col] == -1
        ):
            df_info.loc[col, "trash_score"] = -1
        else:
            df_info.loc[col, "trash_score"] = round(
                (
                    df[col].isna().sum()
                    + df[df[col] == 0].shape[0]
                    + df[df[col] == ""].shape[0]
                )
                / df.shape[0],
                3,
            )

        if df_info.loc[col, "vc_max_prop"] > thr:
            df_info.loc[col, "trash_score"] = max(
                df_info.loc[col, "trash_score"], df_info.loc[col, "vc_max_prop"]
            )
        else:
            df_info.loc[col, "trash_score"] = max(df_info.loc[col, "trash_score"], -1)
        if df[col].nunique() == 0:
            df_info.loc[col, "example1"] = "<Not determined>"
            df_info.loc[col, "example2"] = "<Not determined>"
        elif df[col].nunique() == 1:
            df_info.loc[col, "example1"] = df[col].dropna().unique()[0]
            df_info.loc[col, "example2"] = "<Not determined>"
        else:
            df_info.loc[col, "example1"] = df[col].dropna().unique()[0]
            df_info.loc[col, "example2"] = df[col].dropna().unique()[1]
    return df_info

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# настройка размера шрифта легенды
mpl.rcParams['legend.title_fontsize'] = 13
mpl.rcParams['legend.fontsize'] = 13

def plot_density(df, hue, cols=None, max_cat_thr=20, drop_zero=0):
    """
    Рисует распределения колонок cols

    Распределение данных колонок cols относительно hue;
    для численных колонок: histplot, boxenplot+stripplot,
        barplot с распределением по 0 и NaN
    для категориальных колонок: countplot

    Args:
        df: датафрейм с данными для отрисовки
        hue: колонка, по которой бьется раскраска
        cols: отрисовываемые колонки. Если None, то рисуем df.columns (кроме hue)
        max_cat_thr: максимальное число категорий, колонки с числом
                    уникальных значений <= max_cat_thr считаются категориальными
        drop_zero: если True, отрисовка данных без нулей
    """
    if cols is None:
        cols = df.columns
    cols_num = []
    cols_cat = []
    for col in cols:
        if df[col].nunique() <= max_cat_thr:
            cols_cat.append(col)
        elif df[col].dtype == int or df[col].dtype == float:
            cols_num.append(col)
    for col in cols_cat:
        if df[col].dtype == "category":
            df[col] = df[col].astype("object")

    hue_order = df[hue].sort_values().unique()

    for col in cols_num:
        if col == hue:
            continue
        fig, ax = plt.subplot_mosaic("abc")
        if drop_zero:
            sns.histplot(
                data=df[df[col] != 0],
                x=col,
                hue=hue,
                multiple="stack",
                bins=25,
                element="step",
                stat="count",
                alpha=0.8,
                ax=ax["a"],
                hue_order=hue_order,
            )
        else:
            sns.histplot(
                data=df,
                x=col,
                hue=hue,
                multiple="stack",
                bins=25,
                element="step",
                stat="count",
                alpha=0.8,
                ax=ax["a"],
                hue_order=hue_order,
            )
        ax["a"].set_ylabel("Count", fontsize=15)
        ax["a"].set_xlabel(None)
        ax["a"].tick_params(axis="both", labelsize=15)

        if drop_zero:
            sns.boxenplot(
                data=df[df[col] != 0],
                y=col,
                x=hue,
                ax=ax["b"],
                hue=hue,
                hue_order=hue_order,
                showfliers=False,
            )
            sns.stripplot(
                data=df[df[col] != 0].groupby(hue).sample(20),
                y=col,
                x=hue,
                hue_order=hue_order,
                ax=ax["b"],
                color="black",
                size=4,
            )
        else:
            sns.boxenplot(
                data=df,
                y=col,
                x=hue,
                ax=ax["b"],
                hue=hue,
                hue_order=hue_order,
                showfliers=False,
            )
            sns.stripplot(
                data=df.groupby(hue).sample(20),
                y=col,
                x=hue,
                hue_order=hue_order,
                ax=ax["b"],
                color="black",
                size=4,
            )
        ax["b"].tick_params(axis="both", labelsize=15)
        ax["b"].set_title("no fliers", fontsize=20)
        ax["b"].set(xlabel=None)
        ax["b"].set(ylabel=None)

        df_nan = ((df[[col, hue]]).groupby(hue).agg(lambda x: x.isna().sum())).rename(
            columns={col: "NaN"}
        )
        df_zero = ((df[[col, hue]]).groupby(hue).agg(lambda x: (x == 0).sum())).rename(
            columns={col: "0"}
        )
        df_special = df_nan.join(df_zero)
        df_special = (
            df_special.replace(0, df_special.max().max() * (-0.1)) / df.shape[0]
        )
        df_special = pd.melt(
            df_special.reset_index(),
            id_vars=hue,
            value_vars=["NaN", "0"],
            var_name="Variable",
            value_name="Value",
        )
        sns.barplot(
            data=df_special,
            x="Variable",
            y="Value",
            hue=hue,
            hue_order=hue_order,
            ax=ax["c"],
            legend="brief",
            edgecolor="black",
        )
        ax["c"].axhline(0, color="black", ls="--")
        ax["c"].grid(True, axis="y")
        ax["c"].set_xlabel(None)
        ax["c"].set_ylabel(None)
        ax["c"].tick_params(axis="both", labelsize=15)

        fig.suptitle(f"{col} vs {hue}", fontsize=20)
        fig.set_size_inches(15, 5)
        fig.tight_layout()

    for col in cols_cat:
        if col == hue:
            continue
        fig, ax = plt.subplots()

        sns.countplot(
            data=df.fillna("<NaN>").replace("", "<EMPTY>"),
            x=col,
            hue=hue,
            hue_order=hue_order,
            legend="brief",
            stat="count",
            edgecolor="black",
        )
        ax.tick_params("x", rotation=90)
        ax.grid(True, axis="y")
        ax.set_title("straight hue", fontsize=15)
        ax.set_xlabel(None)
        ax.set_ylabel("Count", fontsize=15)
        ax.tick_params(axis="both", labelsize=15)

        fig.suptitle(f"{col} vs {hue}", y=1.07, fontsize=15)
        fig.set_size_inches(9, 3)

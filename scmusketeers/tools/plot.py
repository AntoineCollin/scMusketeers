import functools
import os
import random

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .utils import ann_subset, densify, dict_to_df

WD_PATH = "/data/analysis/data_collin/Discovair/"
DATA_PATH = WD_PATH + "data/"


colors = [
    "lightgray",
    "#FFF7EC",
    "#FEE8C8",
    "#FDD49E",
    "#FDBB84",
    "#FC8D59",
    "#EF6548",
    "#D7301F",
    "#B30000",
    "#7F0000",
]

cmap = mcolors.LinearSegmentedColormap.from_list("my_palette", colors)
sc.pl.umap = functools.partial(
    sc.pl.umap,
    size=5,
    legend_fontsize="xx-small",
    legend_loc="on data",
    color_map=cmap,
)  # setting default umap params


def quick_view(adata, groupby=None):
    print(adata)
    sc.set_figure_params()
    sc.pl.highest_expr_genes(adata, n_top=20)
    try:
        adata.var["mt"]
    except KeyError:
        adata.var["mt"] = adata.var_names.str.startswith("MT")

    try:
        adata.var["ribo"]
    except KeyError:
        adata.var["ribo"] = adata.var_names.str.startswith("RP")

    sc.set_figure_params(dpi=50)
    sc.pl.violin(
        adata=adata,
        keys=[
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt",
            "pct_counts_ribo",
        ],
        jitter=0.2,
        multi_panel=True,
    )
    sc.set_figure_params(dpi=70, figsize=(6, 4))
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
    sc.pl.scatter(
        adata, x="n_genes_by_counts", y="pct_counts_mt", color=groupby
    )
    sc.pl.umap(
        adata,
        color=[
            groupby,
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt",
            "pct_counts_ribo",
        ],
        legend_loc="on data",
    )
    if groupby:
        sc.pl.violin(
            adata, keys=["n_genes_by_counts"], groupby=groupby, rotation=90
        )
        sc.pl.violin(
            adata, keys=["total_counts"], groupby=groupby, rotation=90
        )
        sc.pl.violin(
            adata, keys=["pct_counts_mt"], groupby=groupby, rotation=90
        )
        sc.pl.violin(
            adata, keys=["pct_counts_ribo"], groupby=groupby, rotation=90
        )


def plot_genes(adata, genes, subset=None):
    """
    Feature plot of the genes on the umap
    """
    if subset in adata.obs["celltype_lv0_V6"].unique():
        to_subset = adata.obs["celltype_lv0_V6"] == subset
    elif subset in adata.obs["celltype_lv1_V6"].unique():
        to_subset = adata.obs["celltype_lv1_V6"] == subset
    elif subset in adata.obs["celltype_lv2_V6"].unique():
        to_subset = adata.obs["celltype_lv2_V6"] == subset
    else:
        to_subset = adata.obs_names
    sc.set_figure_params(dpi=200, figsize=(5, 3))
    sc.pl.umap(adata[to_subset,], color=genes, use_raw=False, size=5)


def violin(adata, subset, gene, group_by, hue, figsize=(5, 3)):
    """
    subset : celltype to subset
    gene : gene to plot
    group_by : x from seaborn arg
    hue : hue from seaborn arg

    ex : violin('Epithelial', 'PRSS2', 'celltype_lv2_V3', 'phenotype', figsize = (20,3))
    """
    sc.set_figure_params(dpi=100, figsize=figsize)
    plt.figure()
    if subset in adata.obs["celltype_lv0_V3"].unique():
        to_subset = adata.obs["celltype_lv0_V3"] == subset
    elif subset in adata.obs["celltype_lv1_V3"].unique():
        to_subset = adata.obs["celltype_lv1_V3"] == subset
    elif subset in adata.obs["celltype_lv2_V3"].unique():
        to_subset = adata.obs["celltype_lv2_V3"] == subset
    else:
        to_subset = adata.obs_names
    to_plot = adata.obs.loc[to_subset, [group_by, hue]]
    to_plot[group_by].cat.remove_unused_categories(inplace=True)
    to_plot[gene] = pd.Series(
        np.array(adata[to_subset, gene].X.todense()).ravel(),
        index=adata[to_subset, :].obs_names,
    )
    sns.violinplot(x=group_by, y=gene, hue=hue, data=to_plot)
    plt.xticks(rotation=70)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_average_gene_list(adata, gene_list, ret_score=False, **kwargs):
    meta_gene = np.mean(adata[:, gene_list].X, axis=1)
    to_plot = sc.AnnData(
        X=meta_gene,
        obs=adata.obs,
        var=pd.DataFrame(index=["mean_expression"]),
        obsm=adata.obsm,
    )
    if ret_score:
        return to_plot
    sc.pl.umap(to_plot, color="mean_expression", use_raw=False, **kwargs)


#### UMAP ####


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def hex2rgb(hexcode):
    return np.array(ImageColor.getcolor(hexcode, "RGB"))


def continuous_palette(start, end, n):
    """
    Retourne une liste de n couleurs allant de start à end.

    Args:
    start: La première couleur de la palette.
    end: La dernière couleur de la palette.
    n: Le nombre de couleurs de la palette.

    Returns:
    Une liste de n couleurs.
    """
    start = hex2rgb(start)
    end = hex2rgb(end)
    step = (end - start) / (n - 1)
    colors = []
    for i in range(n):
        col = start + step * i
        colors.append(rgb2hex(*col))
    return colors


def random_shuffle_umap(adata, shuffle=True, **kwargs):
    """
    same as sc.pl.umap but cells appear in shuffle order
    """
    if shuffle:
        np.random.seed(0)
        random_indices = np.random.permutation(list(range(adata.shape[0])))
        ax = sc.pl.umap(adata[random_indices, :], show=False, **kwargs)
    else:
        ax = sc.pl.umap(adata, show=False, **kwargs)
    return ax


def add_space(n):
    return "{:,}".format(n).replace(",", " ")


def umap_with_ncells(adata, color, **kwargs):
    """
    adds number of cells in the legend
    """
    celltype_counts = adata.obs[color].value_counts()

    celltype_sizes = {
        celltype: celltype
        + " (n = "
        + str(add_space(celltype_counts[celltype]))
        + ")"
        for celltype in celltype_counts.index
    }

    adata.obs["celltype_size"] = adata.obs[color].map(celltype_sizes)
    adata.uns["celltype_size_colors"] = adata.uns[color + "_colors"]

    ax = random_shuffle_umap(adata, color="celltype_size", **kwargs)
    return ax


def umap_subset(adata, obs_key, subset, **kwargs):
    """
    Plot individual subset, the rest is grey
    """
    if type(subset) == str:
        subset = [subset]
    ax = sc.pl.umap(
        adata, color=obs_key, groups=["subset"], show=False, **kwargs
    )

    legend_texts = ax.get_legend().get_texts()

    for legend_text in legend_texts:
        if legend_text.get_text() == "NA":
            legend_text.set_text("other cell types")
    return ax


def feature_umap_subset(adata, obs_key, subset, gene, **kwargs):
    """
    Plot gene value individual subset, the rest is gery
    """

    ax = sc.pl.umap(adata, show=False, **kwargs)

    sc.pl.umap(
        ann_subset(adata, obs_key, subset), color=gene, ax=ax, s=dot_size
    )
    return ax


#### Stackbar, from sccoda #####

from typing import Collection, List, Optional, Tuple, Union

from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap


def stackbar(
    y: np.ndarray,
    type_names: List[str],
    title: str,
    level_names: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    dpi: Optional[int] = 100,
    cmap=None,
    plot_legend: Optional[bool] = True,
) -> plt.Subplot:
    """
    Plots a stacked barplot for one (discrete) covariate
    Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

    Parameters
    ----------
    y
        The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows, one for each group, containing the count
        mean of each cell type
    type_names
        The names of all cell types
    title
        Plot title, usually the covariate's name
    level_names
        names of the covariate's levels
    figsize
        figure size
    dpi
        dpi setting
    cmap
        The color map for the barplot
    plot_legend
        If True, adds a legend
    Returns
    -------
    Returns a plot

    ax
        a plot

    """
    n_bars, n_types = y.shape

    figsize = rcParams["figure.figsize"] if figsize is None else figsize

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    r = np.array(range(n_bars))
    sample_sums = np.sum(y, axis=1)

    barwidth = 0.85
    cum_bars = np.zeros(n_bars)

    for n in range(n_types):
        bars = [
            i / j * 100
            for i, j in zip([y[k][n] for k in range(n_bars)], sample_sums)
        ]
        # plt.bar(r, bars, bottom=cum_bars, width=barwidth, label=type_names[n], linewidth=0) color=cmap(n % cmap.N),
        plt.barh(
            r,
            height=barwidth,
            width=bars,
            left=cum_bars,
            label=type_names[n],
            linewidth=0,
            color=cmap[n],
        )
        cum_bars += bars

    ax.set_title(title)
    if plot_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    ax.set_yticks(r)
    ax.set_yticklabels(level_names, rotation=0)
    ax.set_xlabel("Proportion (%)")
    ax.spines["top"].set_visible(False)
    return ax


def plot_umap_jitter(adata, color, fontsize=10, **kwargs):

    def gen_mpl_labels(
        adata,
        groupby,
        exclude=(),
        ax=None,
        adjust_kwargs=None,
        text_kwargs=None,
    ):
        if adjust_kwargs is None:
            adjust_kwargs = {"text_from_points": False}
        if text_kwargs is None:
            text_kwargs = {}
        medians = {}

        for g, g_idx in adata.obs.groupby(groupby).groups.items():
            if g in exclude:
                continue
            medians[g] = np.median(adata[g_idx].obsm["X_umap"], axis=0)

        if ax is None:
            texts = [
                plt.text(x=x, y=y, s=k, **text_kwargs)
                for k, (x, y) in medians.items()
            ]
        else:
            texts = [
                ax.text(x=x, y=y, s=k, **text_kwargs)
                for k, (x, y) in medians.items()
            ]

        adjust_text(texts, **adjust_kwargs)

    ax = sc.pl.umap(
        adata,
        color=color,
        show=False,
        legend_loc=None,
        frameon=False,
        **kwargs,
    )
    gen_mpl_labels(
        adata,
        color,
        exclude=("None",),  # This was before we had the `nan` behaviour
        ax=ax,
        adjust_kwargs=dict(arrowprops=dict(arrowstyle="-", color="black")),
        text_kwargs=dict(fontsize=fontsize, weight="bold"),
    )

    # fig = ax.get_figure()
    # fig.tight_layout()
    return ax


def pbulk_heatmap(
    pbulk,
    markers,
    genes_ct_level,
    split_by,
    layer=None,
    gene_th=0,
    cells_th=0,
    title=None,
    cmap=cmap,
    figsize_x=None,
    figsize_y=None,
    z_score=None,
    standard_scale=None,
    col_split=None,
    **kwargs,
):
    """
    plots a heatmap of pbulk counts for the given markers. Markers should represent a condition of genes_ct_level.

    markers : a dict with celltypes as keys and genes as values
    genes_ct_level : celltypes represented by the markers
    split_by : list of annotations to show as bar above the heatmap
    gene_th,cells_th : filters on gene an cells expression
    """

    def get_color_dict(adata, color):
        if f"{color}_colors" in adata.uns:
            return dict(
                zip(
                    adata.obs[color].cat.categories,
                    adata.uns[f"{color}_colors"],
                )
            )
        else:
            return None

    colors = {}
    for obs in pbulk.obs.columns:
        if f"{obs}_colors" in pbulk.uns:
            colors[obs] = get_color_dict(pbulk, obs)

    g = []
    for i in markers:
        g += list(markers[i])

    gene_list = []
    for i in g:
        if i not in gene_list:
            gene_list.append(i)

    pbulk = pbulk[:, gene_list]

    if type(split_by) == str:
        split_by = [split_by]

    # Setting cells annotation
    annot_obs = pbulk.obs.loc[
        :, split_by + [genes_ct_level]
    ]  # .astype(str) # Selecting bar legends ['celltype_lv0_V6']
    annot_obs = annot_obs.sort_values(
        [genes_ct_level] + split_by
    )  # Sorting bar legend order + ['celltype_lv0_V6']
    # Setting genes annotation
    gene_df = dict_to_df(markers, val_name="genes", key_name=genes_ct_level)
    gene_ct_map = {
        v: k for v, k in zip(gene_df["genes"], gene_df[genes_ct_level])
    }
    if not layer:
        X = pbulk.X
    else:
        X = pbulk.layers[layer]

    pbulk.var["mean_expr"] = densify(X).mean(axis=0)
    annot_var = pd.DataFrame(pbulk.var["mean_expr"])
    annot_var["genes"] = annot_var.index
    annot_var[genes_ct_level] = annot_var["genes"].replace(gene_ct_map)
    annot_var[genes_ct_level] = pd.Categorical(
        annot_var[genes_ct_level],
        categories=pbulk.obs[genes_ct_level]
        .cat.remove_unused_categories()
        .cat.categories,
    )  # Sorting genes in the order of the celltypes
    # print(annot_var)
    # annot_var = annot_var.sort_values(genes_ct_level)

    # Defining heatmap dataframe
    expr_df = pd.DataFrame(
        densify(X), index=pbulk.obs_names, columns=pbulk.var_names
    )
    # expr_df = expr_df.loc[:,(expr_df>gene_th).any(axis = 0)]# # Removing unexpressed genes
    # expr_df = expr_df.loc[(expr_df>cells_th).any(axis = 1),:]#

    expr_df = expr_df.loc[
        annot_obs.index, annot_var["genes"]
    ]  # Ordering expr matrix to be coherent with annotation
    expr_df = expr_df.transpose()

    y_shape = 0.5 + annot_var.shape[0] * 8 / 45
    if not figsize_x:
        figsize_x = 15
    if not figsize_y:
        figsize_y = y_shape
    figsize = (figsize_x, figsize_y)
    plt.figure(figsize=figsize, dpi=100)  # annot_var.shape[0]*20/45

    if y_shape > 5:
        legend = True
    else:
        legend = False

    if col_split:
        col_split = annot_obs[col_split]
        col_split_order = list(col_split.cat.categories)
    else:
        col_split_order = None

    top_ann = pch.HeatmapAnnotation(  # celltype_lv0_V6 = pch.anno_simple(annot_obs['celltype_lv0_V6'].cat.remove_unused_categories(),add_text=False,height=4,legend=legend,
        # colors = {ct : c for ct, c in colors['celltype_lv0_V6'].items() if ct in annot_obs['celltype_lv0_V6'].cat.remove_unused_categories().cat.categories},
        # legend_kws={'fontsize':10, 'color_text': False}),
        **{
            split: pch.anno_simple(
                annot_obs[split],
                add_text=False,
                height=4,
                legend=legend,
                colors=colors[split],
                legend_kws={"fontsize": 10, "color_text": False},
            )
            for split in split_by[::-1]
        },
        # donor=pch.anno_simple(annot_obs[split_by],add_text=False,height=4,legend=legend, colors = colors[split_by],
        #                      legend_kws={'fontsize':10, 'color_text': False}),
        **{
            genes_ct_level: pch.anno_simple(
                annot_obs[genes_ct_level].cat.remove_unused_categories(),
                add_text=False,
                height=4,
                legend=True,
                colors={
                    ct: c
                    for ct, c in colors[genes_ct_level].items()
                    if ct
                    in annot_obs[genes_ct_level]
                    .cat.remove_unused_categories()
                    .cat.categories
                },
                legend_kws={"fontsize": 10, "color_text": False},
            )
        },
        legend=legend,
        legend_gap=5,
        hgap=0.5,
    )

    right_ann = pch.HeatmapAnnotation(
        axis=0,
        orientation="right",
        **{
            genes_ct_level: pch.anno_simple(
                annot_var[genes_ct_level],
                add_text=False,
                legend=False,
                colors=colors[genes_ct_level],
            )
        },
        legend=False,
        # Mean=pch.anno_barplot(annot_var['mean_expr'],legend=False,height=15,linewidth=0.1, cmap=cmap),
        # label=pch.anno_label(annot_var['mean_expr'].apply(lambda x:str(round(x,1))),colors="black",
        #                  height=1,relpos=(0,0.5)),
        verbose=0,
        label_side="top",
        label_kws={
            "horizontalalignment": "left",
            "rotation": 45,
            "visible": False,
            "fontsize": 3,
        },
    )

    # left_ann = pch.HeatmapAnnotation(**{ct_level:pch.anno_simple(annot_var[ct_level], add_text=False,legend=False,colors = colors[ct_level])},
    # legend=False)

    cm = pch.ClusterMapPlotter(
        data=expr_df,
        top_annotation=top_ann,
        right_annotation=right_ann,  # left_annotation= left_ann,
        # col_cluster=False,
        # row_cluster=False,
        # row_dendrogram=False,
        # col_split=df.AB,row_split=2,
        # col_split_gap=0.5,row_split_gap=0.8,
        label="Exp",
        show_rownames=True,
        show_colnames=False,
        row_names_side="left",
        tree_kws={"row_cmap": "Set1", "colors": "blue"},
        verbose=0,
        cmap=cmap,
        yticklabels=True,
        legend_width=100,
        z_score=z_score,
        col_split=col_split,
        col_split_order=col_split_order,
        standard_scale=standard_scale,
        yticklabels_kws={"labelsize": 5},
        xticklabels_kws={"labelrotation": -90, "labelcolor": "blue"},
        **kwargs,
    )
    cm.ax.set_title(title)
    plt.show()
    return cm


def get_scanpy_cmap():
    return [  # "#000000",  # remove the black, as often, we have black colored annotation
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#6A3A4C",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
        "#61615A",
        "#BA0900",
        "#6B7900",
        "#00C2A0",
        "#FFAA92",
        "#FF90C9",
        "#B903AA",
        "#D16100",
        "#DDEFFF",
        "#000035",
        "#7B4F4B",
        "#A1C299",
        "#300018",
        "#0AA6D8",
        "#013349",
        "#00846F",
        "#372101",
        "#FFB500",
        "#C2FFED",
        "#A079BF",
        "#CC0744",
        "#C0B9B2",
        "#C2FF99",
        "#001E09",
        "#00489C",
        "#6F0062",
        "#0CBD66",
        "#EEC3FF",
        "#456D75",
        "#B77B68",
        "#7A87A1",
        "#788D66",
        "#885578",
        "#FAD09F",
        "#FF8A9A",
        "#D157A0",
        "#BEC459",
        "#456648",
        "#0086ED",
        "#886F4C",
        "#34362D",
        "#B4A8BD",
        "#00A6AA",
        "#452C2C",
        "#636375",
        "#A3C8C9",
        "#FF913F",
        "#938A81",
        "#575329",
        "#00FECF",
        "#B05B6F",
        "#8CD0FF",
        "#3B9700",
        "#04F757",
        "#C8A1A1",
        "#1E6E00",
        "#7900D7",
        "#A77500",
        "#6367A9",
        "#A05837",
        "#6B002C",
        "#772600",
        "#D790FF",
        "#9B9700",
        "#549E79",
        "#FFF69F",
        "#201625",
        "#72418F",
        "#BC23FF",
        "#99ADC0",
        "#3A2465",
        "#922329",
        "#5B4534",
        "#FDE8DC",
        "#404E55",
        "#0089A3",
        "#CB7E98",
        "#A4E804",
        "#324E72",
    ]

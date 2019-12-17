'''Main operations'''
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gtfparse import read_gtf
from .util import *
import logging
import re, os
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s::%(message)s')

def read_matrix(filename, delimiter = '\t', rowname = 0, workdir = 'CNVmap_output'):
    '''Read the matrix file into formatted data type'''
    lines = iter_lines(open(filename, 'r'))
    col_to_skip = 0
    data = []
    rownames = []
    colnames = []
    for line in lines:
        if line.startswith('# '):
            # Skip the comment lines
            continue
        
        if delimiter is not None and delimiter not in line:
            raise ValueError(
                f'Delimiter "{delimiter}" not found in the first line.'
                )
        
        tokens = line.split(delimiter)
        
        if not is_float(tokens[-1]):
            # Encountering the row with cell names
            colnames = tokens
            continue

        if col_to_skip == 0:
            # Identify which colomns are not for matrix numeric contents.
            for i in range(len(tokens)):
                if is_float(tokens[i]):
                    break
            col_to_skip = i
            assert col_to_skip > rowname, "No gene identifiers found in the matrix file."
            colnames = colnames[col_to_skip:]

        rownames.append(tokens[rowname])
        data.append(tokens[col_to_skip:])

    data = np.array(data, dtype = float)
    genes = np.array(rownames, dtype = str)
    cells = np.array(colnames, dtype = str)

    adata = AnnData(data, 
                    obs = dict(obs_names = genes), 
                    var = dict(var_names = cells), 
                    ).T
    adata.X = data.T

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    adata.uns['workdir'] = workdir
    return adata

def add_cell_labels(adata, file_name):
    """Add cell lables.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L237
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    fig_path: `str`, optional (default: '')
        The file path of cell label file.
    fig_name: `str`, optional (default: None)
        The file name of cell label file. If file_name is not specified, 'unknown' is added as the label for all cells.
        
    Returns
    -------
    updates `adata` with the following fields.
    label: `pandas.core.series.Series` (`adata.obs['label']`,dtype `str`)
        Array of #observations that stores the label of each cell.
    """
    df_labels = pd.read_csv(file_name, sep = '\t', header = None, 
       index_col = None, names = ['label'], dtype = str, 
       compression = 'gzip' if file_name.split('.')[-1] == 'gz' else None)
    df_labels['label'] = df_labels['label'].str.replace('/','-')
    df_labels.index = adata.obs_names
    adata.obs['label'] = df_labels

def add_cell_colors(adata,file_path='',file_name=None):
    """Add cell colors.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L269
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    fig_path: `str`, optional (default: '')
        The file path of cell label color file.
    fig_name: `str`, optional (default: None)
        The file name of cell label color file. If file_name is not specified, random color are generated for each cell label.
        
    Returns
    -------
    updates `adata` with the following fields.
    label_color: `pandas.core.series.Series` (`adata.obs['label_color']`,dtype `str`)
        Array of #observations that stores the color of each cell (hex color code).
    label_color: `dict` (`adata.uns['label_color']`,dtype `str`)
        Array of #observations that stores the color of each cell (hex color code).        
    """
    labels_unique = adata.obs['label'].unique()
    
    df_colors = pd.read_csv(file_name, sep = '\t', header = None, 
        index_col = None, names = ['label', 'color'], dtype = str, 
        compression = 'gzip' if file_name.split('.')[-1]=='gz' else None)
    df_colors['label'] = df_colors['label'].str.replace('/', '-')   
    adata.uns['label_color'] = {df_colors.iloc[x, 0]: df_colors.iloc[x, 1] for x in range(df_colors.shape[0])}

    df_cell_colors = adata.obs.copy()
    df_cell_colors['label_color'] = ''
    for x in labels_unique:
        id_cells = np.where(adata.obs['label'] == x)[0]
        df_cell_colors.loc[df_cell_colors.index[id_cells], 'label_color'] = adata.uns['label_color'][x]
    adata.obs['label_color'] = df_cell_colors['label_color']

def log2Transformation(adata):
    logging.info('Log2 Transforming')
    adata.X = np.log2(adata.X+1)

def remove_mt_genes(adata):
    """remove mitochondrial genes.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L455
    """        
    r = re.compile("^MT-",flags=re.IGNORECASE)
    mt_genes = list(filter(r.match, adata.var_names))
    if(len(mt_genes)>0):
        logging.info('remove mitochondrial genes:')
        print(mt_genes)
        gene_subset = ~adata.var_names.isin(mt_genes)
        adata._inplace_subset_var(gene_subset)

def filter_genes(adata, min_num_cells = None, min_pct_cells = None, min_count = None, expr_cutoff = 1):
    '''
    from
    https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L331
    '''
    n_counts = np.sum(adata.X, axis = 0)
    adata.var['n_counts'] = n_counts
    n_cells = np.sum(adata.X > expr_cutoff, axis = 0)
    adata.var['n_cells'] = n_cells
    if(sum(list(map(lambda x: x is None,[min_num_cells,min_pct_cells,min_count])))==3):
        logging.info('No gene filtering')
    else:
        gene_subset = np.ones(len(adata.var_names),dtype=bool)
        if(min_num_cells!=None):
            logging.info('Filter genes based on min_num_cells')
            gene_subset = (n_cells>min_num_cells) & gene_subset
        if(min_pct_cells!=None):
            logging.info('Filter genes based on min_pct_cells')
            gene_subset = (n_cells>adata.shape[0]*min_pct_cells) & gene_subset
        if(min_count!=None):
            logging.info('Filter genes based on min_count')
            gene_subset = (n_counts>min_count) & gene_subset 
        adata._inplace_subset_var(gene_subset)
        logging.info('After filtering out low-expressed genes: {} cells, {} genes'.format(adata.shape[0], adata.shape[1]))


def order_genes_by_gtf(adata, GTF_file_name, ident = 'gene_name'):
    logging.info('Sorting with GTF annotation: {}'.format(GTF_file_name))
    if ident not in {'gene_id', 'gene_name'}:
        raise ValueError("Identifier must be set within {'gene_id', 'gene_name'}")
    ordered_idx = []
    chrlist = []
    with open(GTF_file_name, 'r') as gtfFile:
        for line in gtfFile:
            line = line.rstrip('\r\n')
            if line:
                if line.startswith('#'):
                    continue
                tokens = line.split('\t')
            else:
                break
            if tokens[2] != 'gene':
                continue
            geneIdent = None
            for info in tokens[8].split('; '):
                if info.split(' ')[0] == ident:
                    geneIdent = info.split(' ')[1].strip('"')
                    break
            if geneIdent != None:
                idx = np.where(adata.var_names == geneIdent)
                ordered_idx.extend(list(idx[0]))
                chrlist.extend([tokens[0]] * len(idx[0]))
                
    adata_tmp = adata.T[ordered_idx].copy()
    adata = adata_tmp.T
    adata.var['Chr'] = chrlist
    adata.var['chr'] = adata.var['Chr'].astype('category')
    del adata.var['Chr']
    return adata

def zscore_norm(adata, against = None, by = 'gene'):
    logging.info('Applying z-score normalization')
    input_data = adata.X.copy()
    if by == 'cell':
        axis = 1
    elif by == 'gene':
        axis = 0
    else:
        raise ValueError("'by' argument must be either 'cell' or 'gene'.")
    if against == None:
        Mean = np.mean(input_data, axis = axis)
        VAR = np.var(input_data, axis = axis)
    else:
        Mean = np.mean(against.X, axis = axis)
        VAR = np.var(against.X, axis = axis)

    Z = (input_data - Mean) / (VAR + 1)
    
    adata.uns['normalized'] = Z

def plot_CNVmap(adata, limit = (-5, 5), window = 150, n_cluster = 2, 
                fig_size = (12, 8), downsampleRate = None, save_fig = False, 
                fig_name = 'CNVmap.png'):
    logging.info('Plotting CNV map')
    # Process the expression profile
    input_data = adata.uns['normalized'].copy()
    set_limit(input_data, limit[0], limit[1])
    input_data -= np.mean(input_data)
    smoothen = np.zeros(input_data.shape)
    arm = int(round(window//2))
    for i in range(adata.n_vars):
        smoothen[:,i] = np.mean(input_data[:, max(0, i-arm):min(adata.n_vars, i+arm)], axis = 1)
    adata.uns['smoothened'] = smoothen
    
    # Hierarchical clustering
    Z = linkage(smoothen, 'ward')
    clusters = fcluster(Z, n_cluster, criterion = 'maxclust')
    # "clusters" contains the assigned cluster id for each sample
    sampleOrder = dendrogram(Z)['leaves']
    plt.savefig(os.path.join(adata.uns['workdir'], 'dendropgram.png'))
    # "sampleOrder" lists the original indices of the samples in the rearranged hierarchy. 
    clusteredX = smoothen[sampleOrder]
    devide_row = [sorted(clusters[sampleOrder]).index(i) for i in set(clusters)]
    
    # If too many gene, do a downsampling to be efficient.
    if downsampleRate != None:
        if downsampleRate < 1:
            subIdx = sorted(set(np.random.randint(adata.n_vars, size=int(adata.n_vars * downsampleRate))))
            clusteredX = clusteredX[:,subIdx]
            chrlist = adata.var['chr'].iloc[subIdx].tolist()
        else:
            subIdx = list(range(adata.n_vars))
            logging.warn('Why downsample with a frac >= one?')
            chrlist = adata.var['chr'].tolist()
    else:
        subIdx = list(range(adata.n_vars))
        chrlist = adata.var['chr'].tolist()
    devide_col = [chrlist.index(i) for i in set(chrlist)]
    
    vmax = 0.9 * (np.max(clusteredX) - np.min(clusteredX)) / 2
    vmin = - vmax
    
    # Start to plot
    figure = sns.clustermap(clusteredX, row_cluster = False, 
                            col_cluster = False, vmin = vmin, vmax = vmax, 
                            cmap = "RdBu_r", yticklabels = False, 
                            xticklabels = False, figsize = fig_size, 
                            robust = True) 
    ax = figure.ax_heatmap
    for i in devide_row:
        ax.plot([0, len(subIdx)], [i, i], '-k', linewidth = 0.8)
    for i in devide_col:
        ax.plot([i, i], [0, adata.n_obs], '-k', linewidth = 0.8)
    if save_fig:
        plt.savefig(os.path.join(adata.uns['workdir'], fig_name))
        logging.info('Plot saved to {}'.format(os.path.join(adata.uns['workdir'], fig_name)))
    else:
        plt.show()

# def draw_heatmap(expr_normalized, outputname, maxclust, genelist, celllist, palette):
#     print('...Starting to draw the clustered heatmap...')
#     expr_ave = pd.DataFrame()
#     expr_df = pd.DataFrame(expr_normalized, columns = celllist)
#     expr_df = set_limit(expr_df, -5, 5)
#     expr_df = expr_df - np.mean(expr_df)
#     chrlist = []
#     for i in range(75, len(expr_df)-75):	
#         expr_ave = pd.concat([expr_ave, pd.DataFrame(np.mean(expr_df[i - 75:i + 75])).T])
#         chrlist.append(genelist[i][1])
#     expr_ave_T = expr_ave.T
#     expr_clust = pd.DataFrame()
#     Z = linkage(expr_ave_T, 'ward')
#     clusters = fcluster(Z, maxclust, criterion = 'maxclust')
#     print('...Totally %s clusters.' % np.max(clusters))
#     clustorder = dendrogram(Z)['leaves']
#     last = clusters[clustorder[0]]
#     devidepos_r = []
#     for i in range(len(clustorder)):
#         expr_clust = pd.concat([expr_clust, expr_ave_T[clustorder[i]:clustorder[i] + 1]])
#         if clusters[clustorder[i]] != last:
#             devidepos_r.append(i)
#         last = clusters[clustorder[i]]
#     expr_clust_T = expr_clust.T
#     expr_small = pd.DataFrame()
#     chrlist2 = []
#     for i in range(0, len(expr_clust_T), 7):
#         expr_small = pd.concat([expr_small, expr_clust_T[i:i + 1]])
#         chrlist2.append(chrlist[i])
#     colors = []
#     rg = ['#516572', '#d8dcd6']
#     last = chrlist2[0]
#     change = 0
#     devidepos_c = []
#     for i in range(len(chrlist2)):
#         if chrlist2[i] != last:
#             change += 1
#             devidepos_c.append(i)
#         colors.append(rg[change % 2])    
#         last = chrlist2[i]
#     row_color = []
#     for i in expr_small.keys():
#         if palette == None:
#             row_color.append('w')
#             continue
#         elif i in palette.keys():
#             row_color.append(palette[i])
#         else:
#             row_color.append('w')
#     expr_draw = expr_small.T
#     maxofall = max(expr_draw.max())
#     minofall = min(expr_draw.min())
#     vvmax = 0.9 * (maxofall - minofall) / 2
#     vvmin = - vvmax
#     print('...Drawing')
#     figheight = round(len(expr_draw) / 2.4, 2)
#     sns.set(font_scale = 2.5)
#     if len(expr_small.keys()) >= 100:
#         figure = sns.clustermap(expr_draw, 
#             row_cluster = False, 
#             col_cluster = False, 
#             row_colors = row_color, 
#             col_colors = colors, 
#             vmin = vvmin,
#             vmax = vvmax,
#             cmap = "RdBu_r", 
#             yticklabels = False, 
#             xticklabels = False, 
#             figsize = (80, 40), 
#             robust = True
#             ) 
#     else:
#         figure = sns.clustermap(expr_draw, 
#             row_cluster = False, 
#             col_cluster = False, 
#             row_colors = row_color, 
#             col_colors = colors, 
#             vmin = vvmin,
#             vmax = vvmax,
#             cmap = "RdBu_r", 
#             yticklabels = True, 
#             xticklabels = False, 
#             figsize = (80, figheight), 
#             robust = True
#             ) 
#     sample_color = {'Tumor': '#C65911', 'Normal': '#FCE4D6'}
#     ax = figure.ax_heatmap
#     for i in devidepos_r:
#         ax.plot([0, len(chrlist2)], [i, i], '-k', linewidth = 1.5)
#     for i in devidepos_c:
#         ax.plot([i, i], [0, len(expr_draw)], '-k', linewidth = 1.5)
# ###This part is special for labeling all samples of all patients
# #    for lab in list(sample_color.keys()):
# #        figure.ax_col_dendrogram.bar(0, 0, color = sample_color[lab], 
# #                                     label = lab, 
# #                                     linewidth = 1)
# #    figure.ax_col_dendrogram.legend(loc = "center", ncol = 2)
#     figure.cax.set_position([.07, .2, .03, .45])
#     plt.savefig(outputname)
#     print('...A figure is saved to "%s" at your working directory.' % outputname)




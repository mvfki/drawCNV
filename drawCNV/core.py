'''Main operations'''
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from gtfparse import read_gtf
from .util import *
import logging
import re, os, sys
from copy import deepcopy

def read_matrix(file_name, delimiter = '\t', rowname = 0, 
                name = 'inputMatrix', workdir = 'CNVmap_output'):
    '''
    Read the matrix file into standard AnnData object.
    Arguments:
    ----------
    file_name - str, path like. 
                File name including relative or absolute path.
    delimiter - str. Default "\\t"
                The delimiter for the matrix table file. 
    rowname   - int. Default 0.
                In which column the feature identifiers are specified. 
                0-based index.
    workdir   - str, path like. Default "CNVmap_output".
                Where the output would be saved to.
    Returns:
    ----------
    adata - AnnData object: 
            - Numeric matrix saved in adata.X
            - Sample attributes saved in ad adata.obs
            - Feature (gene) attributes saved in adata.var
            - Other information saved in adata.uns
    See anndata.AnnData.
    '''
    lines = iter_lines(open(file_name, 'r'))
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
    adata.uns['normalized'] = False
    adata.uns['name'] = name
    return adata

def add_cell_labels(adata, file_name):
    """
    Add cell lables.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L237
    Arguments:
    ----------
    adata    - AnnData object.
               Annotated data matrix.
    fig_name - str, path like.
               The file name of cell label file. File content contains one 
               label string per line which corresponds to a sample in the 
               matrix in the same order.
    Returns:
    ----------
    updates `adata` with the following fields.
    label - pandas.core.series.Series, at adata.obs['label']
            Array of #observations that stores the label of each cell.
    """
    df_labels = pd.read_csv(file_name, sep = '\t', header = None, 
       index_col = None, names = ['label'], dtype = str, 
       compression = 'gzip' if file_name.split('.')[-1] == 'gz' else None)
    df_labels['label'] = df_labels['label'].str.replace('/','-')
    df_labels.index = adata.obs_names
    adata.obs['label'] = df_labels

def add_cell_colors(adata, file_name):
    """
    Add cell colors.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L269
    Arguments:
    ----------
    adata    - AnnData object.
               Annotated data matrix.
    fig_name - str, path lke. 
               The file name of cell label color file. File content contains 
               two columns delimited with <tab>. The first column comes with 
               unique labels according to the label file, and the second 
               column comes with a color code.
    Returns:
    ----------
    updates `adata` with the following fields.
    label_color - pandas.core.series.Series, at adata.obs['label_color']
                  Array of #observations that stores the color of each cell 
                  (hex color code).
    label_color - dict, at adata.uns['label_color']
                  Array of #observations that stores the color of each cell 
                  (hex color code).        
    """
    labels_unique = adata.obs['label'].unique()
    
    df_colors = pd.read_csv(file_name, sep = '\t', header = None, 
        index_col = None, names = ['label', 'color'], dtype = str, 
        compression = 'gzip' if file_name.split('.')[-1]=='gz' else None)
    df_colors['label'] = df_colors['label'].str.replace('/', '-')   
    adata.uns['label_color'] = {df_colors.iloc[x, 0]: df_colors.iloc[x, 1] \
                                for x in range(df_colors.shape[0])}

    df_cell_colors = adata.obs.copy()
    df_cell_colors['label_color'] = ''
    for x in labels_unique:
        id_cells = np.where(adata.obs['label'] == x)[0]
        df_cell_colors.loc[df_cell_colors.index[id_cells], 'label_color'] = adata.uns['label_color'][x]
    adata.obs['label_color'] = df_cell_colors['label_color']

def drop_cells_from_list(adata, droplist):
    '''Inplace remove cells from given list.
    Arguments:
    ----------
    adata - AnnData object.
            Annotated data matrix. 
    droplist - iterables
               An array with cell identifiers as elements.
    Returns:
    ----------
    updates `adata` with a subset of cells that are not in the droplist. 
    '''
    droplist = set(droplist)
    if len(droplist) == 0:
        return
    remainingIdx = []
    dropped = 0
    droppedCells = []
    for i in range(adata.obs_names.size):
        if adata.obs_names[i] not in droplist:
            remainingIdx.append(i)
        else:
            dropped += 1
            droppedCells.append(adata.obs_names[i])
    adata._inplace_subset_obs(remainingIdx)
    adata.uns['removedCells'] = droppedCells

def log2Transformation(adata):
    '''
    Log2(N + 1) transformation on the array data.
    Arguments:
    ----------
    adata - AnnData object.
            Annotated data matrix.
    Returns:
    ----------
    updates `adata` with the following fields.
    X - numpy.ndarray, at adata.X
    The transformed data matrix will replace the original array. 
    '''
    adata.X = np.log2(adata.X+1)

def remove_mt_genes(adata):
    """remove mitochondrial genes.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L455
    Arguments:
    ----------
    adata - AnnData object.
            Annotated data matrix.
    Returns:
    ----------
    updates `adata` with a subset of genes that excluded mitochondrial genes. 
    """        
    r = re.compile("^MT-", flags = re.IGNORECASE)
    mt_genes = list(filter(r.match, adata.var_names))
    if len(mt_genes) > 0:
        logging.info('remove mitochondrial genes:')
        sys.stdout.write(str(mt_genes))
        gene_subset = ~adata.var_names.isin(mt_genes)
        adata._inplace_subset_var(gene_subset)

def filter_genes(adata, min_num_cells = None, min_pct_cells = None, 
                 min_count = None, expr_cutoff = 1):
    '''
    Filter out genes based on different metrics.
    from https://github.com/pinellolab/STREAM/blob/master/stream/core.py#L331
    Arguments:
    ----------
    adata         - AnnData object.
                    Annotated data matrix.
    min_num_cells - int, default None. 
                    Minimum number of cells expressing one gene
    min_pct_cells - float, default None. 
                    Minimum percentage of cells expressing one gene
    min_count     - int, default None. 
                    Minimum number of read count for one gene
    expr_cutoff   - float, default 1. 
                    Expression cutoff. If greater than expr_cutoff, the gene 
                    is considered 'expressed'. 
    Returns:
    ----------
    updates `adata` with a subset of genes that pass the filtering. 
    '''
    n_counts = np.sum(adata.X, axis = 0)
    adata.var['n_counts'] = n_counts
    n_cells = np.sum(adata.X > expr_cutoff, axis = 0)
    adata.var['n_cells'] = n_cells
    if sum(list(map(lambda x: x is None,[min_num_cells,min_pct_cells,min_count]))) == 3:
        logging.info('No gene filtering')
    else:
        gene_subset = np.ones(len(adata.var_names), dtype = bool)
        if min_num_cells != None:
            logging.info('Filter genes based on min_num_cells')
            gene_subset = (n_cells > min_num_cells) & gene_subset
        if min_pct_cells != None:
            logging.info('Filter genes based on min_pct_cells')
            gene_subset = (n_cells > adata.shape[0] * min_pct_cells) & gene_subset
        if min_count != None:
            logging.info('Filter genes based on min_count')
            gene_subset = (n_counts > min_count) & gene_subset 
        adata._inplace_subset_var(gene_subset)
        logging.info('After filtering out low-expressed genes: {} cells, {} genes'.format(adata.shape[0], adata.shape[1]))

def order_genes_by_gtf(adata, GTF_file_name, ident = 'gene_name'):
    '''
    Place the gene in the order that follows the annotations from a GTF file. 
    Arguments:
    ----------
    adata         - AnnData object.
                    Annotated data matrix.
    GTF_file_name - str, path like.
                    The file name of the GTF file.
    ident         - str, default "gene_name"
                    The identifier type of the genes in the matrix. Choose 
                    based on the ninth column of the GTF file.
    Returns:
    ----------
    adata - AnnData object. 
            A new object where the order of genes updated. 
    '''
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

def zscore_norm(adata, against = None):
    '''
    Z-score normalization of the expression profile.
    Arguments:
    ----------
    adata   - AnnData object.
              Annotated data matrix.
    against - AnnData object, default None. 
              Another adata where a contol expression profile is saved in. 
              If None, normalization will be done against the adata itself. 
    Returns:
    ----------
    updates `adata` with the following fields.
    normalized - dict, with keys 'data' - numpy.ndarray, 'against' - str, 
                 at adata.uns['normalized'].
                 The normalized data matrix and against.uns['name'] 
    '''
    logging.info('Applying z-score normalization')
    input_data = adata.X.copy()

    if against == None:
        against = adata
        Mean = np.mean(input_data, axis = 0)
        VAR = np.var(input_data, axis = 0)
    else:
        Mean = np.mean(against.X, axis = 0)
        VAR = np.var(against.X, axis = 0)

    Z = (input_data - Mean) / (VAR + 1)
    adata.uns['normalized'] = {'data': Z, 'against': against.uns['name']}

def plot_CNVmap(adata, limit = (-5, 5), window = 150, n_cluster = 2, 
                downsampleRate = 0.2, fig_size = (12, 8), 
                fig_legend_ncol = 4, save_fig = False, 
                fig_name = 'CNVmap.png'):
    '''Plot the CNV map in a heatmap style with the normalized data matrix. 
    Arguments:
    ----------
    adata           - AnnData object. 
                      Annotated data matrix.
    limit           - tuple with 2 elements, default (-5, 5)
                      Limit the expression value to be plot within this 
                      interval. 
    window          - int, default 150. 
                      Window average of the expression levels of the 
                      neighboring N genes will be calculated for each gene, 
                      in order to smoothen the fluctuations and avoid high 
                      expression levels that might not come from high CNVs. 
    n_cluster       - int, default 2. 
                      Number of clusters to cut the hierarchical clustering 
                      result. 
    downsampleRate  - float that > 0 and < 1, default 0.2. 
                      Proportion of total genes to pick from. Smaller value 
                      increases the speed and decreases the resolution. 
    fig_size        - tuple with 2 elements, default (12, 8)
                      The size of the output figure, in inches.
    fig_legend_ncol - int, default 4.
                      Number of columns to display the legends
    save_fig        - bool, default False. 
                      Whether to save the figure. If True, figure would be 
                      saved to "workdir"; else, the figure would show if 
                      possible. 
    fig_name        - str, path like, default "CNVmap.png"
                      Name of the figure file. 
    Returns:
    ----------
    updates `adata` with the following fields.
    smoothen - numpy.ndarray, at adata.uns['smoothen']
               The smoothened expression matrix. 
    normalized - dict, with keys 'data' - numpy.ndarray, 'against' - str, 
                 at adata.uns['normalized']
    cluster - pandas.core.series.Series, at adata.obs['cluster']
              int of cluster id, 1-based.
    '''
    # Process the expression profile
    if not adata.uns['normalized']:
        logging.warn('Given matrix is not normalized, normalizing against itself. ')
        zscore_norm(adata)

    input_data = adata.uns['normalized']['data'].copy()
    set_limit(input_data, limit[0], limit[1])
    input_data -= np.mean(input_data)
    smoothen = np.zeros(input_data.shape)
    arm = int(round(window//2))
    for i in range(adata.n_vars):
        smoothen[:,i] = np.mean(input_data[:, max(0, i-arm): min(adata.n_vars, i+arm)], 
                                axis = 1)
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
    #TODO: Handle the case where only one type of cells. 
    cell_color = adata.obs['label_color'].iloc[sampleOrder].tolist()
    cl_palette = sns.color_palette("husl", len(set(clusters)))
    cluster_color = [cl_palette[i - 1] for i in clusters[sampleOrder]]
    row_col_df = pd.DataFrame({'sample_label': cell_color, 'cluster': cluster_color})
    cell_list_patches = []
    for x in adata.uns['label_color'].keys():
        cell_list_patches.append(Patches.Patch(color = adata.uns['label_color'][x],label=x))
    cl_list_patches = []
    for i in range(len(set(clusters))):
        cl_list_patches.append(Patches.Patch(color = cl_palette[i], label = "Cluster%d"%(i+1)))
    adata.obs['cluster'] = clusters
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')

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
    
    chr_color = []
    rg=['#516572', '#d8dcd6']
    last = chrlist[0]
    change = 0
    for i in range(len(chrlist)):
        if chrlist[i] != last:
            change += 1
        chr_color.append(rg[change % 2])    
        last = chrlist[i]

    # Start to plot
    vmax = 0.9 * (np.max(clusteredX) - np.min(clusteredX)) / 2
    vmin = - vmax
    figure = sns.clustermap(clusteredX, row_cluster = False, 
                            row_colors = [cluster_color, cell_color], 
                            col_colors = chr_color, col_cluster = False, 
                            vmin = vmin, vmax = vmax, cmap = "RdBu_r", 
                            yticklabels = False, xticklabels = False, 
                            figsize = fig_size, robust = True)
    ax = figure.ax_heatmap
    # Add line devider for clusters and chromosomes
    for i in devide_row:
        ax.plot([0, len(subIdx)], [i, i], '-k', linewidth = 0.8)
    for i in devide_col:
        ax.plot([i, i], [0, adata.n_obs], '-k', linewidth = 0.8)
    # Add cell label legend
    fig_legend_ncol = min(8, fig_legend_ncol)
    ax.legend(handles = cell_list_patches, loc = 'center', 
              bbox_to_anchor = (0.5, 1.15), ncol = fig_legend_ncol, 
              fancybox = True, shadow = False, markerscale = 2.5)
    # Add cluster label legend
    nrow_legend1 = np.ceil(len(cell_list_patches) / fig_legend_ncol)
    legend2_anchor_height = 1.15 + (nrow_legend1) * 0.04 + 0.035
    ax1 = ax.twinx()
    ax1.legend(handles = cl_list_patches, loc = 'center', 
               bbox_to_anchor = (0.5, legend2_anchor_height), 
               ncol = fig_legend_ncol, fancybox = True, shadow = False, 
               markerscale = 2.5)
    figure.cax.set_position([.18, .2, .03, .45])

    if save_fig:
        plt.savefig(os.path.join(adata.uns['workdir'], fig_name))
    else:
        plt.show()

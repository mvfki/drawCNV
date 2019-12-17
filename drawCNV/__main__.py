'''Includes the main pipeline to draw the CNV plot'''
import argparse
from .core import *
from .util import *
import logging
import warnings

warnings.simplefilter('ignore')

def getargs():
    parser = argparse.ArgumentParser(description = 'Developed by Yichen Wang, Zhang Lab, SLST, ShanghaiTech University.')
    parser.add_argument('-m', '--matrix', required = True, help = 'The matrix file containing the TPM values. See example.')
    parser.add_argument('-n', '--normal', required = True, help = 'The expression profile matrix file of normal samples (of the same tissue). This should be in the same form of the input matrix, and this is required.')
    parser.add_argument('-o', '--output', required = True, help = 'The file name of the output image file.')
    parser.add_argument('-g', '--gtf', required = True, help = 'The GTF annotation file for ordering, if your gene IDs in the matrix are corresponding to the Ensembl GTF file and your matrix is not ordered.')
    parser.add_argument('--geneIdent', default = 'gene_name', help = 'The GTF gene identifier type that corresponds to the first column of the matrix. Choose from {\'gene_id\', \'gene_name\'}')
    parser.add_argument('-p', '--palette', required = False, help = 'A text file indicating which color to label for each sample. See example.')
    parser.add_argument('-c', '--cluster', default = 2, type = int, help = 'The specified number [INT] of clusters. default: 2')
    #parser.add_argument('-L', '--label', required = False, help = 'A text file with a list of sample names that you assume to be normal samples. We will add a label to the heatmap according to this.')
    parser.add_argument('-d', '--drop', required = False, help = 'A text file with a list of sample names that you want to remove. See example.')
    parser.add_argument('-t', '--transform', action = 'store_true', help = 'Do the log2(TPM+1) transformation to your input matrix. The transformation will be done to the normal matrix automatically regardless of this argument. Both of the matrices have to be done this transformation once before we start calculating.')
    parser.add_argument('--downsample', default = 0.5, type = float, help = 'The downsample factor on genes for plotting CNV map.')
    args = parser.parse_args()
    return args

def main():
    # Basic configuration
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s::%(message)s')
    args = getargs()

    # Preprocess the matrix of interest
    logging.info('Reading the expression profile')
    adata = read_matrix(args.matrix, workdir = args.output)
    if args.drop != None:
        drop_cells_from_list(adata, open(args.drop, 'r').read().splitlines())
    if args.transform:
        log2Transformation(adata)
    filter_genes(adata, min_pct_cells = 0.02, expr_cutoff = 1)
    adata = order_genes_by_gtf(adata, args.gtf, ident = args.geneIdent)

    # Processing the normal sample control
    logging.info("Reading the normal control sample profile")
    adata_normal = read_matrix(args.normal, workdir = args.output)
    adata_normal = keep_genes_as_list(adata_normal, adata.var_names)
    if args.transform:
        log2Transformation(adata_normal)

    # Normalization
    zscore_norm(adata, against = adata_normal, by = 'gene')

    plot_CNVmap(adata, n_cluster = args.cluster, downsampleRate = args.downsample, save_fig = True)
    # if args.palette != None:
    #     palette_raw = open(args.palette, 'r').read().split('\n')
    #     palette = {}
    #     for i in palette_raw:
    #         if i == '':
    #             continue
    #         info = i.split()
    #         palette[info[0]] = info[1]
    # else:
    #     palette = args.palette
    # draw_heatmap(adata.uns['normalized'].T, args.output, args.cluster, adata.obs_names.tolist(), adata.var_names.tolist(), palette)

if __name__ == '__main__':
    main()
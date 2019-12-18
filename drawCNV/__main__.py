'''Includes the main pipeline to draw the CNV plot'''
import argparse
from .core import *
from .util import *
import logging
import warnings
import sys, os

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

def getargs():
    parser = argparse.ArgumentParser(description = 
        'Developed by Yichen Wang, Zhang Lab, SLST, ShanghaiTech University.')

    input_matrix_prop = parser.add_argument_group('Input matrix settings')
    input_matrix_prop.add_argument('-m', '--matrix', required = True, metavar = 'FILE', 
        help = 'The matrix file containing the TPM values. Required. ')
    input_matrix_prop.add_argument('--rowname-matrix', type = int, default = 0, metavar = 'INT', 
        help = 'In which column the gene identifiers are placed in the matrix file, 0-based. Default: 0')
    input_matrix_prop.add_argument('-l', '--label', required = True, metavar = 'FILE', 
        help = 'A text file with each row a label that corresponds to each cell in the matrix. Required. ')
    input_matrix_prop.add_argument('-c', '--label-color', required = True, metavar = 'FILE',
        help = 'A text file with the first column containing unique labels and the second column specifies corresponding color labels. Required. ')
    input_matrix_prop.add_argument('-d', '--drop', metavar = 'FILE', 
        help = 'A text file with a list of sample names that you want to remove.')
    input_matrix_prop.add_argument('-g', '--gtf', metavar = 'FILE', 
        help = 'The GTF annotation file for ordering, if your gene identifier in the matrix can be found in GTF file and your matrix is not ordered.')
    input_matrix_prop.add_argument('--geneIdent', default = 'gene_name', metavar = 'STR', 
        help = 'The GTF gene identifier type that corresponds to the first column of the matrix. Choose from {\'gene_id\', \'gene_name\'}, default "gene_name".')

    normal_matrix_prop = parser.add_argument_group("Normal control matrix settings")
    normal_matrix_prop.add_argument('-n', '--normal', required = True, metavar = 'FILE',
        help = 'The expression profile matrix file of normal control samples (of the same tissue). Required. ')
    normal_matrix_prop.add_argument('--rowname-normal', type = int, default = 0, metavar = 'INT', 
        help = 'In which column the gene identifiers are placed in the normal control matrix file, 0-based. Default: 0')

    calc_prop = parser.add_argument_group("Calculation settings")
    calc_prop.add_argument('-t', '--transform', action = 'store_true', 
        help = 'Do the log2(TPM+1) transformation to your input matrix. ')
    calc_prop.add_argument('-C', '--cluster', default = 2, type = int, metavar = 'INT', 
        help = 'The specified number [INT] of clusters. Default: 2')
    calc_prop.add_argument('--downsample', default = 0.2, type = float, metavar = 'FLOAT', 
        help = 'The downsample factor on genes for plotting CNV map. Default: 0.2')

    # Others
    parser.add_argument('-o', '--output', default = "CNVmap_output", metavar = 'PATH', 
        help = 'The path of the output image files. default "CNVmap_output". ')
    args = parser.parse_args()
    return args

def main():
    # Basic configuration
    args = getargs()

    # Preprocess the matrix of interest
    logging.info('Reading the expression profile: {}'.format(args.matrix))
    adata = read_matrix(args.matrix, 
                        rowname = args.rowname_matrix, 
                        workdir = args.output)
    logging.info('Adding cell labels from: {}'.format(args.label))
    add_cell_labels(adata, args.label)
    logging.info('Adding label colors from: {}'.format(args.label_color))
    add_cell_colors(adata, args.label_color)
    logging.info('{} cells x {} genes'.format(adata.n_obs, adata.n_vars))
    if args.drop != None:
        logging.info('Dropping cells from list: %s'%args.drop)
        drop_cells_from_list(adata, open(args.drop, 'r').read().splitlines())
        logging.info('Forced to remove {} cells from given list. {} left'.format(len(adata.uns['removedCells']), adata.n_obs))
    if args.transform:
        logging.info('Log2 Transforming on input matrix')
        log2Transformation(adata)
    #TODO: Allow filtration parameter tobe twisted from command line
    filter_genes(adata, min_pct_cells = 0.02, expr_cutoff = 1)
    logging.info('Reordering genes with GTF annotation: {}'.format(args.gtf))
    adata = order_genes_by_gtf(adata, args.gtf, ident = args.geneIdent)

    # Processing the normal sample control
    logging.info("Reading the normal control sample profile: {}".format(args.normal))
    adata_normal = read_matrix(args.normal, 
                               rowname = args.rowname_normal, 
                               name = 'normalControl', 
                               workdir = args.output)
    adata_normal = keep_genes_as_list(adata_normal, adata.var_names)
    if args.transform:
        logging.info('Log2 Transforming on normal control samples')
        log2Transformation(adata_normal)

    # Normalize and plot
    logging.info('Normalizing the input matrix against the normal control.')
    zscore_norm(adata, against = adata_normal, by = 'gene')
    logging.info('Plotting CNVmap')
    plot_CNVmap(adata, 
                n_cluster = args.cluster, 
                downsampleRate = args.downsample, 
                save_fig = True)

if __name__ == '__main__':
    main()
'''Minor and simple functions'''
import logging
import numpy as np

def iter_lines(file_like):
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file_like:
        line = line.rstrip('\r\n')
        if line:
            yield line

def is_float(string):
    """Check whether string is float."""
    try:
        float(string)
        return True
    except ValueError:
        return False

def keep_genes_as_list(adata, geneList):
    ordered_idx = []
    notFound = 0
    for gene in geneList:
        idx = np.where(adata.var_names == gene)
        try:
            ordered_idx.append(idx[0][0])
        except IndexError:
            notFound += 1
    adata_tmp = adata.T[ordered_idx].copy()
    logging.info('Forced to keep {} genes from given list'.format(len(geneList)))
    return adata_tmp.T

def set_limit(matrix, minlimit, maxlimit):
    if minlimit >= maxlimit:
        raise ValueError('Minimsum limit should be smaller than maximum limit.')
    matrix[matrix < minlimit] = minlimit
    matrix[matrix > maxlimit] = maxlimit
    
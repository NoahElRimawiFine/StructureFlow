import scanpy as sc
adata = sc.read_h5ad('../data/Renge/hipsc.h5ad')
print(adata)
print(adata.X.shape)
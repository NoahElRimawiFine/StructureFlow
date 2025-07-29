# Scaling Experiment Results

## Performance Comparison: StructureFlow vs NGM-NODE

| N_dim | Sparsity | Method | AUROC | AUPRC | Runtime (s) |
|-------|------|--------|------------|---------------|----------------|
| **N=10** | 5% | StructureFlow | 0.98 ± 0.01 | 0.57 ± 0.05 | 51.50 |
| | | NGM-NODE | 0.76 ± 0.20 | 0.39 ± 0.33 | 317.60 |
| | 20% | StructureFlow | 0.88 ± 0.09 | 0.55 ± 0.07 | 57.90 |
| | | NGM-NODE | 0.77 ± 0.02 | 0.42 ± 0.07 | 330.80 |
| | 40% | StructureFlow | 0.83 ± 0.02 | 0.70 ± 0.06 | 57.50 |
| | | NGM-NODE | 0.76 ± 0.01 | 0.63 ± 0.04 | 335.70 |
| **N=25** | 5% | StructureFlow | 0.97 ± 0.01 | 0.54 ± 0.04 | 69.80 |
| | | NGM-NODE | 0.76 ± 0.06 | 0.22 ± 0.05 | 1064.30 |
| | 20% | StructureFlow | 0.86 ± 0.03 | 0.55 ± 0.02 | 84.60 |
| | | NGM-NODE | 0.66 ± 0.04 | 0.33 ± 0.04 | 1044.50 |
| | 40% | StructureFlow | 0.73 ± 0.01 | 0.63 ± 0.01 | 79.10 |
| | | NGM-NODE | 0.60 ± 0.03 | 0.48 ± 0.01 | 918.30 |
| **N=50** | 5% | StructureFlow | 0.95 ± 0.02 | 0.50 ± 0.02 | 113.90 |
| | | NGM-NODE | 0.62 ± 0.02 | 0.09 ± 0.02 | 3162.20 |
| | 20% | StructureFlow | 0.80 ± 0.01 | 0.52 ± 0.04 | 121.30 |
| | | NGM-NODE | 0.55 ± 0.01 | 0.23 ± 0.01 | 2564.10 |
| | 40% | StructureFlow | 0.64 ± 0.00 | 0.57 ± 0.00 | 120.40 |
| | | NGM-NODE | 0.54 ± 0.02 | 0.43 ± 0.02 | 1293.30 |
| **N=100** | 5% | StructureFlow | 0.97 ± 0.01 | 0.53 ± 0.00 | 184.30 |
| | | NGM-NODE | 0.53 ± 0.01 | 0.06 ± 0.00 | 10280.60 |
| | 20% | StructureFlow | 0.75 ± 0.01 | 0.47 ± 0.00 | 233.20 |
| | | NGM-NODE | 0.52 ± 0.01 | 0.21 ± 0.00 | 3043.70 |
| | 40% | StructureFlow | 0.62 ± 0.00 | 0.53 ± 0.01 | 233.50 |
| | | NGM-NODE | 0.51 ± 0.00 | 0.40 ± 0.01 | 2392.10 |
| **N=200** | 5% | StructureFlow | 0.95 ± 0.00 | 0.49 ± 0.01 | 457.10 |
| | | NGM-NODE | — | — | — |
| | 20% | StructureFlow | 0.69 ± 0.00 | 0.38 ± 0.00 | 652.30 |
| | | NGM-NODE | — | — | — |
| | 40% | StructureFlow | 0.57 ± 0.00 | 0.47 ± 0.00 | 624.90 |
| | | NGM-NODE | — | — | — |
| **N=500** | 5% | StructureFlow | 0.86 ± 0.00 | 0.35 ± 0.00 | 2454.70 |
| | | NGM-NODE | — | — | — |
| | 20% | StructureFlow | 0.53 ± 0.00 | 0.22 ± 0.00 | 5255.50 |
| | | NGM-NODE | — | — | — |
| | 40% | StructureFlow | 0.51 ± 0.00 | 0.41 ± 0.00 | 5365.30 |
| | | NGM-NODE | — | — | — |

## Key Findings

- **StructureFlow maintains superior performance** across all system sizes and sparsity levels
- **NGM-NODE fails to scale** beyond N=100 due to computational constraints
- **Training time advantage**: StructureFlow is **6-56× faster** than NGM-NODE across all testable scales
- **Performance gap widens** with system size: StructureFlow AUROC advantage increases from ~21% (N=10) to ~43% (N=100) for sparse networks 
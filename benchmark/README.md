# Paper reproduction inputs

`mean_ari_by_dataset.csv` and `stbsr_per_run.csv` are numerical exports of
Supplementary Data 1. The former has 20 rows and ten methods. The latter has
200 STBSR runs, including the ten recorded seeds for each dataset. Reported
ARI, NMI and other metrics use the annotated observations from that dataset.

The 151672 domain table records run 5 (seed 11059586, ARI 0.7150 on 3,888
annotated spots). The companion array and tables contain the measured marker
values, UMAP, diffusion ordering, PAGA and marker statistics used in Fig. 2.
The reported ten-run mean for this section is 0.6018.

The Stage 57 domain table records the displayed Fig. 4a partitions (4,410
cells), their coordinates and the measured marker values used in Fig. 4b.
The STBSR assignment is the archived encoder partition with ARI 0.6371.
The reconstruction benchmark used by Supplementary Data 1 has a mean ARI of
0.5970. The boundary and GO tables provide the values displayed in Fig. 4c and d.

Measured expression values here are log1p counts per 10,000. Full training
inputs come from the originating datasets described in the manuscript.

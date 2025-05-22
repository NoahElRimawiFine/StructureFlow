####################
# PACKAGES required:
# kSamples
# glmnet
# ppcor
# pracma
####################
library(kSamples)
library(glmnet)
library(ppcor)

# *** Data loading ***
uploading <- dget("uploading.R")
DATA <- uploading(filename = 'THP1 data/single_cell_kouno_data.csv')

# *** SINCERITIES ***

SINCERITITES <- dget("SINCERITIES functions/SINCERITIES.R")
SIGN <- 0
result <- SINCERITITES(DATA,distance=1,method = 1,noDIAG = 0,SIGN = SIGN)
adj_matrix <- result$adj_matrix

# *** SUBNETWORK EXTRACTION FOR OVERLAPPING TFs ***
subGENES <- as.character(read.csv('THP1 data/SUBNET2_tomaru.csv',header = FALSE)$V1)
idxSUBgenes <- match(subGENES,DATA$genes)
idxSUBgenes <- idxSUBgenes[!is.na(idxSUBgenes)]
DATA$numGENES <- length(idxSUBgenes)
DATA$genes <- DATA$genes[idxSUBgenes]
adj_matrix <- adj_matrix[idxSUBgenes,idxSUBgenes]
for (i in 1:DATA$num_time_points) {
  DATA$singleCELLdata[[i]] <- DATA$singleCELLdata[[i]][,idxSUBgenes]
}
DATA$totDATA <- DATA$totDATA[,idxSUBgenes]

# *** REFERENCE GRN from RNAi EXPERIMENTS (from Tomaru et al.) ***
tomaru2 <- read.csv('THP1 data/tomaru2.csv', header = FALSE)
type_regulation <- tomaru2[,2]
netINFO <- tomaru2[,-2]
adj_ref <- matrix(0,nrow = DATA$numGENES, ncol = DATA$numGENES)
for (i in 1:dim(netINFO)[1]) {
  idxGENEsource <- match(netINFO[i,1],DATA$genes)
  if(!is.na(idxGENEsource)){
    idxGENEtarget <- match(as.character(t(netINFO[i,-1])),DATA$genes)
    idxGENEtarget <- idxGENEtarget[!is.na(idxGENEtarget)]
    if(SIGN==1){
      adj_ref[idxGENEsource,idxGENEtarget] <- type_regulation[i]
    }else{
      adj_ref[idxGENEsource,idxGENEtarget] <- 1
    }
  }
}
tomaruGENES <- sort(unique(as.character(unlist(netINFO))))
NDidx <- match('ND',tomaruGENES)
if(!is.na(NDidx)){
  tomaruGENES <- tomaruGENES[-NDidx]
}

# Final ranked list
adj_matrix <- adj_matrix/max(adj_matrix)
final_ranked_predictions <- dget("SINCERITIES functions/final_ranked_predictions.R")
table <- final_ranked_predictions(adj_matrix,DATA$genes,SIGN=SIGN,fileNAME='prediction4THP1',saveFile = TRUE)

# AUROC (x=fpr/1-specifity; y=recall/sensitivity) and AUPR (x=recall y=precision)
# Auto-regulatory edges are removed for AUROC and AUPR evaluation since
# RNAi experiments would not allow the identification of such edges. 
adj_matrix[row(adj_matrix)==col(adj_matrix)] <- 0
library(pracma)
auc_from_ranks_TC_sign <- dget("SINCERITIES functions/auc_from_ranks_TC_sign.R")
AUCresult <- auc_from_ranks_TC_sign(adj_matrix,adj_ref,1000)
AUROC <- AUCresult$AUROC
AUPR <- AUCresult$AUPR
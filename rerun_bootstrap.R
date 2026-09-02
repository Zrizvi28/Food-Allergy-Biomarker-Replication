# rerun_bootstrap.R — run this from a terminal with: Rscript rerun_bootstrap.R
# Deliberately minimal: loads ONLY what this one job needs, nothing else.
# This step was done with AI assistance

library(minfi)
library(bumphunter)
library(doParallel)

registerDoParallel(cores=4)
cat("Registered", getDoParWorkers(), "parallel workers\n")

library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)

data_dir <- '/home/ethan-xiao/food-allergy-biomarkers/data'

ann <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
mVals_189148_shared <- readRDS(file.path(data_dir, 'mVals_189148_shared.rds'))
allergy_status <- readRDS(file.path(data_dir, 'allergy_status.rds'))
batch <- readRDS(file.path(data_dir, 'batch.rds'))
scan_dates <- readRDS(file.path(data_dir, 'scan_dates_189148.rds'))

status_189148 <- allergy_status[batch == 'GSE189148']
design_189148_bh <- model.matrix(~ status_189148 + factor(scan_dates))

# Rebuilding the ordered chr/pos/cluster objects exactly as cell 29 originally did
chr_b <- ann[rownames(mVals_189148_shared), 'chr']
pos_b <- ann[rownames(mVals_189148_shared), 'pos']
stopifnot(sum(is.na(chr_b)) == 0)

ord_b     <- order(chr_b, pos_b)
m_ord_b   <- mVals_189148_shared[ord_b, ]
chr_ord_b <- chr_b[ord_b]
pos_ord_b <- pos_b[ord_b]
cluster_b <- clusterMaker(chr_ord_b, pos_ord_b, maxGap = 500)

cat("Starting bootstrap run at:", format(Sys.time()), "\n")
t0 <- Sys.time()

set.seed(42)
bumps_b_bootstrap <- bumphunter(m_ord_b, design_189148_bh,
                                 chr = chr_ord_b, pos = pos_ord_b,
                                 cluster = cluster_b, coef = 2,
                                 cutoff = 0.15, B = 250, type = "M",
                                 nullMethod = "bootstrap")

cat("Finished at:", format(Sys.time()), "— took", format(Sys.time() - t0), "\n")
saveRDS(bumps_b_bootstrap, file.path(data_dir, 'bumphunter_189148_B250_bootstrap.rds'))
cat("Saved successfully.\n")
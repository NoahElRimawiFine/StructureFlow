args    <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]                # e.g. "./results/output_scode"
repnum  <- as.integer(args[2])    # number of SCODE replicates

# helper to load one A.csv as a numeric matrix
load_A <- function(i) {
  f <- file.path(out_dir, paste0("out_", i), "A.csv")
  # read as a data.frame of characters, then convert
  df <- read.csv(f, header = FALSE, stringsAsFactors = FALSE)
  # turn into a pure numeric matrix
  M  <- data.matrix(df)
  if (!is.numeric(M)) stop("Non-numeric data in ", f)
  return(M)
}

# load the first replicate
meanA <- load_A(1)

# accumulate the rest
for (i in seq(2, repnum)) {
  meanA <- meanA + load_A(i)
}

# average
meanA <- meanA / repnum

# write it out
out_file <- file.path(out_dir, "meanA.csv")
write.csv(
  meanA,
  file      = out_file,
  row.names = FALSE,
  col.names = FALSE,
  quote     = FALSE
)

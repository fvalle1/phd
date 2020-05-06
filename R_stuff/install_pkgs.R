if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", dependencies=TRUE, repos='http://cran.rstudio.com/')

BiocManager::install("edgeR")
BiocManager::install('DESeq')

install.packages('statmod', dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages("gplots", dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages("fastICA", dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages("svglite", dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages("stringi", dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages("clustree", dependencies=TRUE, repos='http://cran.rstudio.com/')

require(devtools)
install_version("foreign", version="0.8-55", dependencies=TRUE, repos='http://cran.rstudio.com/')

BiocManager::install("WGCNA")
BiocManager::install("GO.db")
BiocManager::install("AnnotationDbi")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("biomaRt")




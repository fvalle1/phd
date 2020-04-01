library(TCGAbiolinks)
subtypes <- PanCancerAtlas_subtypes()
DT::datatable(subtypes,
               filter = 'top',
               options = list(scrollX = TRUE, keys = TRUE, pageLength = 5),
               rownames = FALSE)
write.csv(subtypes, "subtypes.csv")

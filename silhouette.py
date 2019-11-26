import pandas as pd
from sklearn.metrics.cluster import silhouette_score, silhouette_samples
import findspark
findspark.init("/usr/local/spark/spark-2.4.4-bin-hadoop2.7")
import pyspark as spark
from pyspark.sql.functions import udf, col, rand
from pyspark.sql.types import StringType

class silhouette():
    def __init__(self):
        self.sc=spark.SparkContext()
        self.sql=spark.SQLContext(self.sc)
        self.read()

    def read(self):
        self.df = self.sql.read.option("delimiter", '\t').option("header", 'true').csv('GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct')
        self.df = self.df.withColumn('ensg', udf(lambda x: x[:15], StringType())(col('Name')))
        self.df.registerTempTable('gtex')
        genelist=pd.read_csv("https://stephenslab.github.io/count-clustering/project/utilities/gene_names_all_gtex.txt", header=None).values.ravel()
        self.df = self.df.filter(col("ensg").isin(list(genelist)))
        self.df_file = self.sql.createDataFrame(pd.read_csv("https://storage.googleapis.com/gtex_analysis_v7/annotations/GTEx_v7_Annotations_SampleAttributesDS.txt", sep='\t').loc[:,['SAMPID','SMTS', 'SMTSD']])

    def organize(self):
        pass

    def show_sites(self):
        return self.df_file.select(['SMID','SMTS']).distinct().show()

    def 


if __name__=='__main__':
    sil = silhouette()
    sil.show_sites()
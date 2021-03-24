import urllib.request, urllib.parse
import pandas as pd

# ########################################################################################################
#                                                   class library mirDIP_Http                            #
# ########################################################################################################
class mirDIP_Http:

    mapScore = {
    'Very High' : '0',
    'High' : '1',
    'Medium' : '2',
    'Low' : '3'
    }


    url = "http://ophid.utoronto.ca/mirDIP"
        
    map = {}    # results will be here

    def __init__(self):
        return

    
    # unidirectional on genes
    def unidirectionalSearchOnGenes(self, geneSymbols, minimumScore = "Very High"):
    
        self.sendPost(self.url + "/Http_U", geneSymbols, '', self.mapScore[minimumScore])
        return


    # unidirectional on microrna(s)
    def unidirectionalSearchOnMicroRNAs(self, microRNAs, minimumScore = "Very High"):
            
        self.sendPost(self.url + "/Http_U", '', microRNAs, self.mapScore[minimumScore])
        return


    # bidirectional        
    def bidirectionalSearch(self, geneSymbols, microRNAs, minimumScore, sources, occurrances):

        '''
            String url_b = url + "/Http_B";
            
            String parameters = 
                    "genesymbol=" + geneSymbols + 
                    "&" + "microrna=" + microRNAs +
                    "&" + "scoreClass=" + mapScore.get(minimumScore) + 
                    "&" + "dbOccurrences=" + occurrances +
                    "&" + "sources=" + sources;
            
            int responseCode = sendPost(url_b, parameters);
            return responseCode;
        '''

        self.sendPost(self.url + "/Http_B", geneSymbols, microRNAs, self.mapScore[minimumScore], sources, occurrances)
        return        


       
    # .. serve POST request
    def sendPost(self, url_, geneSymbols, microrna, minimumScore, sources = '', occurrances = '1'):

        params = {
        'genesymbol' : geneSymbols,
        'microrna' : microrna,
        'scoreClass' : minimumScore,
        'dbOccurrences' : occurrances,
        'sources' : sources}

        params = bytes( urllib.parse.urlencode( params ).encode() )
        response = ''

        try:
            handler = urllib.request.urlopen(url_, params)
        except Exception:
            traceback.print_exc()
        else:
            self.response = handler.read().decode('utf-8')
            ## print(self.response)
            self.makeMap()

        return


    def makeMap(self):
            
        ENTRY_DEL = 0x01
        KEY_DEL = 0x02
            
        arr = self.response.split(chr(ENTRY_DEL))
            
        for str in arr:
                
            arrKeyValue = str.split(chr(KEY_DEL));
            if len(arrKeyValue) > 1: 
                self.map[arrKeyValue[0]] = arrKeyValue[1]

        return

    def getGeneratedAt(self): 

        if "generated_at" in self.map: 
            return self.map["generated_at"]
        else:
            return ''


    def getGeneSymbols(self):
       
        if "gene_symbols" in self.map:
            return self.map["gene_symbols"]
        else:
            return ''


    def getMicroRNAs(self): 
        if "micro_rnas" in self.map: 
            return self.map["micro_rnas"]
        else:
            return ''

    def getMinimumScore(self): 
        if "minimum_score" in self.map: 
            return self.map["minimum_score"]
        else:
            return ''

    def getDatabaseOccurrences(self):
        if "dbOccurrences" in self.map: 
            return self.map["dbOccurrences"]
        else: 
            return ''

    def getSources(self):
        if "sources" in self.map: 
            return self.map["sources"]
        else:
            return '' 
        
    def getResulsSize(self):
        if "results_size" in self.map: 
            return self.map["results_size"]
        else: 
            return ''

    def getResults(self): 
            data=[row.split("\t") for row in self._results.split("\r\n")]
            return pd.DataFrame(data=data[1:-1], columns=data[0])

    @property
    def _results(self):
        if "results" in self.map: 
            return self.map["results"]
        else: 
            return ''





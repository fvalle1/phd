import networkx as nx
import os
from regulatory.RegulatoryNetwork import RegulatoryNetwork


class MirDip(RegulatoryNetwork):
    def __init__(self):
        cdir = os.path.dirname(os.path.realpath(__file__))
        self.g = nx.read_edgelist(f"{cdir}/mirdip/HUMAN.mirDIP_top90k.Translated.tsv")
        self.nodes = list(self.g.nodes())

        
    def get_neighborns(self, node:str)->list:
        """
        return neighborns names of node
        :param node: node name
        :param graph: network object
        """
        if node not in self.g.nodes:
            if node+"-5p" in self.g.nodes:
                node += "-5p" 
            elif node+"-3p" in self.g.nodes:
                node += "-3p"
            else:
                return []
        return list(self.g.neighbors(node))
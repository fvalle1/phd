import networkx as nx
import os
import pandas as pd
from regulatory.RegulatoryNetwork import RegulatoryNetwork


class Mixed(RegulatoryNetwork):
    def __init__(self):
        cdir = os.path.dirname(os.path.realpath(__file__))
        self.g = nx.read_gpickle(f"{cdir}/graph.pkl.gz")
        self.adjacency = pd.read_csv(f"{cdir}/graph_metadata.csv").set_index("id")
    
    def check_input_data(self, mirnas: list)->list:
        return self.adjacency[self.adjacency["name"].isin(mirnas)].index.values
    
    def get_name(self, id, df_adjiacency):
        try:
            return df_adjiacency.at[id,"name"]
        except:
            return None

    def get_neighborns(self, node)->list:
        """
        return neighborns names of node
        :param node: node name
        :param graph: network object
        """
        return list(filter(lambda x:x is not None, map(lambda node: self.get_name(node, self.adjacency), self.g.neighbors(node))))
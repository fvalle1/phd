import networkx as nx
import pandas as pd

def get_name(id, df_adjiacency):
    try:
        return df_adjiacency.at[id,"name"]
    except:
        return None

def get_neighborns(node, graph: nx.Graph, df_adjiacency: pd.DataFrame)->list:
    """
    return neighborns names of node
    :param node: node name
    :param graph: network object
    """
    return list(map(lambda node: get_name(node, df_adjiacency), graph.neighbors(node)))
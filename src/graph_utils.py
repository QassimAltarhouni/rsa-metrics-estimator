import numpy as np
import networkx as nx
import pandas as pd

def build_request_graph(req: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
        g.add_edge(int(s), int(d),
                   w_sum=g.get_edge_data(s, d, {}).get("w_sum", 0) + float(b))
    return g


def build_demand_matrix(req: pd.DataFrame):
    nodes = np.unique(np.concatenate([req["source"], req["destination"]]))
    idx = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)))
    for s, d, b in req.itertuples(index=False):
        M[idx[s], idx[d]] += b
    return M

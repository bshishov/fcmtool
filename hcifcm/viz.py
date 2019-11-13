from collections import defaultdict

from hcifcm.graph import Graph
from graphviz import Digraph

from hcifcm.utils import distinct


__all__ = ['draw_graph']


def _linebreak(s: str, width=20):
    res = ''
    line = 0
    for ss in s.split(' '):
        if line + len(ss) >= width:
            res += '\\n'
            line = 0
        res += ss + ' '
        line += len(ss)
    return res


def _weight(w: float):
    if w > 0.1:
        return '+'
    elif w < 0.1:
        return '-'
    return '0'


def draw_graph(graph: Graph,
               node_label='value',
               edge_label='value',
               name='Graph',
               filename='graph',
               group_by=None,
               view=False,
               fmt='png'):
    g = Digraph(name, filename=filename, format=fmt)

    if group_by is None:
        for key, node in graph.nodes.items():
            g.node(key, node.get(node_label))
    else:
        groups = defaultdict(dict)
        for key, node in graph.nodes.items():
            group_name = node.get(group_by)
            groups[group_name][key] = node

        for group_name, group in groups.items():
            with g.subgraph(name='cluster_' + group_name) as c:
                c.attr(label=group_name)
                for key, node in group.items():
                    c.node(key, node.get(node_label))

    for (edge_from, edge_to), edge in graph.edges.items():
        g.edge(edge_from, edge_to, label=edge.get(edge_label))

    g.attr(fontsize='20')
    g.attr(overlap='false')
    g.render(cleanup=True, view=view)

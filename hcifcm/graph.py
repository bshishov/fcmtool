import numpy as np


__all__ = ['Graph']


class Graph(object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._lookup = None

    def _set_node(self, key, **attributes):
        if key in self.nodes:
            self.nodes[key].update(attributes)
        else:
            self.nodes[key] = attributes

    def _set_edge(self, key_from, key_to, **attributes):
        if key_from not in self.nodes:
            raise ValueError(f'No node with key {key_from}')

        if key_to not in self.nodes:
            raise ValueError(f'No node with key {key_to}')

        key = (key_from, key_to)
        if key in self.edges:
            self.edges[key].update(attributes)
        else:
            self.edges[key] = attributes

    def node(self, key, **attributes):
        self._set_node(key, id=key, **attributes)

    def edge(self, key_from, key_to, **attributes):
        self._set_edge(key_from, key_to, **attributes)

    def get_node_attr(self, key, attr: str, default=None):
        node = self.nodes.get(key)
        return node.get(attr, default)

    def get_edge_attr(self, key_from, key_to, attr: str, default=None):
        edge = self.edges.get((key_from, key_to))
        return edge.get(attr, default)

    def set_node_attr(self, key, attr: str, value):
        node = self.nodes.get(key)
        node[attr] = value

    def set_edge_attr(self, key_from, key_to, attr: str, value):
        edge = self.edges.get((key_from, key_to))
        edge[attr] = value

    def delete_node(self, key):
        if key not in self.nodes:
            raise ValueError(f'No node with key {key}')
        del self.nodes[key]

        for edge_key in self.edges:
            key_from, key_to = edge_key
            if key_from == key:
                del self.edges[edge_key]
            if key_to == key:
                del self.edges[edge_key]

        self._lookup = None  # Invalidate lookup

    def delete_edge(self, key_from, key_to):
        key = (key_from, key_to)
        if key in self.edges:
            del self.edges[key]

    def get_node_index_lookup(self):
        if self._lookup is not None:
            return self._lookup

        self._lookup = {}
        for i, concept_name in enumerate(self.nodes.keys()):
            self._lookup[concept_name] = i
        return self._lookup

    def nodes_to_array(self, attr: str, dtype=np.float32):
        lookup = self.get_node_index_lookup()
        array = np.zeros(len(self.nodes), dtype=dtype)
        for key, val in self.nodes.items():
            array[lookup[key]] = val[attr]
        return array

    def edges_to_matrix(self, attr: str, dtype=np.float32):
        lookup = self.get_node_index_lookup()
        size = len(self.nodes)
        matrix = np.zeros((size, size), dtype=dtype)
        for (key_from, key_to), val in self.edges.items():
            matrix[lookup[key_from], lookup[key_to]] = val[attr]
        return matrix

    def nodes_from_array(self, array, attr: str):
        lookup = self.get_node_index_lookup()
        for key, index in lookup.items():
            self._set_node(key, **{attr: array[index]})

    def edges_from_matrix(self, matrix, attr: str):
        lookup = self.get_node_index_lookup()
        for (key_from, key_to), val in self.edges.items():
            self._set_edge(key_from, key_to, **{attr: matrix[lookup[key_from], lookup[key_to]]})

    def update(self, other: 'Graph'):
        self.nodes.update(other.nodes)
        self.edges.update(other.edges)

    def prefixed_copy(self, prefix: str) -> 'Graph':
        new_graph = Graph()
        for k, v in self.nodes.items():
            new_graph.nodes[prefix + k] = v.copy()
        for (k1, k2), v in self.edges.items():
            new_graph.edges[(prefix + k1, prefix + k2)] = v.copy()
        return new_graph

    def update_node_attrs(self, **attributes):
        for attr in self.nodes.values():
            attr.update(attributes)

    def update_edge_attrs(self, **attributes):
        for attr in self.edges.values():
            attr.update(attributes)

    def copy(self) -> 'Graph':
        new_graph = Graph()
        new_graph.edges = self.edges.copy()
        new_graph.nodes = self.nodes.copy()
        return new_graph

    def map_node_values(self, fn: callable):
        return map(fn, self.nodes.values())

    def map_node_keys(self, fn: callable):
        return map(fn, self.nodes.keys())

    def __copy__(self) -> 'Graph':
        return self.copy()

    def __contains__(self, item):
        return item in self.nodes

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return '<Graph {0} nodes, {1} edges>'.format(len(self.nodes), len(self.edges))

from typing import List, Optional
import numpy as np

from hcifcm.fuzzy import Variable, Bounds
from hcifcm.graph import Graph


__all__ = ['simulate', 'simulate_step', 'simulate', 'FCM']


def sigmoid(x, k=1.0):
    return 1 / (1 + np.exp(-k * x))


def simulate_step(concept_values: np.ndarray,
                  edge_matrix: np.ndarray,
                  activation_fn: callable,
                  delta_time: float=1.0,
                  rule: str='dynamic'
                  ):
    np.fill_diagonal(edge_matrix, 0)  # i != j constraint
    if rule == 'dynamic':
        new_concept_values = activation_fn(concept_values + edge_matrix.T @ concept_values)
        return concept_values + (new_concept_values - concept_values) * delta_time
    elif rule == 'rescaled':
        cs = 2.0 * concept_values - 1.0
        new_concept_values = activation_fn(cs + edge_matrix.T @ cs)
        return concept_values + (new_concept_values - concept_values) * delta_time
    elif rule == 'kosko':
        new_concept_values = activation_fn(edge_matrix.T @ concept_values)
        return concept_values + (new_concept_values - concept_values) * delta_time
    """
    new_concept_values = activation_fn(concept_values + edge_matrix.T @ concept_values)
    delta = new_concept_values - concept_values
    new_concept_values = activation_fn(concept_values + edge_matrix.T @ delta)
    return concept_values + (new_concept_values - concept_values) * delta_time
    """

    raise RuntimeError(f'Unknown rule: {rule}')


def simulate(concept_values: np.ndarray,
             edge_matrix: np.ndarray,
             activation_fn: callable,
             delta_time: float=1.0,
             iterations: int = 10,
             tolerance: float=1e-4, **kwargs):
    for i in range(iterations):
        new_concept_values = simulate_step(concept_values, edge_matrix, activation_fn, delta_time, **kwargs)
        if np.sum(np.abs(new_concept_values - concept_values)) < tolerance:
            return new_concept_values
        concept_values = new_concept_values
    return concept_values


class FCM(object):
    VALUE_ATTR = 'value'

    def __init__(self, default_value_mfs: List, default_edge_mfs: List):
        self._default_value_mfs = default_value_mfs
        self._default_edge_mfs = default_edge_mfs
        self.activation = sigmoid
        self.graph = Graph()

    def add_concept(self,
                    name: str,
                    initial=None,
                    bounds: Bounds=(0, 1),
                    mfs: Optional[List] = None,
                    **attributes) -> Variable:
        if mfs is None:
            mfs = self._default_value_mfs

        var = Variable(name, bounds, *mfs)

        if initial is not None and not isinstance(initial, (float, int)):
            initial = var.defuzzify(initial)

        self.graph.node(name, var=var, **attributes, **{self.VALUE_ATTR: initial})
        return var

    def get_concept(self, concept: [str, Variable]) -> Variable:
        if isinstance(concept, str):
            return self.graph.get_node_attr(concept, 'var')
        return concept

    def add_edge(self,
                 source: [str, Variable],
                 target: [str, Variable],
                 initial=None,
                 mfs: Optional[List]=None,
                 bounds: Bounds=(-1, 1),
                 **attributes) -> Variable:
        if mfs is None:
            mfs = self._default_edge_mfs
        edge_var = Variable(f'{source}->{target}', bounds, *mfs)

        if initial is not None and not isinstance(initial, (float, int)):
            initial = edge_var.defuzzify(initial)

        self.graph.edge(source, target, var=edge_var, **attributes, **{self.VALUE_ATTR: initial})
        return edge_var

    def update(self, dt: float=1.0, **kwargs):
        """
        # Update fuzzy concept with fuzzy rules
        for rule in concept_update_rules:
            old_val = concept_state[rule.var.name]
            new_val = rule.evaluate(**concept_state, **edges_state)
            concept_state[rule.var.name] = old_val + dt * (new_val - old_val)

        # Update fuzzy edge with fuzzy rules
        for rule in edge_update_rules:
            old_val = edges_state[rule.var.name]
            new_val = rule.evaluate(**concept_state, **edges_state)
            edges_state[rule.var.name] = old_val + dt * (new_val - old_val)
        """
        c = self.graph.nodes_to_array(attr=self.VALUE_ATTR)
        w = self.graph.edges_to_matrix(attr=self.VALUE_ATTR)
        c_new = simulate_step(c, w, self.activation, delta_time=dt, **kwargs)
        self.graph.nodes_from_array(c_new, attr=self.VALUE_ATTR)

    def get_concepts(self) -> np.ndarray:
        return self.graph.nodes_to_array(attr=self.VALUE_ATTR)

    def get_weight_matrix(self) -> np.ndarray:
        return self.graph.edges_to_matrix(attr=self.VALUE_ATTR)

    def set_weight_matrix(self, weights: np.ndarray):
        self.graph.edges_from_matrix(weights, attr=self.VALUE_ATTR)

    def iter_concepts(self):
        return self.graph.nodes.items()

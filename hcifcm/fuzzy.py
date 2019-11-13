import numpy as np
from typing import List, Tuple, Optional


Bounds = Tuple[float, float]
Term = [str, 'MembershipFunction']


class FuzzySet(object):
    def intersect(self, other: 'FuzzySet') -> 'FuzzyIntersect':
        return FuzzyIntersect(self, other)

    def union(self, other: 'FuzzySet') -> 'FuzzyUnion':
        return FuzzyUnion(self, other)

    def sum(self, other: 'FuzzySet') -> 'FuzzySum':
        return FuzzySum(self, other)

    def multiply(self, other: 'FuzzySet') -> 'FuzzyMultiply':
        return FuzzyMultiply(self, other)

    def negate(self) -> 'FuzzyNegate':
        return FuzzyNegate(self)

    def sample(self, x: float):
        raise NotImplementedError

    def calculate_centroid(self, x_min=0.0, x_max=1.0, steps=100):
        s1 = 0.0
        area = 0.0
        for x in np.linspace(x_min, x_max, steps):
            sample = self.sample(x)
            s1 += x * sample
            area += sample

        if area < 1e-8:
            return 0
        return s1 / area

    def center(self):
        raise NotImplementedError

    def to_discrete(self, x_min=0.0, x_max=1.0, steps=100):
        values = np.zeros(steps, np.float32)
        i = 0
        for x in np.linspace(x_min, x_max, steps):
            values[i] = self.sample(x)
            i += 1
        return DiscreteFuzzySet(values, (x_min, x_max))

    def __call__(self, x):
        return self.sample(x)


class DiscreteFuzzySet(FuzzySet):
    def __init__(self, values: [List, np.ndarray], bounds: Bounds):
        self._values = np.asarray(values)
        x_min, x_max = bounds
        self._x = np.linspace(x_min, x_max, len(values))

    def sample(self, x: float):
        return np.interp(x, self._x, self._values)


class FuzzySetCompound(FuzzySet):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def sample(self, x: float):
        raise NotImplementedError


class FuzzyUnion(FuzzySetCompound):
    def sample(self, x: float):
        return np.maximum(self.args[0].sample(x), self.args[1].sample(x))


class FuzzyIntersect(FuzzySetCompound):
    def sample(self, x: float):
        return np.minimum(self.args[0].sample(x), self.args[1].sample(x))


class FuzzyMultiply(FuzzySetCompound):
    def sample(self, x: float):
        return self.args[0].sample(x) * self.args[1].sample(x)


class FuzzyNegate(FuzzySetCompound):
    def sample(self, x: float):
        return 1 - self.args[0].sample(x)


class FuzzySum(FuzzySetCompound):
    def sample(self, x: float):
        a = self.args[0].sample(x)
        b = self.args[1].sample(x)
        return a + b - a * b


class MembershipFunction(FuzzySet):
    def __init__(self, name: str):
        self.name = name

    def sample(self, x: float) -> float:
        raise NotImplementedError()


class ConstMf(MembershipFunction):
    def __init__(self, value: float):
        super().__init__('const')
        self.value = value

    def sample(self, x: float):
        return self.value


class TriangularMf(MembershipFunction):
    def __init__(self, name: str, left: float, support: float, right: float):
        super().__init__(name)
        self.left = left
        self.support = support
        self.right = right

    def sample(self, x: float):
        if x < self.left:
            return 0
        elif x <= self.support:
            return (x - self.left) / (self.support - self.left)
        elif x <= self.right:
            return (self.right - x) / (self.right - self.support)
        else:
            return 0

    def center(self):
        return self.support


class TrapezoidalMf(MembershipFunction):
    def __init__(self, name: str, left: float, left_support: float, right_support: float, right: float):
        super().__init__(name)
        self.left = left
        self.left_support = left_support
        self.right_support = right_support
        self.right = right

    def sample(self, x: float):
        if x < self.left or x > self.right:
            return 0
        elif x <= self.left_support:
            return (x - self.left) / (self.left_support - self.left)
        elif x <= self.right_support:
            return 1
        elif x <= self.right:
            return (self.right - x) / (self.right - self.right_support)

    def center(self):
        return 0.5 * self.right_support - 0.5 * self.left_support


class GaussianMf(MembershipFunction):
    def __init__(self, name: str, mean: float, std: float):
        super().__init__(name)
        self.mean = mean
        self.std = std

    def sample(self, x: float):
        return np.exp(-np.square(x - self.mean) / (2 * np.square(self.std)))

    def center(self):
        return self.mean


class Variable(object):
    def __init__(self, name: str=None, bounds: tuple=(0, 1), *terms):
        self.membership_functions = {}
        for term in terms:
            if isinstance(term, MembershipFunction):
                if term.name in self.membership_functions:
                    ValueError(f'Membership function with name {term.name} is already added to this variable')
                self.membership_functions[term.name] = term
        self.name = name
        self.bounds = bounds

    def fuzzify_all(self, x: float):
        result = {}
        for mf in self.membership_functions.values():
            result[mf.name] = mf.sample(x)
        return result

    def fuzzify_max(self, x: float):
        best_mf = None
        best_sample = 0
        for mf in self.membership_functions.values():
            sample = mf.sample(x)
            if sample > best_sample:
                best_sample = sample
                best_mf = mf
        return best_mf

    def fuzzify(self, mf: Term, x):
        if isinstance(mf, str):
            # Try to get mf by key
            return self.membership_functions.get(mf).sample(x)
        return mf.sample(x)

    def defuzzify(self, mf: Term):
        bounds_min, bounds_max = self.bounds
        if not isinstance(mf, MembershipFunction):
            # Try to get mf by key
            mf = self.membership_functions.get(mf)
        return mf.calculate_centroid(bounds_min, bounds_max)

    def get_mf(self, mf: Term):
        if not isinstance(mf, MembershipFunction):
            # Try to get mf by key
            mf = self.membership_functions.get(mf)
        return mf

    def get_membership_functions(self):
        return self.membership_functions.values()

    def is_(self, mf):
        return Is(self, mf)


class ConditionStatement:
    def evaluate(self, **kwargs) -> float:
        raise NotImplementedError

    def __call__(self, **kwargs) -> float:
        return self.evaluate(**kwargs)

    def and_(self, cond: 'ConditionStatement'):
        return And(self, cond)

    def or_(self, cond: 'ConditionStatement'):
        return Or(self, cond)

    def __or__(self, other):
        return Or(self, other)

    def __and__(self, other):
        return And(self, other)


class Is(ConditionStatement):
    def __init__(self, variable: Variable, mf):
        self._var = variable
        self._mf = mf

    def evaluate(self, **kwargs):
        x = kwargs[self._var.name]
        return self._var.fuzzify(self._mf, x)


class And(ConditionStatement):
    def __init__(self, *operands):
        self._operands = operands

    def evaluate(self, **kwargs):
        results = (op.evaluate(**kwargs) for op in self._operands)
        return min(results)


class Or(ConditionStatement):
    def __init__(self, *operands):
        self._operands = operands

    def evaluate(self, **kwargs):
        results = (op.evaluate(**kwargs) for op in self._operands)
        return max(results)


class If(object):
    def __init__(self, condition: ConditionStatement):
        self._condition = condition
        self._set_var = None
        self._set_mf = None

    def set(self, variable: Variable, mf: Term):
        self._set_var = variable
        self._set_mf = mf
        return self

    def evaluate(self, **kwargs) -> FuzzySet:
        # Mamdani inference
        alpha = self._condition.evaluate(**kwargs)
        mf = self._set_var.get_mf(self._set_mf)
        return mf.intersect(ConstMf(alpha))

    def __call__(self, **kwargs):
        return self.evaluate(**kwargs)


class Rules(object):
    def __init__(self, var: Variable):
        self.var = var
        self.rules = []

    def add_rule(self, term: Term, condition: ConditionStatement):
        self.rules.append((term, condition))

    def evaluate(self, **kwargs) -> float:
        output_set = None
        for term, condition in self.rules:
            # Mamdani inference
            alpha = condition.evaluate(**kwargs)
            mf = self.var.get_mf(term).intersect(ConstMf(alpha))
            if output_set is None:
                output_set = mf
            else:
                output_set = output_set.union(mf)
        return output_set.calculate_centroid(*self.var.bounds)

    def __call__(self, **kwargs):
        return self.evaluate(**kwargs)


def main():
    bad = GaussianMf('bad', 0, 1)
    average = GaussianMf('average', 1, 1)
    good = GaussianMf('good', 2, 1)

    no_tip = GaussianMf('no_tip', 0, 1)
    poor = GaussianMf('poor', 1, 1)
    decent = GaussianMf('decent', 2, 1)

    # variables
    food = Variable('food', (0, 3), bad, average, good)
    service = Variable('service', (0, 3), bad, average, good)
    tip = Variable('service', (0, 3), no_tip, poor, decent)

    rules = Rules(tip)
    rules.add_rule('no_tip', Is(food, 'bad') & Is(service, 'bad'))
    rules.add_rule('poor', And(Or(Is(food, 'average'), Is(food, 'good')), Is(service, 'average')))
    rules.add_rule('decent', And(Is(food, 'good'), Is(service, 'good')))

    print(rules.evaluate(food=2, service=2))

    import matplotlib.pyplot as plt
    m = np.zeros((20, 20), np.float32)
    for i in range(m.shape[0]):
        food_val = 3 * i / (m.shape[0] - 1.0)
        for j in range(m.shape[1]):
            service_val = 3 * j / (m.shape[1] - 1.0)
            t = rules.evaluate(food=food_val, service=service_val)
            m[i, j] = t
    plt.imshow(m)
    plt.show()


if __name__ == '__main__':
    main()

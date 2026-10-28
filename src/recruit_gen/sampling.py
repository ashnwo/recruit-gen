"""Small sampling helpers shared by the stages."""
import bisect
import itertools


class WeightedPicker:
    """Fast repeated weighted draws from {option: weight}. Weights don't need to add to 1."""

    def __init__(self, weights: dict):
        items = [(k, w) for k, w in weights.items() if w > 0]
        self.keys = [k for k, _ in items]
        self.cum = list(itertools.accumulate(w for _, w in items))

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def pick(self, rng):
        x = rng.random() * self.cum[-1]
        return self.keys[min(bisect.bisect_right(self.cum, x), len(self.keys) - 1)]

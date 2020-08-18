#
# Interval arithmetic (Moore, 1979) and Introduction to custom analysis
#
import math

import numpy as anp


class Interval:

    def __init__(self, lower, upper=None):
        if isinstance(lower, Interval):
            self.lower = lower.lower
            self.upper = lower.upper
        else:
            self.lower = lower
        if upper is None:
            self.upper = lower
        else:
            self.upper = upper

        if type(self.upper) == str:
            self.lower = float(self.lower)
        if type(self.upper) == str:
            self.upper = float(self.upper)

        if self.lower > self.upper:
            pass
            # print('interval invalid')

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Interval(self.lower + other, self.upper + other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __radd__(self, other):
        return Interval(other) + self

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Interval(self.lower - other, self.upper - other)
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def __rsub__(self, other):
        return Interval(other) - self

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Interval(min(self.lower * other, self.upper * other), max(self.lower * other, self.upper * other))

        if isinstance(other, anp.ndarray):
            return anp.array([self * v for v in other])
        a = self.lower * other.lower
        b = self.lower * other.upper
        c = self.upper * other.lower
        d = self.upper * other.upper
        return Interval(min(a, b, c, d), max(a, b, c, d))

    def __rmul__(self, other):
        return Interval(other) * self

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            if math.isnan(other):
                return Interval(math.nan)
            return self * (1 / Interval(other))
        if other.lower == 0 and other.upper > 0:
            return Interval(1 / other.upper, math.inf)
        if other.lower < other.upper == 0:
            return Interval(-math.inf, 1 / other.lower)
        return self * Interval(1 / other.lower, 1 / other.upper)

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return Interval(other) / self
        return other / self

    def __pow__(self, n):
        if isinstance(n, Interval):
            raise TypeError("Only integer or decimal are allowed")
        if self.lower > 0 or n % 2 != 0:
            return Interval(pow(self.lower, n), pow(self.upper, n))
        if self.upper < 0 and n % 2 == 0:
            return Interval(pow(self.upper, n), pow(self.lower, n))
        if self.upper == 0 or self.lower == 0 and n % 2 == 0:
            return Interval(0, pow(self.mag(), n))
        if n == 0.5 and self.lower < 0:
            return Interval(pow(self.lower, n), pow(self.upper, -n))
        a = pow(self.lower, n)
        b = pow(self.upper, n)
        return Interval(min(a, b), max(a, b))

    def exp(self):
        return Interval(math.exp(self.lower), math.exp(self.upper))

    def log(self, base=2):
        return Interval(math.log(self.lower, base), math.log(self.upper, base))

    def sqrt(self):
        return pow(self, 0.5)

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.upper < other
        return self.upper < other.lower or self.possibility(other) < 0.5

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.lower > other
        return self.lower > other.upper or self.possibility(other) > 0.5

    def __le__(self, other):
        return self.possibility(other) <= 0.5

    def __ge__(self, other):
        return self.possibility(other) >= 0.5

    def __eq__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self.lower == other.lower and self.upper == other.upper

    def __ne__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self.lower != other.lower and self.upper != other.upper

    def possibility(self, other):
        if isinstance(other, (float, int)) and self.lower != self.upper:
            return (self.upper - other) / (self.upper - self.lower)
        elif isinstance(other, (float, int)):
            return self.upper - other
        if self.lower == other.lower and self.upper == other.upper:
            return 0
        if self.upper == self.lower and other.upper == other.lower:
            if self.upper > other.upper:
                return 1
            if self.upper < other.lower:
                return -1
            return 0
        return (self.upper - other.lower) / ((self.upper - self.lower) + (other.upper - other.lower))

    def poss_greater_than_or_eq(self, other):
        poss = self.possibility(other)
        if poss <= 0:
            return 0
        if poss >= 1:
            return 1
        return poss

    def poss_smaller_than_or_eq(self, other):
        return 1 - self.poss_greater_than_or_eq(other)

    def __isub__(self, other):
        return self - other

    def __iadd__(self, other):
        return self + other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other

    def __neg__(self):
        return Interval(-1 * self.lower, -1 * self.upper)

    def __pos__(self):
        return Interval(1 * self.lower, 1 * self.upper)

    def intersection(self, other):
        if isinstance(other, (float, int)):
            if other < self.lower or self.upper < other:
                return Interval(0)
            return Interval(max(self.lower, other), min(self.upper, other))
        if other.upper < self.lower or self.upper < other.lower:
            return Interval(0)
        return Interval(max(self.lower, other.lower), min(self.upper, other.upper))

    def union(self, other):
        if isinstance(other, (float, int)):
            return Interval(min(self.lower, other), max(self.upper, other))
        return Interval(min(self.lower, other.lower), max(self.upper, other.upper))

    def width(self):
        return self.upper - self.lower

    def midpoint(self):
        return 0.5 * (self.lower + self.upper)

    def sin(self):
        if - math.pi / 2 <= self.lower <= math.pi / 2 and -math.pi / 2 <= self.upper <= math.pi / 2:
            return Interval(math.sin(self.lower), math.sin(self.upper))
        a = math.sin(self.lower)
        b = math.sin(self.upper)
        return Interval(min(a, b), max(a, b))

    def mag(self):
        return max(abs(self.lower), abs(self.upper))

    def __abs__(self):
        l = self.lower
        u = self.upper
        if l >= 0:
            return Interval(l, u)
        if u <= 0:
            return Interval(min(abs(l), abs(u)), max(abs(l), abs(u)))
        if abs(l) > u:
            return Interval(0, abs(l))
        return Interval(0, abs(u))

    def __str__(self):
        return '{:.3f} {:.3f}'.format(self.lower, self.upper)

    def __repr__(self):
        return '{:.3f} {:.3f}'.format(self.lower, self.upper)

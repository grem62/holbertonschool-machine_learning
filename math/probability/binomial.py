#!/usr/bin/env python3
"""_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            n (_type_, optional): _description_. Defaults to 1.
            p (_type_, optional): _description_. Defaults to 0.5.
        """
        self.data = data
        if data is None:
            self.n = int(n)
            self.p = float(p)
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < self.p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.n = round(n)
            self.p = sum(data) / len(data)
            mean = sum(data) / len(data)
            v = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - (v / mean)
            n = round(mean / p)
            p = mean / n
            self.n = round(n)
            self.p = float(p)

    def pmf(self, k):
        """_summary_

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        if k is not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return (self.factoriel(self.n) / (self.factoriel(k) *
                                          self.factoriel(self.n - k))) *\
            (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def factoriel(self, k):
        """_summary_

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        if k == 0:
            return 1
        return k * self.factoriel(k - 1)

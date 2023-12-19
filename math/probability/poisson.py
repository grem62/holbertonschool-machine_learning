#!/usr/bin/env python3
"""
    _summary_
"""


class Poisson:
    """
    _summary_
    """
    def __init__(self, data=None, lambtha=1.):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            lambtha (_type_, optional): _description_. Defaults to 1..
        """
        self.data = data
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def factorial(self, k):
        """_summary_

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        if k == 0:
            return 1
        return k * self.factorial(k - 1)

    def pmf(self, k):
        """_summary_

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        exponentiel = 2.7182818285
        if k is not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        pmf_value = (self.lambtha ** k) * (exponentiel ** (-self.lambtha))
        return pmf_value / self.factorial(k)

    def cdf(self, k):
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
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value

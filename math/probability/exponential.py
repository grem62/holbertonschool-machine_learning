#!/usr/bin/env python3
"""
    _summary_
"""


class Exponential:
    """_summary_
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
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        if x < 0:
            return 0

        Exponential = 2.7182818285

        pdf_value = self.lambtha * (Exponential ** (-self.lambtha * x))
        return pdf_value

    def cdf(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        Exponential = 2.7182818285
        if x < 0:
            return 0
        cdf_value = 1 - (Exponential ** (-self.lambtha * x))
        return cdf_value

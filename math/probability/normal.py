#!/user/bin/env python3
"""_summary_
"""


class Normal:
    """_summary_
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            mean (_type_, optional): _description_. Defaults to 0..
            stddev (_type_, optional): _description_. Defaults to 1..
        """
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = ((sum([(x - self.mean) ** 2 for x in data])
                            / len(data)) ** 0.5)

    def z_score(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        self.x_value = x
        value_x = self.x_value
        pi = 3.1415926536
        e = 2.7182818285
        pdf_v = (e ** (-0.5 * ((value_x - self.mean) / self.stddev) ** 2)) / (
            (self.stddev * ((2 * pi) ** 0.5))
        )
        return pdf_v

    def erf(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        pi = 3.1415926536
        erf_value = 2 / (pi ** 0.5) * (x - (x ** 3) / (
            3 + (x ** 5) / 10 - (x ** 7) / 42 + (x ** 9) / 216)
        )
        return erf_value

    def cdf(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        cdf_value = (1 + self.erf((x - self.mean) / (
            (self.stddev * (2 ** 0.5)))) / 2
        )
        return cdf_value

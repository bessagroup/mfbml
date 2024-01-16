# this script is used for implementing mfb test suites from Handing Wang's 2018
# paper


import torch


class MFB1:
    """class for mfb1 function"""

    def __init__(self,
                 num_dim: int,
                 noise_std: float,
                 phi: float) -> None:
        """constructor"""
        # get the dimension
        self.num_dim = num_dim
        # set the noise std
        self.noise_std = noise_std
        # set the phi
        self.phi = phi

    def __call__(self, samples: dict) -> dict:
        """evaluate the function

        Parameters
        ----------
        samples : dict
            samples

        Returns
        -------
        dict
            responses
        """

        # get samples
        hf_samples = samples["hf"]
        lf_samples = samples["lf"]

        # evaluate the problem
        responses = {"hf": self.hf(hf_samples),
                     "lf": self.lf(lf_samples)}

        return responses

    def hf(self, x: torch.Tensor,
           noise_hf: float = None  # type: ignore
           ) -> torch.Tensor:
        """high fidelity function

        Parameters
        ----------
        x : torch.Tensor
            high fidelity input x
        noise_hf : float, optional
            noise std, by default None#type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_hf is None:  # use the default noise
            noise_hf = self.noise_std

        obj = (x**2+1-torch.cos(10*torch.pi*x)).sum(dim=1, keepdim=True) + \
            noise_hf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def lf(self,
           x: torch.Tensor,
           noise_lf: float = None  # type: ignore
           ) -> torch.Tensor:
        """low fidelity function

        Parameters
        ----------
        x : torch.Tensor
            low fidelity input x
        noise_lf : float, optional
            noise std, by default None#type:ignore

        Returns
        -------
        torch.Tensor
            outputs
        """

        if noise_lf is None:  # use the default noise
            noise_lf = self.noise_std

        obj = self.hf(x, noise_hf=0.0) + self.error(x) + \
            noise_lf * torch.randn(x.shape[0]).reshape(-1, 1)

        return obj.reshape(-1, 1)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """error function

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            outputs
        """

        obj = self.a*torch.cos(self.w*x + self.b +
                               torch.pi).sum(dim=1, keepdim=True)

        return obj.reshape(-1, 1)

    @property
    def a(self) -> float:
        """a function

        Returns
        -------
        float
            a
        """

        return self.theta

    @property
    def b(self) -> float:
        """b function

        Returns
        -------
        float
            b
        """

        return 0.5*torch.pi*self.theta

    @property
    def w(self) -> float:
        """w function

        Returns
        -------
        float
            w
        """

        return 10.0*torch.pi*self.theta

    @ property
    def theta(self) -> float:
        """theta function

        Returns
        -------
        float
            theta
        """

        return 1 - 0.0001 * self.phi

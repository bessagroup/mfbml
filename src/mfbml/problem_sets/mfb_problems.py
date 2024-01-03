# this script is used for implementing mfb test suites from Handing Wang's 2018
# paper


import torch


def hf_function(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    """high fidelity function

    Parameters
    ----------
    x : torch.Tensor
        input
    noise_level : float
        noise level

    Returns
    -------
    torch.Tensor
        response
    """
    # evaluate the function
    obj = (x**2 + 1 - torch.cos(10*torch.pi*x)).sum(dim=1, keepdim=True)

    # add noise
    obj += noise_level * torch.randn(obj.shape)

    return obj


# mfb1 function
class MFB1:
    """class for mfb1 function"""

    def __init__(self, noise_std: float) -> None:
        """constructor"""
        # set the noise std
        self.noise_std = noise_std

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

        obj = torch.sin(8*torch.pi*x)**2*(x - torch.sqrt(torch.Tensor([2.0]))) + \
            noise_hf * torch.randn(x.shape)

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

        obj = 0.5 * self.hf(x, noise_lf) + 0.5 * \
            self.hf(x, noise_lf * 0.1) + noise_lf * torch.randn(x.shape)

        return obj.reshape(-1, 1)

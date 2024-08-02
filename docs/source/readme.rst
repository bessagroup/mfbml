
Summary
=======

Multi-fidelity machine learning methods address the accuracy-efficiency trade-off by integrating scarce, resource-intensive high-fidelity data with abundant but less accurate low-fidelity data. We propose a practical multi-fidelity strategy for problems spanning low- and high-dimensional domains, integrating a non-probabilistic regression model for the low-fidelity with a Bayesian model for the high-fidelity. The models are trained in a staggered scheme, where the low-fidelity model is transfer-learned to the high-fidelity data and a Bayesian model is trained for the residual. This three-model strategy -- deterministic low-fidelity, transfer learning, and Bayesian residual -- leads to a prediction that includes uncertainty quantification both for noisy and noiseless multi-fidelity data. The strategy is general and unifies the topic, highlighting the expressivity trade-off between the transfer-learning and Bayesian models (a complex transfer-learning model leads to a simpler Bayesian model, and vice versa). We propose modeling choices for two scenarios, and argue in favor of using a linear transfer-learning model that fuses 1) kernel ridge regression for low-fidelity with Gaussian processes for high-fidelity; or 2) deep neural network for low-fidelity with a Bayesian neural network for high-fidelity. We demonstrate the effectiveness and efficiency of the proposed strategies and contrast them with the state-of-the-art based on various numerical examples. The simplicity of these formulations makes them practical for a broad scope of future engineering applications. 



Authorship
----------

**Authors**:
    - Jiaxiang Yi(yagafighting@gmail.com)


**Authors affiliation:**

    - Delft University of Technology



Community Support
-----------------

If you find any issues, bugs or problems with this package, you can raise an issue 
on the github page, or contact the Authors directly.

License
-------

Copyright 2023, Jiaxiang Yi

All rights reserved.


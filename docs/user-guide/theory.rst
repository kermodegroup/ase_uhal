.. _theory:
Theory
=======

Linear MLIPs & Bayesian Regression
----------------------------------
A linear model is any model which can be expressed as :math:`M(x, \theta) = f(x)\theta`, where :math:`\theta` are model weights. 
For linear MLIP models, this :math:`f(x)` is often the descriptor function (e.g. the ACE descriptor for ACE models).
Often this is formulated on a per-atom basis, thus the energy of atom :math:`i` is expressed as :math:`E_i = d(x)\theta`. 
Here, it is more useful to formulate this in terms of a total energy, 

.. math:: 
    E_T &= \sum_i E_i \\
        &= \left(\sum_i d(x_i)\right) \theta \\
        &= D(x) \theta

where :math:`D(x)` can be thought of as an "energy descriptor". Forces and stresses can be easily obtained by taking derivatives:

.. math::
    F_i &= \nabla_{r_i} E_T \\
        &= \left(\nabla_{r_i} D(x)\right) \theta

and thus :math:`\nabla_{r_i} D(x)` can be though of as a "force descriptor" for the force on atom i.

If we have a structure :math:`\Pi` we would like to train on, linear MLIP models usually start by assembling a design matrix
:math:`\Phi`, which can be partitioned into sub-matrices by energy, force, and stress observations:

.. math:: \Phi &= \begin{pmatrix} \Phi_E \\ \Phi_F \\ \Phi_S\end{pmatrix}

where :math:`\Phi_{E} = w_E D(\Pi)`, :math:`\Phi_{F;i} = w_F \nabla_{r_i}D(\Pi)`, etc... Here, :math:`w_E` and `w_F` are weights 
on energy and force observations (to describe a relative importance of each). Including observations from multiple structures can 
again be achieved by constructing an augmented matrix 

.. math:: \Phi = \begin{pmatrix} \Phi_0 \\ \Phi_1 \\ \vdots \\ \Phi_N \end{pmatrix}

Given a zero-mean prior, with prior covariance :math:`P`, we can express the model weights as :math:`\theta \sim \mathcal{N}(\mu_\theta, \Sigma_\theta)`,
where

.. math::
    \Sigma_\theta &= \left(\Phi^T\Phi + P\right)^{-1} \\
    \mu_\theta &= \Sigma_\theta^{-1} \Phi y

where :math:`y` is a vector of energy, force, and stress observations for each structure.

Typically we use the mean of the distribution :math:`\mu_\theta` to inform the model weights, but we could also draw samples from the distribution 
to obtain a committee of models.

ACEHAL
------
Hyperactive Learning (HAL), developed by Van der Oord et. al. :cite:`ACEHAL`, forms the basis of the approach used here. 
Their ACE-specific implimentation can be found on the `ACEHAL GitHub <https://github.com/ACEsuit/ACEHAL/tree/main>`__.


Approximating ACEHAL
---------------------------------

Descriptors from Non-linear models
----------------------------------


Error-Biased Dynamics
---------------------

.. rubric:: References
.. bibliography::

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


where :math:`\Phi_{E} = w_E D(\Pi)`, :math:`\Phi_{F;i} = w_F \nabla_{r_i}D(\Pi)`, etc... Here, :math:`w_E` and :math:`w_F` are weights 
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

The basis of ACEHAL is to first form a committee of ACE models based on some dataset, and then define a biasing potential energy based 
on the standard deviation of the committee energy predictions. This gives a modified energy of

.. math::
    E = E_\mu + \tau \sqrt{\frac{1}{N}\sum_j^N (E_j - E_\mu)^2}


where :math:`E_\mu` is the energy predicted by the mean model (using weights of :math:`\mu_\theta`), :math:`E_j` is the energy prediction made by committee member :math:`j`, and
:math:`\tau` is a biasing strength parameter. Bias forces can be obtained by taking the derivative of this bias potential with respect to each atomic position.

Van der Oord et. al. also define a selection criterion based on the balance between bias forces (before multiplication with the biasing strength parameter :math:`\tau`) and the 
"true" forces from the mean model:

.. math::
    F_{s;i} &= \frac{|F_{b;i}|}{|F_{\mu;i}| + \epsilon} \\
    s &= \max\left(\frac{\mathrm{e}^{F_{s;i}}}{\sum_j \mathrm{e}^{F_{s;j}}}\right)


where :math:`F_{\mu;i}` is the unbiased force on atom :math:`i`, :math:`F_{b;i}` is the biasing force on atom :math:`i`, :math:`\epsilon` is some small regularisation constant,
and :math:`s` is the overall score for the structure.

Their HAL protocol proceeds by running molecular dynamics (MD) with the mean + bias model, and selecting structures once the score for that structure exceeds some fixed tolerance :math:`s_\text{tol}`.
After each structure is selected, the code uses some reference ASE calculator (usually DFT) to obtain ground truth energies, forces, and stresses, and then refits the committee based on the newly added structure.

Approximating ACEHAL
---------------------------------
One major drawback of the original HAL approach is the poor scaling with dataset size (due to the ACE refit taking longer), number of atomic species (as the ACE descriptor has poor scaling w.r.t number of species), and
the fact that only a single structure can be selected between each call to the DFT code and model refit.

The scaling w.r.t the number of species can be solved by using a descriptor which uses atomic embedding to not scale in complexity with the number of chemical species (e.g. MACE; see the next section for more details.)

We can also reduce the impact of the other issues by trying to reduce the number of times we require calls to DFT or to model refits. One way of doing this is by making approximations to a full Bayesian treatment.

From a true Baysian treatment, we obtain the posterior covariance

.. math::
    \Sigma_\theta &= \left(\Phi^T\Phi + P\right)^{-1}


where :math:`\Phi` is the design matrix formed from descriptor vectors and derivatives. We do not actually require any targets :math:`y` (i.e. energy, force, and/or stress observations) in order to calculate the posterior covariance.
The weights of a committee member can be found using

.. math::
    \theta_j = \mu_\theta + \Psi z_j \\
    \Psi^T \Psi = \Sigma_\theta


where :math:`\mu_\theta` are the mean model weights (which do require targets :math:`y` to calculate), :math:`\Psi` is a square root of the posterior covariance matrix, 
and :math:`z_j` is a sample from a multivariate normal distribution with unit variance and zero mean.

If we instead use a zero-mean committee (by removing the :math:`\mu_\theta` term), it is clear we also lose the dependence on targets :math:`y`. 
This allows the commmittee to be calculated without needing any DFT calculations, at the cost of losing the "true" potential energy surface. 
In the HAL framework, we still have the biasing potential, but no mean function.

We can however use some existing MLIP model as the mean function. If this model is an ACE potential, and the zero-mean committee was constructed using the same ACE descriptor parameters,
this is equivalent to the approximation that the mean function is constant between refits of the model in the HAL protocol.

The advantage this approach brings is that we no longer need to run DFT between each selected structure (as we are able to update the now zero-mean committee without needing the DFT energy, forces, and stresses),
which then allows us to select structures at a higher rate (in terms of real-time duration between each selection). Once we have a collection of selected structures, we can then also perform the DFT calculations
for those structures in parallel (taking advantage of whatever High Performance Compute (HPC) resources are available), and then perform a single refit of the mean function model.
This could then be used as a starting point for a new round of biased dynamics, and a new set of selected structures.

The main drawback of this new approach is that we do not allow the mean function model to update as regularly. This means that the approach works best when the mean function model can provide a reasonable approximation
of the true potential energy surface. Foundation models, such as the MACE MPA model, provide a near-univerally adequate starting point for such a task.

Updating the Linear System
++++++++++++++++++++++++++

If we have some initial design matrix :math:`\Phi` of an existing dataset, and a design matrix :math:`\Phi^*` corresponding to some structure(s) we'd like to add to the dataset, we can describe the updated posterior covariance
in two main ways.

.. math::
    \left(\Sigma_\theta^*\right)^{-1} &= \Phi^T\Phi + \Phi^{* T}\Phi^* + P \\
    &= \begin{pmatrix}\Phi & \Phi^* \end{pmatrix} \begin{pmatrix}\Phi \\ \Phi^* \end{pmatrix} + P


The first way constructs an augmented design matrix by appending the new design matrix to the old one. The advantage to this approach is numerical stability when we come to solve for the committee weights.


.. math::
    \left(\Sigma_\theta^*\right)^{-1} &= \Phi^{* T}\Phi^* + \left(\Phi^T\Phi + P\right) \\
    &= \Phi^{* T}\Phi^* + \Sigma_\theta^{-1}


The second way uses the old posterior covariance as a prior for the new linear system. This is much more memory efficient (assuming :math:`N_\text{obs} \gg N_\text{desc}`), and also has the advantage that the old posterior covariance could be saved
to a file for reuse. In this way, we are able to "distill" foundation model datasets into good priors for new linear systems, using the descriptor vector provided by that foundation model.

Descriptors from Neural Network models
++++++++++++++++++++++++++++++++++++++
Neural Network models, such as MACE, can be viewed as a set of transformations on an original descriptor vector (though graph-based approaches do essentially perform some mixing between descriptors from neighbour atoms).
We can therefore describe the result after each transformation (i.e. the set of node values at a hidden layer) as some new descriptor vector. We can then generate linear surrogate models based on MACE descriptor vectors, where 
we cleave the MACE architecture at some layer (by default, we cleave just before the energy readouts), and utilise a committee of such linear models to describe a HAL-style biasing potential.

Unlike the ACE descriptor, these descriptor functions are model-dependent (and not just hyperparameter-dependent) which means the descriptor space is not conserved when the model is retrained. One solution to this is to freeze model weights
during training (e.g. the MACE freeze approach :cite:`mace-freeze`)


.. rubric:: References
.. bibliography::

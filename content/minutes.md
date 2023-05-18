+++
title = "Minutes"
date = "2022-01-11"
aliases = ["meeting-minutes"]
name = "Hugo Authors"
+++

## 18th May 2023

- Dan, Tom, Henry, and Francesco.
- Moving towards v1
  - Nitty gritty tasks
    - Clean up LinOps
      - More tests should be added to ensure that LinOps is working as expected.
      - Once this is done, we can add functionality to define arithmetic operations between LinOps. 
    - Add the ability to attach priors to the `PyTree` object
      - We currently support bijectors and the trainability status to be attached to a parameter. The only thing missing is the ability to attach a prior distribution to a parameter.
        - Considerations should be made to ensure that the prior is defined on the correct space. 
    - Representation of the PyTree
      - Not a hard need, but it would be helpful to have a way of printing the PyTree out in the terminal with the parameters' value, bijector, trainability status, and prior distribution. Similar to how [PyTreeClass](https://github.com/ASEM000/PyTreeClass) does it.
  - Operator kernels (Henry)
    - Focussed around equation discovery
    - End result will be some cool notebooks
    - Long-term, using GP to discover these operator
  - Non-stationary kernels (Francesco)
    - Spectral approximations for decoupled sampling
  - Generalise out the decoupled sampler (Dan)
    - Abstract out the sampler
    - Once the abstraction has been done, look into using other approximations within the sampler.
    - Tentative idea, have the ApproximateSampler return a Prior/Posterior GP whose methods are identical to its exact analogue.
  - Conjugate gradients in GPJax (Tom)
    - Want to have something that supports Float32
    - Also ties into the decoupled sampler routine as another prior approximation
    - Long-term, it'd be nice to tie this into probabilistic numerics e.g., iterGP.
  - Variational expectation abstractions (Tom)
    - Currently we use quadrature everywhere
    - For Gaussian likelihoods, the integral can be done analytically
    - We should abstract out this logic and supply it to the ELBO.


## 22nd February 2023

- Dan, Tom
- GPJax Refactoring
  - To remove
    - `abstractions.py`
      - Training loops to be moved into JaxUtils
      - Should have an explicit training loop in a notebook. Probably regression
    - `config.py`
      - All parameters' config handled through the new
      - jitter can be handled on the class
    - `kernels.py`, `parameters.py`, `types.py`, `utils.py`
  - To change
    - `gps.py`,
      - Perhaps restructure `AbstractPrior` and `AbstractPosterior` to be a common `GP` object.
      - Biggest change will be the need for a `loss_functions` module
      - `init_params` will go.
      - `predict` and `__call__` will go.
    - `likelihoods.py`, `mean_functions.py`
      - Refactoring to be done here `__call__`
    - `variational_families.py` and `variational_inference.py`
      - Similar to `gps.py`
  - Unchanged
    - `gaussian_distribution.py`, `quadrature.py`
      - Open an issue to say that it is not currently Equinox compatible but it is not an issue right now.
  - To add
    - `loss_functions.py`
      - The individual loss fn will accept the posterior as input
      - Return a fn that we can then pass into `fit`.
    - 

## 15th February 2023

- Dan, Tom
- Update on `JaxUtils` (Dan)
  - Progress bar decorator removed.
  - Replaced with `vscan` - verbose `scan` function that is a little safer and allows for more explicit messages to be printed to the terminal.
  - A lot of work has gone into `module.py`
    - Models can be built by subclassing `ju.Module`
    - `ju.param` wraps any parameter that we wish to compute derivatives with respect to.
      - Can attach transformations and trainability status to the parameters.
- Wrapper class for Distrax/TFP bijectors
  - Right now we can just pass in a bijector from Distrax/TFP and it will work. Nothing will break.
  - The PyTree that is built within Distrax is suboptimal implementation.
    - Quite fragile. For example, running a `vmap` over the distribution can break it.
  - Proposed long-term solution would be to define our own bijectors and distributions. 
  - Short-term we can leave things as they are.
- Refactoring GPJax:
  - Have an `objectives.py` that contains MLL, log-posterior, collapsed and uncollapsed ELBOs.
- Takeaways:
  - Add Priors into the parameter metadata (Dan)
  - Schedule meeting with Patrick to review the `ju.Module` implementation (Dan)
  - Begin a JaxUtils intro notebook (Dan)
  - JaxKern refactoring (Tom)
  - ARD bug in kernel (Tom)
  - Begin refactoring the docs TOC (Tom)
  - Discuss next week the GPJax v0.6 release 


## 8th February 2023

- Dan, Henry, Tom
- Takeaways:
  - Equinox-based API discussion 
    - Move the bijectors and trainability objects inside the model - the user should not have to worry about these floating around
    - Happy with the overall API that allows for natural JAX functions to be used, not Equinox functions
    - Will begin transitioning JaxKern to use the new backend 
      - Henry to refactor stationary kernels
      - Tom all other kernels
  - Looking to open up a line-of-work on heteroscedastic likelihood functions
  - Henry to sketch out a framework for the decoupled sampling implementation.

## 25h January 2023

- Dan, Henry, Tom
- Overview of `PyTree`s in Equinox
  - Using the PyTree in Equinox will resolve performance issues in GPJax that Patrick highlighted
  - Using the Equinox PyTree will allow us to _register_ the parameters as nodes in the tree.
  - Operations would now be done on the PyTree itself, not directly on the parameters.
  - Summary for changing:
    - Branch on each module
    - Kernels must be done first
    - Losses must be defined outside of the model
    - Delete params everywhere and update params
- Decoupled sampling
  - No point supporting the old RFF sampling
  - Option 1 - have sampling functionality bolted onto model
    - This is helpful as the sampling approach is coupled to the model e.g., GPR/SVGP
  - Option 2 - Have a sampling module that takes in a model
    - Nicer abstraction
    - It's slightly less easy to use as it requires an extra step
  - Option 3 - Have a truncated `Prior` object
    - Might be a pain if you forget to truncate the prior and have optimised the model.
  - Need to make sure that the decoupled sampling with a heteroscedastic likelihood is supported.

## 11th January 2023

- Dependency structure
  - Create a DAG diagram
  - Have a config file in JaxUtils
    - When importing GPJax, add in the bijectors information to the config file
    - Possible drop a global dictionary in favour of using a PyTree for parameters  
      - Equinox does something clean here
      - Tom to look at and open an issue.
- Plans for future development page on website
  - Detail the [roadmap]({{< ref "/roadmap" >}}) for future development at a package level.
- Move training abstraction out into TuneGP
  - Ensure it's deprecated for 1-2 months before dropping entirely from GPJax.
- v0.6.0
  - Will support non-Gaussian likelihoods (Rens PR).
    - This is likely not to be a breaking change
  - Having parameter PyTrees would be a nice addition (Tom).
  - Inter-domain inducing points (Dan)
    - Almost certainly will break existing code
- Documentation (Tom)
  - Spring clean
  - Regression notebook
    - Demonstrate a few steps of GD
    - Then show it using an abstraction
  - Split up the kernel guide
  - Consolidate docs into one site
    - Have individual sections for packages where relevant e.g., JaxKern, MOGPJax
  - Make subdomain of jaxgaussianprocesses.com (i.e., docs.jaxgaussianprocesses.com)

- Takeaways
  - Dan to explore how other packages leverage inter-domain inducing point computations. 
  - Dan to scope out what would be needed in the JaxKern and/or GPJax to support inter-domain ivs.
  - Tom to create a subdomain and consolidate docs.
  - Tom to scope out the use of PyTrees for parameters e.g., can grads to be taken? How easy/clean is it to attach bijectors, priors.etc to the parameter? Do NumPyro do something clever here?
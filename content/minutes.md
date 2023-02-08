+++
title = "Minutes"
date = "2022-01-11"
aliases = ["meeting-minutes"]
name = "Hugo Authors"
+++

## 8th February 2022

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

## 25h January 2022

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

## 11th January 2022

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
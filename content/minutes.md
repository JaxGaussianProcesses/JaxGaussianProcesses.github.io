+++
title = "Minutes"
date = "2022-01-11"
aliases = ["meeting-minutes"]
name = "Hugo Authors"
+++

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
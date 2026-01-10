---
layout: index_sample
title: Launch Sarek on Binder
---

# Try Sarek in your Browser

You can experiment with Sarek without installing anything on your local machine using **Binder**. 

Binder provides a temporary cloud environment with OCaml 5.4, the Sarek framework, and a Jupyter notebook interface already configured.

## Launch the Environment

Click the button below to start your private Sarek playground:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mathiasbourgoin/Sarek/main?urlpath=lab/tree/notebooks/Introduction_to_Sarek.ipynb)

*(Note: It may take a few minutes to build the environment the first time you click this link.)*

## What's included?

- **OCaml 5.4.0**: Fully utilizing the modern multicore runtime.
- **Sarek Framework**: The core DSL and the Native Parallel backend.
- **Interactive Notebooks**: Real-world examples you can edit and run immediately.

## Why Binder?

Since GPGPU programming typically requires specialized hardware (NVIDIA/AMD GPUs), it can be difficult to try out for the first time. 

By using the **Native CPU Backend**, Sarek allows you to run parallel kernels on standard cloud CPUs. The code you write here is **identical** to the code you would run on a high-end data-center GPU.

## Next Steps

Once the notebook launches:
1. Open `Introduction_to_Sarek.ipynb`.
2. Run the cells to initialize the library.
3. Modify the kernel logic to see how it affects the results!

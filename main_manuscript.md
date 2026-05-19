---
title: 'pyfli: A Unified Python Platform for Fluorescence Lifetime Imaging Data Processing'
tags:
  - Python
  - fluorescence lifetime imaging
  - FLIM
  - FLI
  - TCSPC
  - SPAD
  - ICCD
  - phasor analysis
  - Laguerre deconvolution
  - GPU computing
  - biomedical optics
authors:
  - name: Vikas Pandey
    orcid: 0000-0000-0000-0000
    corresponding: true
    email: pandev2@rpi.edu
    affiliation: 1
affiliations:
  - name: Center for Modeling, Simulation and Imaging in Medicine, Rensselaer Polytechnic Institute, USA
    index: 1
date: 18 May 2026
bibliography: paper.bib
---

# Summary

Fluorescence Lifetime Imaging (FLI) measures the nanosecond-scale decay of
fluorescent emission and provides a quantitative, intensity-independent
contrast for biological imaging — from intracellular metabolic state to
*in vivo* tumour characterisation. In practice, FLI data are acquired with
three very different detector families: gated cameras, single-photon
avalanche diode (SPAD) arrays, and time-correlated single-photon counting
(TCSPC) microscopes. Each platform produces data in its own format with its
own noise statistics, and vendor analysis tools rarely interoperate. This
fragmentation forces every laboratory to maintain its own pipeline and makes
cross-platform reproducibility difficult.

`pyfli` is an open-source Python library that brings ingestion, simulation,
and lifetime estimation for FLI under a single hardware-agnostic interface.
It implements five established lifetime estimators, matched CPU and GPU
solver backends, a physics-aware noise simulator, and a compressed-sensing
reconstruction module for hyperspectral single-pixel FLI. The package is
distributed on PyPI as `pyfli-lib` and imported as `pyfli`.

![High-level architecture of `pyfli`. Raw vendor data are ingested through a
single `DataOperations` registry, preprocessed (IRF alignment, background and
hot-pixel correction), and dispatched to either analytical estimators or
CPU/GPU solver backends. A parallel simulator path produces calibrated
synthetic data for training and validation of deep-learning
models.\label{fig:architecture}](figures/architecture.png){width="90%"}

# Statement of need

FLI is a mature quantitative imaging modality used widely in biophysics,
microscopy, and preclinical biomedical optics [@Becker2012; @Berezin2010].
Its hardware ecosystem, however, is heterogeneous. Intensified charge-coupled
device (ICCD) cameras dominate gated wide-field and macroscopic FLI;
SPAD arrays such as SwissSPAD2/3 provide megapixel-scale photon counting
[@Bruschini2019]; TCSPC microscopes record per-pixel arrival histograms in
scanning configurations. Each platform produces data with different geometry,
noise statistics, and file conventions.

The analytical landscape is equally fragmented. Non-linear least-squares
fitting dominates classical FLIM; phasor analysis [@Digman2008] is preferred
in lipid microscopy; maximum-likelihood estimation is used in the
photon-starved regime typical of *in vivo* imaging [@Kollner1992];
Laguerre-expansion techniques [@Jo2005] address model-free deconvolution; and
deep-learning approaches [@Smith2019] now offer real-time inference when
paired with calibrated simulators. These methods are typically scattered
across vendor software, academic prototypes, and disconnected community
packages, with inconsistent APIs and few common benchmarks.

`pyfli` was developed to (i) provide a uniform Python interface across ICCD,
SPAD, and TCSPC data; (ii) expose the major analytical estimators behind
consistent class APIs so that they can be compared on the same data;
(iii) provide matched CPU and GPU backends that allow the same estimator to
scale from a single decay to multi-megapixel cubes; and (iv) include a
hardware-aware simulator producing labelled synthetic data suitable for
training and validating deep-learning models. The library targets two
audiences: experimentalists who need a reliable, format-agnostic pipeline,
and method developers who need a common substrate for prototyping and
benchmarking new lifetime estimators.

# State of the field

Several open packages address parts of the FLI workflow. `FLUTE`
[@Gottlieb2023] and `PhasorPy` [@PhasorPy2024] provide phasor-based analysis
with strong visualisation; `FLIMfit` [@Warren2013] delivers a mature graphical
environment for least-squares FLIM fitting; `napari-flim-phasor-plotter`
integrates phasor analysis into the napari ecosystem. On the simulation side,
`flimlib` provides fast curve-fitting primitives but no end-to-end pipeline.
None of these packages simultaneously target (a) data ingestion across all
three major detector families, (b) multiple estimator families behind a
common API, (c) GPU-accelerated per-pixel fitting, and (d) a calibrated noise
simulator designed for generating deep-learning training data.

`pyfli` occupies this combined niche. To our knowledge it is the first
package that bundles least-squares, phasor, maximum-likelihood, rapid
lifetime determination, and Laguerre deconvolution behind a unified Python
interface with GPU support and an integrated simulator.

# Software design

`pyfli` is implemented in pure Python (≥3.11), comprises approximately 70
modules and 12,000 lines of code, and is organised into seven thematic
sub-packages exposed through a single top-level namespace:

- **`dataIO`** — universal data and IRF loaders for `.sdt`, `.mat`, `.tif`,
  `.npy`, `.txt`, and `.asc` files, with a detector-abstracted import path
  for ICCD, SwissSPAD2/3, and TCSPC outputs.
- **`dataCC`** — IRF alignment, normalisation, region-of-interest operations,
  background subtraction, hot-pixel rejection, and pile-up correction.
- **`analytical_methods`** — phasor analysis, non-linear least-squares
  fitting, Poisson maximum-likelihood estimation, and a Laguerre Expansion
  Technique (LET) fitter for model-free IRF deconvolution.
- **`solver`** — CPU and GPU lifetime processors built on NumPy/SciPy and
  PyTorch respectively, plus binned, global, and MLE variants and a fitting
  comparator.
- **`simulator`** — a forward-model engine, parameter distribution samplers,
  modular noise models (Poisson shot noise, dark count rate, Gaussian read
  noise, gain, quantisation), batch simulators, and a calibration/validation
  engine.
- **`spAnalysis`** — single-pixel compressed-sensing reconstruction for
  hyperspectral time-resolved imaging, with Hadamard and Fourier-DCT bases
  and linear, total-variation, and Poisson-likelihood reconstructors
  [@Pian2017].
- **`dataVnP`, `roiMaker`, `utils_common`** — visualisation, ROI editing,
  and shared utilities.

Three design decisions shape the package. First, **registry-based dispatch**
(loaders, noise models, reconstructors) is preferred over deep class
hierarchies, keeping each subsystem extensible without subclassing pressure.
Second, **CPU/GPU symmetry**: the `Fli_CPUProcessor` and `Fli_GPUProcessor`
classes accept the same fitter object and dataset and produce comparable
outputs, with the GPU path using differentiable reparameterisations
(`exp` for positive scalars, `sigmoid` for amplitude fractions, offset
parameterisation for ordered lifetimes) to enforce physical bounds without
boundary-handling pathologies common in box-constrained gradient methods.
Third, the **simulator closes the loop with experiment**: `FLICalibrator`
fits simulator hyperparameters against an experimentally acquired calibration
sample, so the synthetic data used to train downstream deep-learning models
matches the noise statistics of the target instrument.

Quality control is provided by a `pytest` suite covering parameter
validation, distribution sampling, noise model statistics, phasor coordinate
computation, and the Laguerre fitter. Tests use only synthetic NumPy arrays
and require no external data.

# Research impact statement

`pyfli` consolidates analytical methods and hardware support that previously
required several disconnected codebases, lowering the activation energy for
laboratories adopting FLI as a quantitative readout. By exposing
least-squares, phasor, MLE, RLD, and Laguerre estimators behind a common API
on identical data, it enables direct method comparison — a step that is
needed for principled estimator selection but has historically been
discouraged by interface friction. The matched CPU/GPU backends additionally
allow the same pipeline to scale from a single calibration decay to
multi-megapixel preclinical FLI cubes, supporting both bench-scale
methodological work and high-throughput biomedical studies.

The integrated simulator and calibration engine are intended to support the
growing body of work on deep-learning FLI inference, where realistic, labelled
training data calibrated to specific detectors is a recurrent bottleneck
[@Smith2019]. By generating training corpora that match the noise statistics
of a target instrument, `pyfli` aims to make these methods more reproducible
across laboratories and detector platforms.

# AI usage disclosure

Portions of the documentation and this manuscript were drafted with
assistance from a large language model (Anthropic Claude, 2026), used for
prose drafting and structural editing. All scientific content, software
design decisions, code, and final wording were reviewed and approved by the
author. The AI was not used to generate experimental results, software
behaviour claims, or citations; references were verified by the author.

# Acknowledgements

*[Funding sources, lab affiliations, and individual acknowledgements to be
added.]*

# References

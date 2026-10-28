# Third-party notices

This implementation was written for the supplied BasisBFN manuscript and
uses the published algorithms and interfaces of the following projects as
technical references.

## Restormer

Copyright (c) 2022 Syed Waqas Zamir and contributors.

Source: <https://github.com/swz30/Restormer>

License: MIT.

The MDTA, GDFN, pixel-unshuffle downsampling, and pixel-shuffle upsampling
designs in `sgbformer/models/layers.py` and
`sgbformer/models/backbone.py` follow the Restormer architecture.

## Bayesian Flow Networks

Copyright 2023 NNAISENSE SA.

Source: <https://github.com/nnaisense/bayesian-flow-networks>

License: Apache License 2.0.

The continuous Gaussian accuracy schedule, sender distribution, Bayesian
parameter update, and discrete-time loss in `sgbformer/models/bfn.py` follow
the equations and reference implementation of Bayesian Flow Networks.

## OpenCLIP

Source: <https://github.com/mlfoundations/open_clip>

License: MIT.

OpenCLIP is an optional runtime dependency used to load the frozen CLIP image
encoder. It is not vendored into this repository.

## SSR

Source: <https://github.com/LIDAYU-DayuLi/SSR>

The optional `ExternalSSRFoundation` loads the official SSR architecture and
checkpoint at runtime. No SSR source code or weights are included in this
repository. The upstream repository did not contain a license file when this
integration was prepared, so redistribution or modification of its source
requires permission from the SSR authors.

The released SSR checkpoint contains EDFFN parameters that do not match the
plain FFN currently present in the SSR repository. The compatibility FFN in
`sgbformer/models/foundation.py` follows the MIT-licensed EDFFN from:

<https://github.com/kkkls/EVSSM>

## MODEM

Copyright (c) 2025 Hainuo Wang.

Source: <https://github.com/hainuo-wang/MODEM>

License: MIT.

MODEM is not included as a runtime dependency. Its published checkpoint and
All-Weather results are used only as a reproducible 2025 comparison target.

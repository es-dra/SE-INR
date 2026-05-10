# Sinc Observation Property Test

This diagnostic compares the analytic normalized sinc response used by SC-INR with high-order numerical box integration on random 2D Fourier components.

- trials: `2000`
- quadrature order: `64`
- max absolute error: `5.329071e-15`
- p99 absolute error: `3.441691e-15`
- mean absolute error: `8.843487e-16`

This supports the implementation-level observation formula only; it does not prove that sinc is the sole cause of benchmark gains.

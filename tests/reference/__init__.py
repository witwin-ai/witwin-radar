"""CPU/Torch reference oracles for the radar suite.

Nothing here ships. These are independent implementations used to check the
production native kernels, and the architecture requires them to live under
``tests/`` precisely so that no production module can import or dispatch to
one.
"""

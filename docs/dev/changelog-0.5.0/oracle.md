## Added

- An analytic oracle and survival test suite under `tests/oracle/` (R-ADR-032): the radar equation off the polarization plane, sampled-surface RCS against analytic plate, sphere and trihedral cross sections, image-method ghost range, Doppler and angle of arrival, moving-target carrier-phase precision against a float64 trajectory, per-waveform SNR against kTB, wall-bounce level against the slab reflection coefficient, and a crash-and-context survival check of edge-case scenes. Known defects are pinned as strict expected failures naming their finding, so a fix that lands is detected.

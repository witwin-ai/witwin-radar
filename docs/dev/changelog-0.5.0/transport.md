# transport

## Changed

- The outbound leg of every round trip is evaluated from the receive element toward the scatter site (Lorentz reciprocity, still a separate Channel evaluation). The composed scalar co-polar numbers move by at most 1.3e-7 relative per row (float32 rounding of the two evaluation orders). `AspectScatterResponse` now accepts reflected outbound rows: it reads the site's departure bearing off the outbound leg's arrival segment.

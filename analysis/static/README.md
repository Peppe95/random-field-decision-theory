# Static analyses

This directory is for the final frozen HAB22 and Choices13k production analyses.

Please add:
- the exact RFDT-P runner and imported modules;
- HAB22 task and participant fold construction;
- participant-fit benchmark code;
- the separate pooled restriction analysis;
- Choices13k subset construction and ten-fold problem cross-validation;
- optimizer settings, seeds and parameter bounds;
- compact final result tables;
- static figure scripts and small figure inputs.

Do not add raw or participant-level third-party data, developmental model variants, or intermediate tuning scripts that are not part of the reported pipeline.

The frozen static specification is:
- signed-power utility `u_rho(x) = sign(x)|x|^rho`;
- alpha in [0,1];
- beta in [0.05,4];
- kappa in [0,0.995];
- lambda in [0.1,8];
- rho in [0.05,1.5].

HAB22 is the development benchmark. Choices13k is the frozen external static validation.

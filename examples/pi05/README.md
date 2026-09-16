# pi0.5 + ASRR

This directory defines the tensor boundary between a frozen pi0.5 runner and
ASRR. It does not import a private checkout or assume any filesystem layout.

The external runner returns a policy-native action proposal and any context it
chooses to expose:

```text
base_action:    [B, H, D]
global_context: optional [B, C]
step_context:   optional [B, H, C_step]
task_index:     optional [B]
```

`build_pi05_refiner` creates either ASRR-A (action only) or ASRR-C (available
context). `editable_action_dims` builds a coordinate mask so padding or command
dimensions can remain fixed. The wrapper returns the refined proposal in the
same action space; normalization and native execution stay in the pi0.5 runner.

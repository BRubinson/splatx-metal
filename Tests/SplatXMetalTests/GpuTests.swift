import Testing

/// Common parent for GPU-sharing tests — `.serialized` applies recursively to child suites, so
/// ProbeTests (the sustained-load probe) and SplatXTrainerTests (training E2E) never overlap
/// **with each other** either.
/// (With separate suites, even if each is serialized on its own, cross-suite parallel execution
/// remains possible and other tests' GPU load inflates the sustained probe's batch times — the
/// contamination mode described in the ProbeTests comment.)
@Suite(.serialized) struct GpuTests {}

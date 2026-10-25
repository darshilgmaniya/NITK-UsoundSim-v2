# Documentation Index - Integration 01 (Acoustic Propagation 128 el + Tissue Interaction)

## By audience
| You are | Start with | Then |
|---|---|---|
| New to the project | [01_GETTING_STARTED](01_GETTING_STARTED.md) | [02_USER_GUIDE](02_USER_GUIDE.md), [15_GLOSSARY](15_GLOSSARY.md) |
| Running simulations | [02_USER_GUIDE](02_USER_GUIDE.md) | [07_CONFIGURATION_REFERENCE](07_CONFIGURATION_REFERENCE.md), [09_PERFORMANCE](09_PERFORMANCE.md), [12_TROUBLESHOOTING_FAQ](12_TROUBLESHOOTING_FAQ.md) |
| Receiver Beamforming team | [10_RECEIVER_BF_HANDOFF](10_RECEIVER_BF_HANDOFF.md) | [06_DATA_DICTIONARY](06_DATA_DICTIONARY.md) |
| Acoustic Propagation or Tissue Interaction team | [MIGRATION_64_TO_128](MIGRATION_64_TO_128.md), [04_PHYSICS_AND_MATH](04_PHYSICS_AND_MATH.md) | [13_DECISION_LOG](13_DECISION_LOG.md) |
| Developer adding a module | [03_ARCHITECTURE](03_ARCHITECTURE.md), [11_DEVELOPER_GUIDE](11_DEVELOPER_GUIDE.md) | [05_API_REFERENCE](05_API_REFERENCE.md) |
| Reviewer / supervisor | [08_VERIFICATION_REPORT](08_VERIFICATION_REPORT.md) | [14_LIMITATIONS_AND_ROADMAP](14_LIMITATIONS_AND_ROADMAP.md), [Technical Report PDF](NITK-UsoundSim_Integration_Technical_Report.pdf) |

## All documents
| # | Document | Type | Content |
|---|---|---|---|
| 01 | [Getting Started](01_GETTING_STARTED.md) | tutorial | install, first run, expected output |
| 02 | [User Guide](02_USER_GUIDE.md) | how-to | sources, options, recipes |
| 03 | [Architecture](03_ARCHITECTURE.md) | explanation | layers, data flow, sequence, principles |
| 04 | [Physics and Math](04_PHYSICS_AND_MATH.md) | theory | equations, time axis, algorithms, 64 vs 128 |
| 05 | [API Reference](05_API_REFERENCE.md) | reference | classes, functions, errors |
| 06 | [Data Dictionary](06_DATA_DICTIONARY.md) | reference | every file and key, shape, unit |
| 07 | [Configuration Reference](07_CONFIGURATION_REFERENCE.md) | reference | every setting and how they interact |
| 08 | [Verification Report](08_VERIFICATION_REPORT.md) | report | test inventory, measured results, requirements trace |
| 09 | [Performance](09_PERFORMANCE.md) | report | timing, memory, scaling |
| 10 | [Receiver BF Hand-off](10_RECEIVER_BF_HANDOFF.md) | interface guide | what the next team receives, example code |
| 11 | [Developer Guide](11_DEVELOPER_GUIDE.md) | guide | layout, conventions, tests, adding modules, releases |
| 12 | [Troubleshooting and FAQ](12_TROUBLESHOOTING_FAQ.md) | support | symptoms, causes, fixes |
| 13 | [Decision Log](13_DECISION_LOG.md) | record | design decisions with alternatives |
| 14 | [Limitations and Roadmap](14_LIMITATIONS_AND_ROADMAP.md) | planning | limits, risks, next steps |
| 15 | [Glossary](15_GLOSSARY.md) | reference | terms |
| - | [Migration 64 to 128](MIGRATION_64_TO_128.md) | change guide | what changed, checklist |
| - | [TI and AP Interface Contract](TI_AP_INTERFACE_CONTRACT.md) | contract | original agreement between the two teams |
| - | [Changelog](../CHANGELOG.md), [Contributing](../CONTRIBUTING.md) | project | history, how to contribute |
| - | [Technical Report (PDF)](NITK-UsoundSim_Integration_Technical_Report.pdf) | consolidated | all of the above in one document |

## Module documentation (unchanged by the integration)
- Acoustic Propagation: `teams/T3_Acoustic_Propagation/README.md`
- Tissue Interaction: `teams/T4_Tissue_Interaction/README.md`, `teams/T4_Tissue_Interaction/docs/` (PHYSICS, TISSUE_TABLE, TISSUE_INTERACTION_EXPLAINED)

## Keeping documentation true
Numbers in 08 and 09 come from `integration/01_ap_ti_relay/verification/run_verification.py` (results in `docs/verification/results.json`). Re-run it and update both reports when physics or defaults change.

| 16 | [Receiver Beamforming and the full relay](16_RECEIVER_BEAMFORMING_AND_FULL_RELAY.md) | five-runner relay, baton, how to run |
| 17 | [Full pipeline: any phantom to the final image](17_FULL_PIPELINE.md) | runners 6 and 7, data contract, limits |

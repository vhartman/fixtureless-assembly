# Development tools

The planner's importable core and primary entry points stay in the repository root. Standalone
development commands are grouped here by purpose:

| Directory | Purpose |
| --- | --- |
| `calibration/` | Generate and check robot-model calibration artifacts |
| `validation/` | Check robot-model kinematics and imported Fabrica geometry |
| `exporters/` | Export assembly geometry, open-loop trajectories, and web replays |
| `visualization/` | Interactive motion-plan, witness, robot, and assembly viewers |

Run commands from the repository root. For example:

```bash
python3 tools/validation/fk_compare.py --help
python3 tools/exporters/export_assembly_parts.py --help
python3 tools/visualization/mp_viser_saved.py --help
```

The scripts add the repository root to their import path, so direct execution keeps the same
behavior as the former root-level commands.

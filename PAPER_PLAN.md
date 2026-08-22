# From the Current Planner to a Full Paper

This document turns the current project status into a possible paper scope and evaluation plan.
The recommended direction is a **search paper with calibrated end-to-end validation**. The paper
should not become primarily a controller, retiming, or contact-simulation paper: those layers
should demonstrate that the plans are executable and that the search abstractions correspond to
physical reality.

The companion [SEARCH_METHOD.md](SEARCH_METHOD.md) describes the search itself in intuitive and
paper-style form.

## 1. Recommended paper claim

A concise central claim is:

> Multi-robot assembly planning becomes tractable by solving a fact-aware disassembly relaxation
> backward, using it to construct and repair forward milestone plans, and treating stochastic
> continuous feasibility as an anytime, resumable witness-search problem.

This supports three primary technical contributions:

1. **Signature-collapsed backward search.** A disassembly relaxation incorporates dependency,
   placement, tool-capability, grasp, and wrench-support facts without retaining every raw grasp
   and robot assignment.
2. **Adaptive milestone planning.** The backward distances guide forward local fills, while
   widening, detached-module construction, and chronological repair recover from decomposition
   decisions that are infeasible in the concrete robot problem.
3. **Open continuous validation.** Nonlinear IK failure is treated as “not found with this effort
   yet” rather than as an immediate infeasibility proof. Effort is distributed across candidate
   edges, and successful joint-space witnesses are retained as exact feasibility certificates.

The milestone-specific anytime wrapper is a useful fourth component. In the current
implementation it should be described as an **incumbent-preserving improvement layer**, not as an
asymptotically optimal algorithm. The present weighted A*/PEA* engines do not reopen a concrete
state after its first feasible pop, and the labeled heuristic is not currently known to be
consistent. A stronger optimality claim would require a consistency proof, reopening, or another
transition-specific argument.

Motion planning, phase scheduling, smoothing, compliant execution, and physics simulation should
be presented as the validation pipeline around these search contributions. They are substantial
engineering components, but treating all of them as equal algorithmic contributions would make
the paper difficult to focus.

## 2. Is real execution necessary?

A planning paper can be simulation-only if it contains:

- sufficiently large and varied problem instances;
- strong baselines and ablations;
- realistic robot geometry and collision checking;
- statistically meaningful results;
- clear scaling results; and
- honest separation between kinematic, quasi-static, and dynamic claims.

This project, however, makes unusually physical decisions involving insertion force, bracing,
table support, vacuum engagement, grasp wrench limits, heterogeneous tools, and compliant skill
funnels. These naturally invite the question: **do the facts used by search predict anything
outside the planner's own model?**

For this reason, a modest real-robot evaluation would add considerably more credibility than a
large effort to make every aspect of simulation photorealistic or contact-exact. The recommended
paper combines:

1. a broad quantitative search evaluation;
2. a calibrated dynamics simulation for controlled ablations and robustness trials; and
3. a small number of convincing real executions.

The real experiments establish transfer. Simulation supplies scale, repeatability, controlled
negative examples, and uncertainty sweeps.

## 3. Recommended real-robot tasks

### 3.1 Car-mini: headline cooperative assembly

Car-mini is probably the strongest headline demonstration:

- three robots;
- simultaneous holders and supporting grasps;
- `add_grasp`, bracing, and coupled assembly rendezvous;
- significant insertion force; and
- ordering and support decisions that genuinely depend on wrench reasoning.

It demonstrates why the problem is multi-robot assembly rather than parallel pick-and-place.

### 3.2 Peg stool: heterogeneous tools and insertion geometry

The peg stool covers a complementary capability set:

- mixed vacuum and two-finger end effectors;
- broad parts naturally held by vacuum;
- repeated peg-in-hole insertions with actual concavity and chamfers;
- placement and table interaction; and
- a reusable skill funnel rather than a purely geometric mate transform.

It is a good test of heterogeneous tool allocation, grasp sampling, repeated assembly skills, and
the relationship between the search's abstract mate action and a real compliant insertion.

### 3.3 Cross: repeatability and controlled discrimination

The cross is geometrically simpler and visually easy to interpret. It is a useful platform for:

- repeated trials with perturbed initial poses;
- controlled wrench-on versus wrench-off comparisons;
- early hardware integration; and
- isolating support failures without the execution complexity of the larger assemblies.

The chair remains valuable as a difficult search and motion-planning stress test. It should not be
required for the real-robot headline result, because its many contacts, mixed tools, and long plan
make it easy for execution details to dominate the evaluation.

## 4. What “proper simulation models” should mean

The current MuJoCo path is already more than a kinematic animation. It includes articulated UR
arms, joint position servos, effort and velocity limits, payload dynamics, optional physical
cross-arm contacts, breakaway grasp constraints, force/torque monitoring, task-space skill
servos, and collision-audited phase retiming.

The important remaining idealizations are:

- acquired grasps are represented by bounded breakaway welds rather than finger-pad or suction
  contact dynamics;
- completed mates become datum welds after successful seating;
- robot, table, gripper, and contact parameters have not all been systematically identified from
  hardware;
- sensing noise, state-estimation error, delay, and calibration error are mostly absent; and
- several grasp and contact limits are modeling choices rather than measured quantities.

The goal should not be to remove every idealization. It should be to make the simulation
**calibrated, internally consistent, and explicit about which claim each layer validates**.

### 4.1 A useful fidelity hierarchy

| Layer | Principal question it validates |
|---|---|
| Kinematics | Are joint limits, reachability, geometry, and robot collisions respected? |
| Actuation | Can the arms track the reference under realistic velocity, torque, and payload limits? |
| Manipulation | Do grasp overload, table support, and insertion reaction behave consistently with the wrench model? |
| Uncertainty | Does the pipeline tolerate measured pose, friction, mass, latency, and calibration variation? |
| Real execution | Does the complete modeling and control stack transfer to hardware? |

This hierarchy also makes the mate-weld assumption precise. A mate weld validates the dynamics of
the assembled relation **after** a seating detector succeeds. It does not by itself validate the
contact-rich insertion leading to that state. A real peg-in-hole execution, or a separately
validated insertion simulation, must carry that claim.

## 5. Simulation work worth doing

### 5.1 Freeze one physically consistent paper configuration

All reported validation runs should use a single production configuration with:

- articulated, actuated arms;
- unscaled robot collision geometry (`--coll-scale 1`);
- physical part masses (`--mass-scale 1`);
- physical cross-arm contacts enabled;
- complete path collision and joint-limit preflight;
- the same relevant geometry in search, motion planning, and MuJoCo; and
- one controller parameter set rather than assembly-specific tuning.

Artifacts produced with shrunken collision models remain useful diagnostics, but they should not
be counted as paper evidence for the unmodified robot geometry.

### 5.2 Calibrate a small set of primitive models

Useful hardware calibration experiments include:

- free-space tracking error at several speeds and payloads;
- controller settling time;
- table friction;
- two-finger pull-out, lateral slip, and rotational slip limits;
- vacuum pull-off, lateral slip, and peel-torque limits;
- insertion force versus depth and lateral/angular misalignment; and
- payload-induced grasp and joint tracking error.

These measurements should set the simulator and wrench-model parameters. The parameters should be
frozen before running the final assembly experiments. A simple measured model is more defensible
than a sophisticated uncalibrated one.

### 5.3 Add measured uncertainty distributions

Robustness trials should perturb:

- initial part pose;
- grasp-frame and hand-eye calibration;
- table height and orientation;
- part mass and table/contact friction;
- actuator delay or bandwidth; and
- insertion-axis alignment.

Where possible, perturbation ranges should be estimated from calibration data. The output should
be a success probability and failure taxonomy rather than a single deterministic rollout.

### 5.4 Keep grasp dynamics proportional to the claim

For the main evaluation, calibrated breakaway constraints are defensible as bounded grasp
interfaces. A stronger optional model could use physical finger pads during acquisition and
require vacuum engagement to satisfy contact, orientation, force, and seal-support conditions.
After successful acquisition, a calibrated bounded constraint remains sufficient for testing
whether planned loads exceed the grasp capability.

Continuous leakage, cup deformation, and detailed elastomer contact are probably unnecessary
scope unless suction modeling becomes a primary contribution.

### 5.5 Scope mate dynamics around the skill funnel

For peg-in-hole assembly, the execution contract should be:

1. reach a collision-free funnel-mouth configuration;
2. perform a compliant, force-guarded insertion;
3. detect success from seating depth and contact behavior; and
4. activate the rigid assembled relation only after seating succeeds.

The final rigid relation can represent a seated or fastened connection. It must never snap a part
from outside the funnel into the assembled pose.

## 6. Search benchmark design

Real execution alone does not establish the search contribution. The paper needs a reproducible
benchmark matrix that separates high-level guidance, continuous validation, and final motion
planning.

### 6.1 Assemblies

The existing set should include at least:

- Cross;
- Car-mini;
- Cube;
- Chair; and
- Peg stool.

It should be supplemented by scalable synthetic families, for example:

- chains with increasing numbers of insertions;
- branching dependency trees;
- detachable modules followed by a module join;
- repeated high-force mates requiring different brace configurations;
- heterogeneous-tool variants; and
- placement-heavy instances in which in-hand reorientation is unavailable.

The benchmark should vary:

- part count beyond the current five-to-eight-part demonstrations;
- two to four robots;
- homogeneous versus mixed grippers;
- grasp-database size;
- insertion load and noise;
- allowed placements and handovers; and
- randomized initial scatter and robot configuration.

### 6.2 Baselines and ablations

At minimum, compare:

- flat concrete search;
- a count or unlabeled heuristic;
- raw labeled backward search;
- signature-collapsed backward search;
- milestone search without widening or repair;
- the complete milestone method;
- legacy terminal stochastic IK failure;
- open witness validation; and
- the milestone incumbent before and after the bounded global improvement pass.

Signature ablations should independently evaluate:

- `press`;
- `pickface`;
- `contact`;
- `solohold`;
- `braceside`;
- all components together;
- identity collapse; and
- raw grasp labels.

The mixed-tool experiments must report whether `--mixed-wrench-h` is enabled. Without it,
per-tool capacity masks remain active but the expensive per-kind wrench signature facts fall back
to the faster topological abstraction.

### 6.3 Metrics

Record at least:

- success rate under fixed wall-clock and memory budgets;
- time to first validated plan;
- incumbent action cost over time;
- final action cost;
- concrete and abstract state counts;
- backward-table construction time and memory;
- nonlinear IK calls, solves, failures, and wall time;
- open-witness effort tiers and reopenings;
- milestone failures, repairs, modules, and window widening;
- exact witness coverage;
- motion-planning success; and
- peak resident memory.

Every stochastic comparison should use multiple search and IK seeds and report uncertainty, not
only a single paired timing. Current development measurements are useful hypotheses and regression
evidence, but are not yet statistical paper results.

## 7. Real-world experimental protocol

For each real assembly, a reasonable target is approximately ten trials spanning randomized
initial-pose and calibration perturbations. Record:

- complete task success;
- successfully completed assembly steps;
- human interventions;
- insertion attempts and retries;
- planning and execution time;
- measured peak force and torque;
- final assembly error;
- minimum observed clearance; and
- force-guard, servo, or recovery activations.

The strongest protocol would be:

1. Calibrate physical parameters using primitive manipulation tests and, if needed, one
   development assembly.
2. Freeze the planner, wrench, simulation, and controller parameters.
3. Treat Cross, Car-mini, and Peg stool as held-out evaluations.

Negative wrench ablations are useful, but deliberately toppling or dropping hardware is not
necessary. Unsafe negative cases can be performed in simulation and supported by safe primitive
hardware measurements of the relevant grasp or support limit.

## 8. Suggested paper structure

1. **Introduction and motivating failures.** Why multi-robot assembly needs joint reasoning over
   ordering, support, tools, grasps, and continuous feasibility.
2. **Related work.** Multi-robot TAMP, assembly planning, abstraction heuristics, and lazy
   continuous validation.
3. **Problem formulation.** Discrete actions, contacts, costs, continuous witnesses, and goals.
4. **Signature-collapsed backward abstraction.** Facts, inverse transitions, and admissibility.
5. **Forward milestone construction.** Local fills, widening, modules, and repair.
6. **Open-witness scheduling and anytime improvement.** Effort semantics, persistence, and
   incumbent handling.
7. **Continuous planning and execution pipeline.** What is certified and what remains an
   execution assumption.
8. **Search evaluation.** Scaling, baselines, ablations, time-to-first-solution, and memory.
9. **Calibrated simulation and real-robot experiments.** Robustness, transfer, and physical
   failure taxonomy.
10. **Limitations.** Search completeness, stochastic IK, contact idealizations, calibration, and
    generalization to new assembly skills.

### 8.1 Key figures and tables

- A system overview connecting the backward abstraction to forward witnessed fills.
- A small example of backward disassembly states and grasp-signature collapse.
- Time-to-first-solution and incumbent-cost curves.
- Success rate and resource use versus instance size.
- Distribution of IK effort for legacy versus open witness validation.
- Ablation results for signature facts and milestone mechanisms.
- Photographs or synchronized frames of the real assemblies.
- A table separating search, motion-planning, simulation, and real-execution outcomes.

## 9. Recommended work sequence

1. Freeze the algorithmic claim and explicitly keep phase smoothing outside the main search
   contribution.
2. Build a reproducible benchmark runner that emits aggregate search statistics and complete run
   metadata.
3. Add scalable synthetic assembly families so the evidence is not limited to handcrafted
   demonstrations.
4. Establish the single strict, unscaled simulation configuration.
5. Run primitive hardware calibration and freeze the physical parameters.
6. Execute Cross first as the hardware integration and repeatability test.
7. Execute Car-mini and Peg stool as the two principal physical demonstrations.
8. Run the complete benchmark, ablation, and uncertainty matrices after parameters are frozen.

## 10. Scope boundary

The largest risk is scope creep. A full suction simulator, a new MPC controller, perfect mesh
contact, and a new search algorithm could each become a separate project. For this paper,
realistic execution should validate the planner's abstractions and assumptions rather than become
the primary technical subject.

The minimum convincing paper is therefore:

- a clearly stated and theoretically qualified search method;
- a broad, statistically sound search evaluation;
- one strict calibrated simulation configuration;
- real Cross integration; and
- successful Car-mini and Peg-stool demonstrations using frozen parameters.

If real execution is unavailable, the fallback is a simulation-only planning paper with stronger
scaling families, more baselines, measured uncertainty sweeps, and explicit limitations on all
physical-feasibility claims.

# Code order and development logic

This document explains the development order of the scripts in this repository and the reason each stage was added.

The project did not begin directly as a network-science pipeline. It started as a synthetic forward-model framework for sequential atrial mapping, then gradually evolved into a multiplex network analysis project aligned with the final Network Science assignment.

---

## Stage 1: Early exploration

The scripts in `src/01_exploration/` document the step-by-step construction of the synthetic environment and the modelling logic that later supported the final pipeline.

These scripts are not the final reproducible pipeline, but they are important because they preserve the developmental history of the project.

### `code1.py` to `code12.py`

These scripts were used to explore and validate intermediate parts of the framework, including:

- patch geometry,
- sequential trajectory generation,
- hidden substrate construction,
- spontaneous and paced feature generation,
- patch-level descriptors,
- global masking and hotspot logic,
- multilayer graph ideas,
- sensitivity checks,
- synthetic ablation reasoning,
- intermediate sanity checks and visual inspection.

### Why this stage mattered

This stage was necessary because the project could not responsibly move to a multiplex network analysis before the underlying synthetic world had been defined clearly. In other words, the network layer was not the starting point; it emerged only after the synthetic observation process became sufficiently structured and interpretable.

This stage established the forward logic:

1. define a hidden substrate,
2. move a small sensing patch across it,
3. generate local observations under spontaneous and paced conditions,
4. extract interpretable local features,
5. prepare these local observations for later network construction.

---

## Stage 2: Plan A forward model

The second stage produced the main synthetic forward-model backbone used by the final project.

Files:
- `src/02_plan_a_forward/code13.py`
- `src/02_plan_a_forward/code15.py`

### `code13.py`

**Purpose**  
Generate the synthetic substrate and compute electrode-level feature tables for spontaneous and paced observations.

**What it does**
- defines the spatial domain,
- generates hidden source, block, low-voltage, and vulnerability regions,
- moves a 4×4 patch sequentially across the domain,
- computes electrode-level features such as activation-related and morphology-related quantities,
- aggregates these observations into placement-level descriptors,
- exports the main feature tables needed by later stages.

**Why it was added**  
The project needed a stable forward-model script that could produce the core synthetic dataset in a reproducible and interpretable way. Earlier scripts were exploratory, but `code13.py` became the main structured generator for the final framework.

**Why it matters scientifically**  
This script defines the hidden ground-truth world that later network analysis tries to interpret indirectly. Without this script, the rest of the project would have no controlled substrate or reproducible feature tables.

---

### `code15.py`

**Purpose**  
Generate synthetic waveform examples from the feature tables produced by `code13.py`.

**What it does**
- reads spontaneous and paced feature tables,
- constructs synthetic unipolar waveforms,
- derives synthetic bipolar examples,
- produces waveform plots and supporting metadata,
- links abstract feature values to signal-like behaviour.

**Why it was added**  
The project needed to show that the forward model was not only generating tabular descriptors but also producing signal-level consequences. This made the synthetic framework more interpretable and helped connect the feature tables to electrophysiological intuition.

**Why it matters scientifically**  
This script strengthens the forward model by showing how feature-level differences may appear in waveform morphology. It supports interpretation, but it is not itself the main network-science contribution.

---

## Stage 3: Transition from modelling to network science

At this point, the project changed emphasis.

Plan A had already produced:
- sequential patch observations,
- spontaneous and paced feature tables,
- patch-level descriptors,
- a local-to-global forward framework.

However, the assignment required a genuine Network Science analysis. Therefore, the project shifted from asking only:

> “Where are the abnormal regions?”

to asking:

> “What role does each observed patch play in a multiplex observation network?”

This led to Plan B.

---

## Stage 4: Plan B multiplex network construction

Files:
- `src/03_plan_b_network/CodeB1.py`

### `CodeB1.py`

**Purpose**  
Construct the synthetic multiplex network from sequential patch observations.

**What it does**
- converts patch placements into network nodes,
- defines a spontaneous layer and a paced layer,
- constructs intra-layer connectivity using spatial, temporal, and feature-similarity information,
- introduces inter-layer identity coupling,
- builds the supra-adjacency structure used in later analysis.

**Why it was added**  
The project needed an explicit network representation that aligned with the assignment brief. Before `CodeB1.py`, the project had a synthetic mapping framework, but not yet a formal multiplex graph.

**Why it matters scientifically**  
This is the script where the project becomes recognisably a Network Science project. It transforms sequential local mapping observations into a two-layer observation network.

---

## Stage 5: Initial node-role analysis

Files:
- `src/03_plan_b_network/CodeB2.py`

### `CodeB2.py`

**Purpose**  
Compute initial network metrics and define the first node-role interpretations.

**What it does**
- calculates node strength,
- calculates clustering coefficients,
- identifies core-like regions,
- computes signed discordance between spontaneous and paced importance,
- computes an initial bridge-like metric based on multiplex topology,
- visualises node-role maps.

**Why it was added**  
Once the multiplex graph had been built, the next question was not merely structural description, but role analysis. This script was introduced to ask which nodes were stable, which were condition-sensitive, and which appeared important for connectivity.

**Why it matters scientifically**  
This script establishes the first full node-role vocabulary of the project:
- core nodes,
- discordant nodes,
- bridge-like nodes.

It is the first direct answer to the assignment’s requirement for meaningful network analysis.

---

## Stage 6: AF-aware refinement

Files:
- `src/03_plan_b_network/CodeB25.py`

### `CodeB25.py`

**Purpose**  
Refine the multiplex formulation using AF-aware assumptions.

**What it does**
- modifies the weighting logic of the graph,
- introduces recency-sensitive interpretation,
- incorporates a stronger notion of phenotype transition,
- refines bridge-related interpretation beyond pure topology,
- produces updated node-role maps.

**Why it was added**  
The initial bridge formulation in `CodeB2.py` was mainly topological. However, the scientific meaning of the project required something closer to electrophysiological interpretation: newer observations may matter more, phenotype transitions may matter more, and bridge-like importance may not be purely structural.

**Why it matters scientifically**  
This script moves the project closer to its domain-specific meaning. It acknowledges that in sequential mapping, nodes are not just graph points; they are time-separated observations of a continuing hidden substrate.

---

## Stage 7: Boundary-primary bridge refinement

Files:
- `src/03_plan_b_network/CodeB26.py`

### `CodeB26.py`

**Purpose**  
Develop a stronger bridge formulation centred on phenotype boundaries.

**What it does**
- introduces a boundary-primary bridge metric,
- compares boundary-driven and topology-driven bridge concepts,
- defines a hybrid bridge formulation,
- evaluates how bridge interpretation changes under different assumptions.

**Why it was added**  
The earlier AF-aware bridge still mixed topology and phenotype transition in a way that needed clearer separation. `CodeB26.py` was added to ask a more precise question:

> Should a bridge node be defined mainly by topological shortest-path importance, or mainly by its position at a phenotype boundary?

**Why it matters scientifically**  
This is one of the most conceptually important steps in the project. It shows that bridge-ness is not a single universal idea; it depends on what kind of transition the investigator cares about:
- structural mediation,
- phenotype boundary transition,
- or a hybrid of both.

---

## Stage 8: Null model and sensitivity analysis

Files:
- `src/03_plan_b_network/CodeB3.py`

### `CodeB3.py`

**Purpose**  
Test whether the observed node-role findings are robust and non-trivial.

**What it does**
- constructs null comparisons,
- measures empirical correlation against null distributions,
- evaluates ranking stability under parameter variation,
- performs bridge-mix sensitivity analysis,
- tests whether the main node-role findings depend strongly on design choices.

**Why it was added**  
Without null models and sensitivity analysis, the project would risk becoming a descriptive visualisation exercise. This script was added to provide statistical and methodological support for the observed node-role interpretations.

**Why it matters scientifically**  
This script makes the project more defensible. It addresses the question:
- are the results meaningful,
- or are they artefacts of arbitrary graph construction choices?

This is the stage that most directly supports the credibility of the final report.

---

## Final main pipeline

The final reproducible pipeline used in the report is:

1. `code13.py`
2. `code15.py`
3. `CodeB1.py`
4. `CodeB2.py`
5. `CodeB25.py`
6. `CodeB26.py`
7. `CodeB3.py`

This order reflects the logic of the project:
- first define the synthetic world,
- then generate observations,
- then construct the multiplex network,
- then analyse node roles,
- then refine bridge meaning,
- then test robustness.

---

## Summary of development logic

In short, the project evolved through three major conceptual phases:

### Phase 1: Synthetic world construction
Build a controlled hidden substrate and define the sequential observation process.

### Phase 2: Forward modelling and signal interpretation
Generate features and waveform examples that make the synthetic world interpretable.

### Phase 3: Multiplex network analysis
Reframe sequential observations as a two-layer network and identify stable, discordant, and bridge-like node roles.

The final repository preserves this full progression because the network analysis did not arise in isolation; it emerged from a longer process of synthetic modelling, interpretation, and methodological refinement.

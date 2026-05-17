# Synthetic Multiplex AF Network Project

This repository contains the code, selected figures, representative outputs, and final report for a synthetic atrial mapping project developed for a Network Science assignment. The project began as a forward synthetic modelling framework for sequential atrial mapping and later evolved into a multiplex network analysis pipeline designed to identify stable, condition-dependent, and bridge-like regions from spontaneous and paced observations.

## Project overview

The central motivation of this project is that sequential local mapping does not observe the full atrial substrate at once. Instead, a small sensing patch samples a limited region of a larger hidden system, then moves to a new position after a time delay. The underlying substrate continues evolving while the observation window changes. This creates a local-to-global inference problem: how can multiple temporally separated local observations be integrated into a meaningful global interpretation?

To study this question in a controlled setting, the project uses a synthetic two-dimensional atrial substrate with known hidden structure. A 4×4 electrode patch is moved sequentially across the domain, and both spontaneous and paced observations are generated. These observations are then analysed in two stages:

1. **Plan A** constructs the synthetic substrate, electrode-level features, and waveform examples.
2. **Plan B** converts sequential patch observations into a multiplex network and applies network-science analysis to identify node roles.

## Research question

Can paced propagation dynamics act as a probe for revealing hidden conduction abnormalities in a synthetic multiplex atrial mapping network, and which patch locations behave as structurally stable, condition-dependent, or bridge-like regions?

## Why a synthetic dataset was used

This project does not use a clinical dataset. Instead, it generates a synthetic mapping dataset with known hidden ground truth. This choice was made for methodological reasons:

- it allows the hidden substrate to be known exactly,
- it makes forward assumptions explicit,
- it supports controlled testing of node-role metrics,
- it avoids overclaiming clinical validity.

The project is therefore framed as a **methodological network-science investigation**, not as a clinically validated electrophysiology study.

A fuller description of the dataset is provided in `docs/dataset_card.md`.

## Main idea of the project

The project treats each patch placement as an information-bearing observation of a larger hidden substrate. Two observation conditions are used:

- **Spontaneous mapping**, which reflects naturally emerging activity patterns.
- **Paced mapping**, which introduces a controlled perturbation and may reveal latent conduction abnormalities more clearly.

These two observation modes are represented as two layers of a multiplex network. The resulting framework allows the project to ask three different questions:

- Which regions look abnormal in **both** conditions?
- Which regions behave differently between conditions?
- Which regions occupy an important **bridging** role in the multiplex structure?

## Repository structure

- `src/01_exploration/` – early exploratory scripts used to build the synthetic mapping framework step by step
- `src/02_plan_a_forward/` – forward-model scripts for synthetic substrate generation, feature construction, and waveform synthesis
- `src/03_plan_b_network/` – multiplex network construction, node-role analysis, refinement, null models, and sensitivity analysis
- `docs/` – project documentation
- `docs/figures/` – selected figures illustrating the main results
- `report/` – final report files
- `sample_outputs/` – representative output tables and summaries from the main stages of the pipeline

## Code organisation

This repository contains both:

1. **Historical exploratory scripts** documenting the development of the framework step by step.
2. **The main reproducible pipeline** used for the final report.

The detailed development logic and code order are described in `docs/code_order.md`.

## Main reproducible pipeline

The final project pipeline is based on the following scripts:

1. `code13.py`  
   Builds the synthetic substrate and generates electrode-level spontaneous and paced feature tables.

2. `code15.py`  
   Generates synthetic waveform examples from the feature tables, including unipolar and bipolar traces.

3. `CodeB1.py`  
   Constructs the multiplex network from sequential patch observations.

4. `CodeB2.py`  
   Computes initial node-role metrics such as strength, clustering, core score, discordance, and bridge-like behaviour.

5. `CodeB25.py`  
   Refines the multiplex graph using AF-aware weighting and recency-sensitive assumptions.

6. `CodeB26.py`  
   Introduces refined bridge formulations, including boundary-primary and hybrid bridge metrics.

7. `CodeB3.py`  
   Performs null-model analysis and parameter sensitivity testing.

## Exploratory development scripts

The scripts in `src/01_exploration/` document the earlier stages of the project. These scripts were used to develop and validate the geometry, trajectory, hidden substrate construction, patch-level descriptors, masking behaviour, ablation utility, and intermediate sanity checks before the final pipeline was stabilised.

They are included to preserve the development history of the project, but they are not required for reproducing the final reported results.

## Dataset summary

The synthetic dataset represents a two-dimensional atrial-like substrate containing hidden structural and dynamical components. The domain includes:

- two source regions,
- one block region,
- one low-voltage region,
- one vulnerability region.

A 4×4 electrode patch is sequentially placed across the domain. At each placement, electrode-level measurements are generated under spontaneous and paced conditions. These local observations are later aggregated into patch-level quantities and used as nodes in a multiplex network.

The dataset is synthetic, controlled, and reproducible through fixed random seeds.

More detail is provided in `docs/dataset_card.md`.

## Network-science contribution

The network-science part of the project begins when patch placements are interpreted as network nodes rather than merely local signal snapshots. Two multiplex layers are constructed:

- a spontaneous layer,
- a paced layer.

Edges are defined using a combination of spatial proximity, temporal continuity, and phenotype similarity. This allows the project to examine not only the severity of abnormality, but also the **role** of a patch within the evolving observation network.

The main node-role concepts used in the project are:

- **Core score**: regions that remain important across both spontaneous and paced conditions
- **Signed discordance**: regions whose behaviour changes between the two conditions
- **Bridge-like metrics**: regions that occupy structurally or phenotypically important transition positions in the multiplex graph

## Main findings

At a high level, the project found that:

- core-like nodes align most strongly with stable block-related structure,
- discordant nodes reflect condition sensitivity rather than a simple hidden-field surrogate,
- bridge formulations depend strongly on whether one prioritises topology, phenotype transition, or a hybrid of both,
- paced observations are useful as a probe, but the information they reveal depends on how the multiplex graph is constructed and weighted.

## Selected figures and outputs

This repository includes selected final figures in `docs/figures/` and representative outputs in `sample_outputs/`. These are included to illustrate the main results without storing every intermediate generated file.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Dependencies:

The project mainly uses:

numpy
pandas
matplotlib
networkx
scipy
python-docx

How to run:
Run the main scripts in this order:
python src/02_plan_a_forward/code13.py
python src/02_plan_a_forward/code15.py
python src/03_plan_b_network/CodeB1.py
python src/03_plan_b_network/CodeB2.py
python src/03_plan_b_network/CodeB25.py
python src/03_plan_b_network/CodeB26.py
python src/03_plan_b_network/CodeB3.py

Reproducibility:

The project uses fixed seeds and deterministic synthetic generation rules where possible. The goal is to make the main results reproducible from the provided code structure rather than dependent on hidden manual processing.

Limitations:

This repository presents a synthetic methodological framework. Its main limitations are:

the substrate is synthetic rather than clinical,
the waveform model is illustrative rather than biophysically complete,
node roles depend on graph design choices,
the framework supports inverse-problem reasoning only indirectly unless paired with dedicated recovery analysis.

The project should therefore be interpreted as a controlled Network Science study of a multiplex mapping framework, not as a clinically validated cardiac mapping tool.

Report:

The final written report is stored in the report/ directory.

Author:

Sara Yasinian

Licence:

This repository is shared for academic and educational purposes.



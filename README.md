# Food-Allergy-Biomarker-Replication
Computational investigation of molecular signatures associated with food allergy using public epigenetic and transcriptomic datasets.

## Research Question:
Which molecular signatures distinguish food allergy, and how robust and informative are they across datasets? Can they support effective classification?

## Approach:
The project evaluates candidate molecular signatures through several complementary analyses:

- Differential DNA methylation analysis
- Cross-cohort comparison
- Candidate gene and DMR identification
- Transcriptomic analysis
- ISG / interferon-related signatures
- Classification of allergic vs. control samples
- Pathway-level interpretation

This project did not stop at statistical significance but considered signal strength, predictive performance, replication, biological coherence, all while maintaining rigor by identifying and removing common sources of data leakage in the computational food allergy space.

## Current datasets
GSE114134 — infants (~12 months), n = 59 (39 allergic / 20 control)

GSE189148 — adolescents (10–15 years, resting/unstimulated), n = 43 (30 allergic / 13 control)

Both provide publicly available molecular data relevant to food allergy and are used to investigate whether candidate signals persist across independent cohorts.

## Repository
data/        Dataset information
scripts/     Analysis scripts
notebooks/   Exploratory analyses
results/     Figures and outputs

## Results

### Classification
Within-cohort (GSE114134):	AUC 0.532, p-value 0.365 (~chance, not significant)

GSE114134 → GSE189148: AUC	0.498, p-value	0.535 (~chance, not significant)

GSE189148 → GSE114134: AUC	0.590, p-value	0.367 (little more than chance, not significant)

All valid classifiers were non-significant.

A separate fixed-feature analysis produced AUC = 0.994, but the features were selected from the same cohort in which they were evaluated. This circular analysis is retained only as a demonstration of selection bias and is not considered a valid result.

### Signals
Two adjacent CpG probes upstream of ISG15, cg08469540 and cg25610492, showed direction-consistent hypomethylation in allergic samples across both cohorts.

The probes also formed a replicated two-probe region in bumphunter analysis.

The corresponding region showed:

GSE114134: p = 0.0175
GSE189148: p = 0.00589

However, the region did not survive family-wise error correction (FWER = 1 in both cohorts). The individual probes likewise do not independently establish a statistically significant finding in the adolescent cohort.

In the companion RNA-seq cohort (GSE189149), the probes were associated with decreased ISG15 expression (log₂FC ≈ −0.926), closely matching the published estimate (−0.931).

The main evidence is therefore cross-cohort directional concordance, rather than definitive statistical validation.

## Status
Active research — 2026

The next stage is to expand this analysis into a Food Allergy Biomarker Atlas, systematically evaluating published and computationally identified signatures for replication, predictive performance, and biological coherence.

Author: Zamin Rizvi

Candidate signatures identified here are exploratory and are not clinically validated.


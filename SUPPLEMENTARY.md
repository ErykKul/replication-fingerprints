# Supplementary material: the four lens distillation prompts

Each paper is distilled once by Anthropic Claude 3 Haiku (`claude-3-haiku-20240307`), temperature 0, into four
orthogonal lenses. Each lens is a separate instruction; all are domain-stripped (topic, field, and application
words removed so that the transferable structure remains). Controlled facets take one lowercase value from the
listed vocabulary; free-text fields are one to a few sentences. The bag-of-words Naive Bayes readout is applied
to the concatenation of a lens's fields (the "multi-lens union" concatenates all four lenses).

---

## Lens 1 — computational

> STRUCTURED COMPUTATIONAL FINGERPRINT. Describe the paper's computational core so that two papers from
> different fields using the SAME computation agree on the mechanism facets. If the paper has no computational
> core (qualitative / position / descriptive), set the structural facets to `none` and give a one-sentence
> mechanism. Fields:
> - `mechanism`: 6-12 sentence GENERIC-math skeleton; strip all domain/application words and famous method names.
> - `domain`: 3-8 words.
> - `structure`: dense linear algebra | sparse linear algebra | spectral or transform | N-body or all-pairs |
>   structured grid | unstructured mesh | map-reduce or embarrassingly-parallel | combinational logic | graph
>   traversal | dynamic programming | backtracking or branch-and-bound | graphical models | finite-state
>   machine | other:<> | none
> - `data_object`: dense matrix or tensor | sparse matrix | grid or lattice | mesh | graph or network | point
>   set | sequence or time-series | tree or hierarchy | set or table | continuous function or field | none
> - `inference`: deterministic or closed-form | frequentist point estimate | Bayesian posterior | variational |
>   sampling or Monte-Carlo | bootstrap or resampling | optimization only | none
> - `problem_form`: estimation | prediction or classification | optimization | decision or test | search |
>   counting | simulation or generation | proof or characterization | control | ranking or retrieval
> - `distribution`: outcome + assumed distribution, or none
> - `complexity`: closed-form | polynomial iterative | combinatorial or NP-hard | consistency | finite-sample
>   bound | convergence rate | regret bound | not stated
> - `data_availability`: dataset-with-doi-or-handle | dataset-in-repository | public-benchmark-used |
>   data-on-request | proprietary | none
> - `code_availability`: public-repository | on-request | none
> - `preregistration`: registered-report | preregistered | analysis-plan-stated | none
> - `evidence_basis`: empirical-with-released-data | empirical-with-private-data | simulation-study |
>   mathematical-proof | reanalysis-of-existing-data | review-or-position

## Lens 2 — experiment (design and rigor)

> PSYCHOLOGY EXPERIMENT FINGERPRINT. Describe WHAT the study did and HOW rigorously, domain-stripped, so that
> two studies testing the same kind of effect agree regardless of topic. Fields:
> - `what_manipulated`: the independent variable(s), topic words removed (one phrase).
> - `what_measured`: the dependent variable(s), topic words removed (one phrase).
> - `design`: between-subjects | within-subjects | mixed | correlational | longitudinal | field | quasi-experiment
> - `manipulation`: randomized-experiment | measured-individual-difference | natural-manipulation | none
> - `sample_basis`: convenience | student | online-panel | representative | clinical | other
> - `rigor`: minimal | standard | controls-and-power-reported | preregistered-with-power
>   (assign the highest level ONLY if the paper explicitly states it; quote-or-downgrade, do not infer power
>   from covariates or counterbalancing).
> - report the stated sample size, effect size, and p-value if present (else `not-stated`).

## Lens 3 — finding (the claim)

> Extract the paper's main empirical FINDING as a single domain-stripped, conceptual claim: the direction and
> what-affects-what, with the specific topic, population, and construct words removed and only the conceptual
> structure kept. One or two sentences.

## Lens 4 — qualitative (interpretive)

> QUALITATIVE / INTERPRETIVE FINGERPRINT (the meaning layer), domain-stripped. Fields:
> - `phenomenon`: one phrase for the thing being studied, generalized.
> - `constructs`: 2-4 space-separated construct labels.
> - `claim`: one sentence, the interpretive claim.
> - `narrative`: counterintuitive-reversal | new-mechanism | boundary-condition | context-effect |
>   incremental-extension | confirmation
> - `scope`: broad-generalization | specific-context | single-population
> - `stance`: bold-claim | measured-claim | hedged-claim

---

*The distillation code, the four fingerprint corpora, and all analysis scripts are released with the paper.*

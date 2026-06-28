# LLM Prompt Feature Distribution Analysis

This report audits the numerical EEG features that were actually inserted into the LLM prompts. It focuses on distribution, class overlap, and neuroscience-rule consistency rather than on re-running model training.

## What was analyzed

- Input path: held-out test set, LLM preprocessing, `window_size=1000`, `step_size=500`, channel-wise z-score normalization.
- Number of prompt windows analyzed: **4,017**.
- True class counts: negative=1,323, neutral=1,308, positive=1,386.
- Feature schema: 18 prompt features grouped into global spectral ratios, frontal asymmetry, regional activity, complexity, and derived ratios.

## LLM behavior to explain

- Window accuracy: **38.11%**.
- Trial accuracy: **44.44%**.
- At trial level, the LLM prediction distribution was negative: 0 (0.0%), neutral: 34 (75.6%), positive: 11 (24.4%).

The trial-level confusion matrix shows the most important failure mode: the LLM does not recover a reliable negative class. This matters because the prompt asks the model to reason from neuroscience-inspired indicators, but the feature distributions below show that those indicators are not cleanly class-separated in the values the model receives.

## Main distribution finding

Most prompt features have substantial overlap across negative, neutral, and positive windows. The strongest features by eta-squared still explain only a limited part of label variance, so the prompt gives the LLM ambiguous evidence rather than a rule-like mapping from features to emotion.

![Standardized feature distributions](standardized_feature_distributions_by_class.png)

![Feature separability](feature_class_separability_eta_squared.png)

Top features by class separability:

| Feature | Group | eta-squared | positive vs negative d | negative median | neutral median | positive median |
|---|---|---:|---:|---:|---:|---:|
| gamma_ratio | Global spectral state | 0.4796 | 1.504 | 0.1414 | 0.1222 | 0.2123 |
| gamma_beta_ratio | Derived ratios | 0.4431 | 1.458 | 0.3527 | 0.3258 | 0.5205 |
| alpha_ratio | Global spectral state | 0.3269 | -0.934 | 0.2361 | 0.3134 | 0.1711 |
| beta_alpha_ratio | Derived ratios | 0.3013 | 0.901 | 1.724 | 1.224 | 2.447 |
| temporal_alpha | Regional activity | 0.1950 | -0.369 | 0.02779 | 0.04809 | 0.02196 |
| occipital_alpha | Regional activity | 0.0869 | -0.100 | 0.02989 | 0.0466 | 0.02826 |

Least label-sensitive features:

| Feature | eta-squared | Interpretation risk |
|---|---:|---|
| frontal_beta | 0.0041 | Low distributional separation; weak evidence for zero-shot prompt reasoning. |
| frontal_occipital_alpha_ratio | 0.0013 | Low distributional separation; weak evidence for zero-shot prompt reasoning. |
| activity | 0.0008 | Low distributional separation; weak evidence for zero-shot prompt reasoning. |
| entropy | 0.0005 | Low distributional separation; weak evidence for zero-shot prompt reasoning. |

## FAA interpretation issue

Frontal Alpha Asymmetry needs careful wording in the paper because the implementation convention is:

`FAA = log(left alpha power) - log(right alpha power)`

In many affective-neuroscience interpretations, alpha power is treated as inversely related to cortical activation. Under that interpretation, a positive value in this implementation does not directly mean stronger left frontal activation. It can instead mean stronger left alpha power, which may correspond to weaker left activation. Therefore, using these values as direct evidence of left-dominant positive affect would be risky unless the sign convention is explicitly corrected or justified.

![FAA sign rates](faa_positive_sign_rate_by_class.png)

FAA separability summary:

| FAA feature | eta-squared | negative median | neutral median | positive median |
|---|---:|---:|---:|---:|
| faa_f3_f4 | 0.0248 | 0.1837 | -0.1317 | 0.1836 |
| faa_fp1_fp2 | 0.0169 | -0.08353 | -0.141 | -0.08453 |
| faa_f7_f8 | 0.0056 | 0.8525 | 0.7681 | 0.8603 |

## Prompt-rule diagnostics

The prompt expects positive emotion to be supported by organized frontal engagement, stronger frontal/posterior balance, and weak posterior inhibition. The aggregated diagnostic scores do not form a clean monotonic pattern. Mean prompt-positive evidence was:

- Negative windows: **-0.078**
- Neutral windows: **-0.956**
- Positive windows: **0.591**

If positive evidence is not clearly highest for positive windows and clearly lowest for negative windows, the LLM has no stable numerical basis for applying the intended neuroscience logic. It may then default to broad semantic priors from the prompt, especially neutral when evidence is mixed.

![Prompt rule diagnostics](prompt_rule_diagnostics_heatmap.png)

## Discussion text for the paper

The LLM-based classifier underperformed because the feature representation supplied to the language model did not provide sufficiently separable or conventionally interpretable evidence for the three emotion classes. Although the prompt described plausible EEG-emotion relationships, the numerical values inserted into the prompt were highly overlapping across negative, neutral, and positive trials. This means that the LLM was not receiving a clear statistical signal from which the stated neuroscience rules could be applied consistently.

This is especially visible for frontal asymmetry features. FAA is often discussed in relation to emotional valence and approach-withdrawal tendencies, but its interpretation depends strongly on preprocessing, referencing, channel selection, frequency-band definition, and sign convention. In this implementation, FAA is computed as left alpha power minus right alpha power in log space. Because alpha power is commonly interpreted as inversely related to cortical activation, this convention cannot be directly read as left frontal activation dominance without an additional sign interpretation. As a result, apparent positive FAA values may not support positive affect in the way a simple reading of the prompt suggests.

The feature-distribution audit therefore suggests that the LLM failure is not only a prompt-engineering problem. The model was asked to perform symbolic reasoning over numerical EEG summaries whose class-conditional distributions were weakly separated and partly ambiguous. A supervised ML or DL model can learn empirical class boundaries from these values, but a zero-shot LLM has no fitted mapping between this dataset's feature distributions and the target labels. This explains why the LLM remained far below the best classical and deep-learning models, and why its predictions collapsed toward safer or more frequent semantic choices rather than following the intended EEG affect logic.

A stronger LLM-based approach would require either task-specific calibration/fine-tuning, explicit dataset-level normalization statistics in the prompt, clearer sign-convention handling for FAA, or a richer representation that preserves temporal and spatial EEG structure. Without these additions, the LLM treats the EEG values as generic scalar tokens rather than as measurements embedded in a subject-dependent neurophysiological system.

## Generated files

- `llm_prompt_feature_values.csv`: every feature value inserted into prompts.
- `feature_distribution_summary.csv`: descriptive statistics and separability metrics.
- `prompt_rule_diagnostics.csv`: standardized diagnostic scores for prompt-rule evidence.
- `prompt_rule_diagnostics_by_class.csv`: class-level averages of the diagnostic scores.
- `faa_sign_rates_by_class.csv`: FAA sign rates under the implemented convention.

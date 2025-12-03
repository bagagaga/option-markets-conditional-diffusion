# Finance-Structured Conditional Diffusion for Option Markets

## Problem Statement

This project aims to develop a model that jointly predicts two key market structures for the next trading day:
- the forward curve of the S&P 500 futures, and
- the implied volatility surface (IV surface).

Given these predicted structures, option prices can be computed via the Black-Scholes framework.

The goal is to obtain a realistic, arbitrage-consistent generative model of the market’s future state - one that respects financial constraints and outperforms standard approaches on P&L-oriented metrics, rather than only on pointwise errors such as MSE between predicted and realized prices.

The practical motivation is improved risk forecasting and option-portfolio hedging, especially in periods of elevated volatility and nonlinear market dynamics.

## Thesis Plan

Here is a complete plan of stages that are expected sution research work. Note, that the "●" sign marks each official event considered by the department, "➎" - checkpoints (CP) that are stated officially for an experimental part itself, and "◯" is an approximate inner step that was pre-designed by the author of this thesis.


| CP | Event | Date |   Description | Status |
| -------- | -------- | ------- | --------        | ------- | 
| ● | Topic selection   | 05.11   | Finalize title and supervisor form. Define initial scope.  | ☑ |
| ➊ | Plan + Literature Review  | 03.12   | Write a concise literature review (diffusion models, generative IVS, term structures). Define problem setup, data sources, baseline models, evaluation metrics (MSE + hedging P&L).  | ☑ |
| ○ | Full Stable Data/Model Pipeline  | 30.12   | Complete data preprocessing pipeline (futures/IVS history, bucketing, normalization). Implement baseline models: historical resampling baseline, parametric SABR/Local-Vol + forecast baseline, simple ML predictor (ConvLSTM or MLP on factors), implement core diffusion model skeleton (conditional inputs + training loop).  | ☐ |
| ➋ |  Baselines  | 20.01   | Deliver working pipeline regardless of research outcome.  | ☐ |
| ◯ | Experiments + Article Draft  | 01.02   | Train full conditional diffusion model. Add no-arbitrage constraints. Compare to baseline methods. Compute hedging P&L experiments. Draft full results section  | ☐ |
| ◯  |  Paper Completion |  01.03  | Complete article. Develop pip package as supplementory materials.  | ☐ |
| ➌ | Main Research Work  | 10.03   | Tune parameters. Compare generative architecture models.  | ☐ |
| ◯  | Review  | 01.03 - 10.05   | Make editions according to paper revision iterations.  | ☐ |
| ● | Deadline for changing topic  | 15.03   | Change optionally.  | ☐ |
| ● | Mandatory Pre-defence prep  | 01.04 - 05.04   | Prepare slides. Polish methodology and contributions. Prepare pre-defence deck + short speech.  | ☐ |
| ➍ | Final Checkpoint  | 10.05   | Final proofreading. Supervisor approval. Thesis submission. Defence preparation. | ☐ |
| ● | Defence  | 20.05 – 10.06  | Defence thesis.  | ☐ |




---
title: A data-driven prognostic score estimation
date: 2022-11-29 01:00:00 +0000 
---


[The prognostic score](https://academic.oup.com/biomet/article-abstract/95/2/481/230183?redirectedFrom=fulltext) is an analog to the propensity score. 
This sentence is pretty vague; to be mathematically precise, let $X$ and $Y$ denote covariates and outcomes, respectively.
Then, we say $\psi(X)$ is a prognostic score if $Y \perp X \,|\, \phi(X)$.
Intuitively, a prognostic score is a 'summary' of covariates relevant to the outcome.

Then, how do we get or calculate a prognostic score $\psi(X)$? 
The original work says it depends on the assumed model regarding the covariates and the outcomes rather than just on covariates and outcome data.
Furthermore, it assumes $\psi(X)$ is a scalar. 
However, it is not always the case; consider $Y = (X_1 + \epsilon_1)(X_2 + \epsilon_2)$, where $\epsilon_1$ and $\epsilon_2$ are independent noises.
As conditioning only one of $X_1$ or $X_2$ does not specify the distribution of $Y$, $\psi(X)$ cannot be a one-dimensional value.

To resolve these issues, we devised a data-driven method to estimate a multi-dimensional prognostic score. 
The main idea of this procedure is the following two steps:
- Decompose $X$ into independent components,
- Choose components most relevant to $Y$ among those.
  
We use the independent component analysis (ICA) for the first step.
I learned it while studying causal discovery; ICA is one prominent approach for causal discovery, including the seminal work [LinGAM](https://www.jmlr.org/papers/v7/shimizu06a.html). 
ICA assumes the sources are non-Gaussian, so we need the same assumption here. 

For the second step, we use mutual information. 
It indicates the amount of information two variables share, so we sort components by mutual information of them and $Y$ in descending order and take the first few of them.
This process has a drawback in that we should select the number of independent components to be chosen. 
We may change how to choose the relevant components; for example, we can threshold mutual information. 

To verify our method, we calculated the average treatment effect on the treated (ATT) by using the estimated prognostic score.


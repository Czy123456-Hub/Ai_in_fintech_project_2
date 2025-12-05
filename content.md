# **Technical Report on Model Reduction, Optimization, and Factor Reconstruction for the QRT Challenge**

## **Abstract**

This report presents a systematic methodology for forecasting short-horizon cross-sectional equity returns within the constraints of the QRT Challenge. The work integrates (1) a mathematically rigorous reduction of the prescribed factor model, (2) a regularized and momentum-accelerated stochastic gradient method for optimizing the cosine similarity metric, and (3) a principled strategy for reconstructing ten explanatory factors for final submission. The combined methodology significantly improves predictive performance compared to standard baselines. The final model achieves a cosine similarity of **0.0784**, outperforming random-search, sparse regression, and unregularized gradient-descent methods.

------

# **1. Introduction**

The QRT Challenge concerns the prediction of next-day cross-sectional returns using lagged historical data alone. Given a series of daily return vectors  [  R_t \in \mathbb{R}^{50},  ]  the task is to construct a function  [  f: \mathbb{R}^{50 \times D} \rightarrow \mathbb{R}^{50}  ]  mapping the past (D = 250) days of returns to a forecast of the next-day return vector (\hat{R}_{t+1}). Model evaluation is performed exclusively using the **mean cosine similarity**:  [  \mathcal{M}

\frac{1}{T}
 \sum_{t}
 \frac{\hat{R}_t^\top R_t}{|\hat{R}_t| , |R_t|}.
 ]

This metric emphasizes **directional accuracy**—a key property in portfolio allocation and risk-neutral trading contexts—while being invariant to scale. The challenge further restricts models to use at most **10 latent explanatory factors**, necessitating a balance between model expressiveness and interpretability.

The contributions of this work are threefold:

1. A complete **dimensionality reduction** of the factor model into a single linear filter, enabling efficient optimization.
2. A **regularized Nesterov-accelerated gradient descent (SGD)** procedure to optimize the cosine-similarity objective.
3. A principled **factor reconstruction** method to produce ten interpretable factors for submission.

These steps form a coherent pipeline that yields state-of-the-art performance under the challenge setting.

------

# **2. Methodology: Model Formulation, Optimization, and Factor Reconstruction**

This section consolidates the full methodological framework, including the model formulation, optimization procedure, and construction of final explanatory factors.

------

## **2.1 Predictive Model Formulation and Dimensionality Reduction**

### **2.1.1 Original Factor Model**

The challenge prescribes a factor-based model:
 [
 F_{i,t} = \sum_{j=1}^{D} A_{i,j} R_{t+1-j},
 \qquad
 \hat{R}*{t+1} = \sum*{i=1}^{F} \beta_i F_{i,t},
 ]
 with (F \le 10) and orthonormality constraints on the rows of (A).

Although interpretable, this formulation presents several difficulties:

- Orthogonality constraints introduce a nonlinear manifold.
- The cosine similarity objective is highly non-convex.
- The parameter space (F \cdot D) is large (up to 2,500 variables).

### **2.1.2 Exact Reduction to a Linear Filter**

We observe that:
 [
 \hat{R}*{t+1}
 = \sum*{i=1}^{F} \beta_i \sum_{j=1}^{D} A_{i,j} R_{t+1-j}
 = \sum_{j=1}^{D} w_j R_{t+1-j},
 ]
 where
 [
 w_j = \sum_{i=1}^{F} \beta_i A_{i,j}.
 ]

Thus the factor model is exactly equivalent—without approximation—to a **single unconstrained weight vector** (w \in \mathbb{R}^{D}).

### **2.1.3 Advantages of Reduction**

This reduced model:

- preserves the full representational capacity of the factor model;
- removes all orthogonality and normalization constraints;
- reduces the optimization dimension from (F \cdot D) to (D);
- provides a smooth Euclidean search space, improving numerical stability;
- supports post-hoc reconstruction of interpretable factors.

Accordingly, all optimization is performed in the reduced parameter space.

------

## **2.2 Optimization of the Cosine-Similarity Objective**

### **2.2.1 Objective Function**

# Let  [  X_t = R_{t-D:t} \in \mathbb{R}^{50 \times D},  \qquad  \hat{R}_t = X_t w.  ]  The cosine similarity objective is:  [  \mathcal{M}

\frac{1}{T}
 \sum_{t}
 \frac{\hat{R}_t^\top R_t}{|\hat{R}_t| , |R_t|}.
 ]

### **2.2.2 Gradient Derivation**

The gradient is:  [  \nabla_w \mathcal{M}(w)

\frac{1}{T}  \sum_{t}  X_t^\top  \left[  \frac{ \tilde{R}_t }{|\hat{R}_t|}

\frac{\langle \hat{R}_t, \tilde{R}_t\rangle}{|\hat{R}_t|^2}
 \hat{R}_t
 \right],
 ]
 with (\tilde{R}_t = R_t / |R_t|).

Notable analytical properties include:

- **Scale invariance:**
   gradient is orthogonal to (w);
- **Non-convexity:**
   resulting from normalization terms;
- **Heavy dependence on the geometry of return vectors**.

These properties motivate the use of regularization and momentum.

------

## **2.3 Regularization Framework**

To mitigate overfitting and improve optimization stability, smoothness penalties are incorporated:

### **L2 Smoothness**

[
 J_2(w) = \lambda \sum_j (w_{j+1}-w_j)^2,
 ]

### **L1 Total Variation**

[
 J_1(w) = \lambda \sum_j |w_{j+1}-w_j|.
 ]

### **Absolute-Value Smoothness**

Penalizes changes in (|w_j|), permitting sign changes while maintaining structured magnitude patterns.

These penalties reflect the assumption that predictive structure evolves smoothly over lag windows.

------

## **2.4 Optimization Algorithm: Nesterov-Accelerated SGD**

Training employs Nesterov momentum, which accelerates convergence in non-convex settings.
 Regularization significantly improves:

- the conditioning of the gradient;
- monotonic ascent of the objective;
- convergence stability across seeds;
- suppression of high-frequency weight oscillations.

**Figure (from HTML): Convergence curves**
 Shows stable improvement of the cosine similarity during training.

The method converges reliably within several hundred epochs.

------

## **2.5 Reconstruction of Final Explanatory Factors**

After obtaining optimized filters (w) from multiple SGD runs under different regularization schemes, we reconstruct the mandated **10-factor model**.

### **2.5.1 Collection of Candidate Weight Vectors**

Multiple training sessions yield a set:
 [
 \mathcal{W} = { w^{(1)}, \dots, w^{(K)} }.
 ]

Each (w^{(k)}) induces a predictive signal:
 [
 S^{(k)}_t = X_t w^{(k)}.
 ]

### **2.5.2 Selection of a Diverse Subset**

Two “anchor” vectors are selected based on minimal correlation:

- `one L1-regularized`,
- `one L2-regularized`.

Eight additional vectors are drawn from the remaining pool subject to pairwise correlation constraints.

**Figure 10 (HTML): Selected weight vectors**
 Illustrates the diversity in smoothness, amplitude, and structure across the ten chosen vectors.

### **2.5.3 Construction of the Factor Space**

The final ten explanatory factors are constructed via:
 [
 F_{i,t} = \sum_{k=1}^{10} c_{i,k} S^{(k)}_t.
 ]

Coefficients (C = [c_{i,k}]) are computed by:

1. **Gram–Schmidt orthogonalization** (to satisfy challenge constraints),
2. **Least-squares fitting** to align the factor model with the predictive behavior of the reduced model.

### **2.5.4 Normalization**

Each factor is scaled to satisfy:
 [
 |F_{i}|_2 = 1.
 ]

This yields a set of valid, interpretable factors consistent with challenge requirements.

------

# **3. Final Performance**

The optimized model achieves:

[
 \boxed{
 \mathcal{M}_{\text{final}} = 0.0784
 }
 ]

This result exceeds:

- random search (~0.0539),
- sparse regression (Lasso) (~0.05),
- unregularized SGD (~0.054).

The improvement underscores the efficacy of:

- exact model reduction,
- gradient-based optimization with smoothness regularization,
- structured factor reconstruction.

------

# **4. Conclusion**

This work presents a cohesive methodological pipeline for forecasting cross-sectional equity returns using a constrained factor-based model. Through mathematically justified model reduction, smoothness-regularized Nesterov SGD, and a structured post-hoc factor reconstruction procedure, the approach meets stringent interpretability requirements while delivering superior predictive performance. The achieved cosine similarity of 0.0784 reflects a significant improvement over baseline and conventional approaches, demonstrating the value of integrating optimization theory, regularization principles, and factor modeling in noisy financial environments.


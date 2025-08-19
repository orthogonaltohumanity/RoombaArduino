# `sketch_aug16pmi.ino`

This Arduino sketch drives a small robot via two minimal **binary neural networks** and a probabilistic reward scaler based on **pointwise mutual information (PMI)**. It removes moving averages, relies on raw sensor reads, and updates its parameters online with a tiny log‑linear softmax model. This document details every major component: the binary networks, the modified Fisher–Yates mutator, the reward pipeline, the PMI scaling, and the learning rule.

## Binary neural architecture
The controller is composed of two BinaryNNs:

* **Embedding network** – encodes the previous action into a 8‑dimensional embedding.
* **Action network** – maps the embedding and current sensor features to logits for motor, servo, and beep bins.

Every weight is a single bit \(w_{ij}\in\{-1,+1\}\), so matrix–vector products reduce to integer adds/subtracts. Activations are hard signs, yielding extremely cheap inference on the AVR.

### Modified Fisher–Yates mutation
Exploration mutates the binary weights via a two‑dimensional Fisher–Yates walk augmented with Hebbian kernels:

1. Pick a layer and choose a starting weight \(w_1\) at coordinates \((x_1,y_1)\).
2. Sample a partner \(w_2\) at \((x_2,y_2)\) by drawing independent row/column offsets \(d_r=|x_2-x_1|\) and \(d_c=|y_2-y_1|\) from truncated geometric laws
   \(P(d_r)\propto 2^{-\lambda d_r}\) and \(P(d_c)\propto 2^{-\lambda d_c}\).
3. Compute four cosine similarities among the associated row/column vectors:
   \(d_a=\cos(\vec w^r_{x_1},\vec w^r_{x_2})\),
   \(d_b=\cos(\vec w^c_{y_1},\vec w^c_{y_2})\),
   \(d_c=\cos(\vec w^c_{y_1},\vec w^r_{x_2})\),
   \(d_d=\cos(\vec w^r_{x_1},\vec w^c_{y_2})\).
4. Accept \(w_2\) with probability proportional to
   \(2^{\lambda(|d_a|+|d_b|+|d_c|+|d_d|)}\), favouring swaps between neurons with similar firing patterns.
5. Swap \(w_1\) and \(w_2\); repeat for up to `FY_MAX_SWAPS` bases.

Thus the joint kernel obeys
\[
P(d_r,d_c,d_h) \propto 2^{-\lambda d_r}
                       2^{-\lambda d_c}
                       2^{\lambda|d_a|}
                       2^{\lambda|d_b|}
                       2^{\lambda|d_c|}
                       2^{\lambda|d_d|},
\]
where \(d_h\) denotes the four cosine terms. High reward pushes \(\lambda\) up, biasing toward local swaps between similar neurons; low reward flattens it, recovering global, pattern‑agnostic shuffles.

During a hill‑climb episode the mutated network runs; if cumulative reward exceeds the baseline, the swap log is kept, otherwise all swaps are reverted. A background bit flip with 0.5–2 % probability per step injects additional stochastic exploration, irrespective of Fisher–Yates swaps.

## Reward pipeline
1. **Raw reward** \(r_{\text{raw}}\) from sensors is stored in Q8 (\([-256,256]\)):
   \[
   r_{\text{raw}} = \begin{cases}
     r_{\text{plate}} & \text{if plate contact}\\
     r_{\text{button}} & \text{else if button pressed}\\
     \dfrac{256}{3\cdot1023}(l+r+c) + r_{\text{idle}} & \text{otherwise}
   \end{cases}
   \]
   The brightness term uses raw LDR readings \(l,r,c\), and an idle penalty is added when both motors are off.
2. **Delta reward** \(dR = r_{\text{raw}}(t) - r_{\text{raw}}(t-1)\) is binned with threshold \(\tau=16\):
   \[
   k = \begin{cases}
     2 & dR > \tau\\
     0 & dR < -\tau\\
     1 & \text{otherwise}
   \end{cases}
   \]

## Log‑linear models
A compact softmax model estimates \(P(k\mid A)\) and a global model estimates \(P(k)\). For motor bin \(m\), servo bin \(s\), and beep bin \(b\), the conditional logits are
\[
  u_k = \beta_k + \theta^{(M)}_{k,m} + \theta^{(S)}_{k,s} + \theta^{(B)}_{k,b},
\]
scaled by \(\alpha=32\) before the softmax:
\[
  P(k\mid A) = \frac{\exp(u_k/\alpha)}{\sum_j \exp(u_j/\alpha)}.
\]
Global logits \(\gamma_k\) yield
\[
  P(k) = \frac{\exp(\gamma_k/\alpha)}{\sum_j \exp(\gamma_j/\alpha)}.
\]
All parameters are int8 and shared across actions, consuming \(<100\) bytes.

## PMI‑based scaling
For the observed bin \(k_{obs}\), the PMI is
\[
  \text{PMI} = \log P(k_{obs}\mid A) - \log P(k_{obs}).
\]
After clamping to \([-C,C]\) with \(C=2\) and applying a sigmoid with gain \(\kappa=1\), the scaling factor in \([0,1]\) is
\[
  \text{scale} = \sigma(\kappa\,\mathrm{clip}(\text{PMI},-C,C)).
\]
The final reward is \(r = r_{\text{raw}} \cdot \text{scale}\); thus the raw reward retains its sign and magnitude but is damped or amplified according to how surprising the outcome was.

## Online update
For each bin, gradient \(g_k = \mathbf{1}[k=k_{obs}] - P(k\mid A)\) updates the parameters with learning rate \(\eta=1/4\):
\[
  \theta \leftarrow \mathrm{clip}(\theta + \eta g_k, -96, 96).
\]
This applies to \(\beta_k,\ \theta^{(M)}_{k,m},\ \theta^{(S)}_{k,s},\ \theta^{(B)}_{k,b}\), and \(\gamma_k\), tying action‑conditional and global models together.

## Main loop summary
1. Encode previous action and compute an embedding.
2. Read raw sensors and build features.
3. Run the action network, choose motor/servo/beep bins by argmax.
4. Execute actuators and compute \(r_{\text{raw}}\) and \(dR\).
5. Scale reward via PMI, update models, and feed reward to the hill‑climbers.


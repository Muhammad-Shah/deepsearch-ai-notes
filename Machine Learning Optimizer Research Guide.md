

# **The Mathematical Engine of Intelligence: A Comprehensive Analysis of Optimization Algorithms in Deep Learning**

## **1\. The Landscape of High-Dimensional Optimization**

The training of deep neural networks is fundamentally an optimization problem, yet it differs continuously and categorically from the classical optimization problems found in convex analysis or operations research. While the elementary objective—minimizing a scalar loss function $L(\\theta)$ with respect to parameters $\\theta \\in \\mathbb{R}^d$—remains consistent, the geometric reality of the loss landscape in deep learning is a terrain of unimaginable complexity. We are not merely descending a smooth, convex bowl; we are navigating a high-dimensional manifold riddled with pathological curvature, saddle points, extensive flat plateaus, and non-convex ravines.1

The evolution of optimizers, from the rudimentary Gradient Descent proposed by Augustin-Louis Cauchy in 1847 to the AI-discovered algorithms of the 2020s like Lion, represents the field's attempt to navigate this hostility.3 This report provides an exhaustive technical dissection of these algorithms, not merely as heuristic tools, but as mathematical objects with distinct convergence properties, implicit biases, and implementation nuances.

### **1.1 The Curse of Dimensionality and Saddle Points**

A pervasive intuition in optimization is the fear of local minima—valleys where the optimizer might get stuck, failing to reach the global optimum. However, in the high-dimensional spaces characteristic of deep learning (where $d$ can exceed billions), the probability of finding a true local minimum—where the Hessian matrix is positive definite (all eigenvalues $\> 0$)—is vanishingly small.

Statistical physics and random matrix theory suggest that as dimensionality increases, the vast majority of critical points (where $\\nabla L \= 0$) are **saddle points**, where the surface curves upward in some dimensions and downward in others.4

* **Geometric Intuition:** If a critical point is a local minimum, the curvature must be positive in all $d$ directions. If we assume a 50% chance of positive curvature in any single direction, the probability of a critical point being a minimum is $0.5^d$. For a modest network with one million parameters, this probability is effectively zero.  
* **The Saddle Point Problem:** Standard gradient descent can theoretically stall at saddle points because the gradient is zero. However, saddle points are unstable; any perturbation along a direction of negative curvature will cause the parameters to "slide off" the saddle. This geometric property makes **noise**—specifically the gradient noise inherent in Stochastic Gradient Descent (SGD)—a crucial feature rather than a bug. The noise prevents the optimizer from stagnating on saddles, effectively "kicking" it into directions where the loss can be further minimized.6

### **1.2 The Generalization Paradox**

Deep learning optimizers face a dual mandate: they must minimize the training loss (optimization) and ensure the solution performs well on unseen data (generalization). These goals are often at odds. A solution that achieves zero training loss might settle in a "sharp" minimum—a narrow canyon where slight perturbations (or shifts in data distribution) result in massive error increases. Conversely, "flat" minima—wide basins of low loss—are associated with better generalization because they are robust to parameter perturbations.8

The choice of optimizer dictates which type of minimum is found. This introduces the concept of **Implicit Bias**: the preference of an algorithm for certain types of solutions not explicitly encoded in the loss function. For instance, SGD with a large learning rate struggles to enter sharp minima (it bounces out), thereby implicitly preferring flat, generalizable regions.9 This phenomenon, known as the "Edge of Stability," challenges classical convergence theories that demand small step sizes, suggesting instead that operating at the brink of divergence is essential for modern model performance.

## **2\. First-Order Stochastic Methods: The Bedrock**

Despite the proliferation of complex adaptive methods, the family of Stochastic Gradient Descent (SGD) remains the gold standard for many tasks, particularly in computer vision. Its simplicity belies a complex dynamical behavior that acts as a regularizer.

### **2.1 Stochastic Gradient Descent (SGD)**

The fundamental update rule for SGD is:

$$\\theta\_{t+1} \= \\theta\_t \- \\eta \\nabla\_{\\theta} L(\\theta\_t; x\_i, y\_i)$$

where $\\eta$ is the learning rate and $(x\_i, y\_i)$ is a mini-batch of data.3

#### **2.1.1 The Role of Batch Size**

The transition from Batch Gradient Descent (using the full dataset) to SGD (using one sample) to Mini-Batch SGD represents a trade-off between gradient accuracy and computational efficiency.

* **Batch GD:** Computes the true gradient. The trajectory is deterministic and smooth. However, it is computationally prohibitive for large datasets and, crucially, lacks the stochastic noise needed to escape saddle points or select for flat minima.10  
* **SGD (Single Sample):** Extremely noisy. The gradient variance is high, causing the trajectory to oscillate violently. While this noise helps escape local traps, it prevents convergence to the exact minimum unless the learning rate is annealed to zero.  
* **Mini-Batch SGD:** By averaging gradients over a batch $B$ (typically 32 to 4096), we reduce the variance of the gradient estimate by a factor of $1/B$. This balances stability with the necessary stochasticity. Large-batch training ($B \> 8192$) often degrades generalization (the "Generalization Gap") because the reduced noise allows the optimizer to settle into sharp minima.8

### **2.2 Momentum: Overcoming Pathological Curvature**

Standard SGD has trouble navigating "ravines"—areas where the surface curves much more steeply in one dimension than in another. In these regions, SGD oscillates rapidly across the steep slope while making negligible progress along the gentle slope toward the optimum.3

Classical Momentum (Polyak Heavy Ball) introduces a velocity variable $v\_t$, which accumulates the history of gradients:

$$v\_t \= \\mu v\_{t-1} \+ g\_t$$

$$\\theta\_{t+1} \= \\theta\_t \- \\eta v\_t$$

Here, $\\mu$ (typically 0.9) is the momentum coefficient.

* **Physics Analogy:** Imagine a heavy ball rolling down the loss surface. $g\_t$ is the force of gravity (slope), and $\\mu$ represents air resistance. The ball gains momentum, allowing it to plow through small local bumps and maintain speed in flat directions. In ravines, the oscillations in the steep direction tend to cancel each other out (alternating signs), while the components in the flat direction accumulate, accelerating convergence.13

### **2.3 Nesterov Accelerated Gradient (NAG): The Lookahead**

Nesterov Momentum refines the physical analogy by introducing prescience. In classical momentum, we compute the gradient at the current position $\\theta\_t$, then take a big step governed by the accumulated velocity. If the velocity is about to carry us up the other side of a valley, we won't know until we have already overshot.

Nesterov suggested computing the gradient after applying the velocity step. This "lookahead" gradient tells us where we are going to be, allowing the optimizer to correct its course before making the update.13  
The canonical formulation is:

$$v\_{t} \= \\mu v\_{t-1} \+ \\nabla L(\\theta\_t \- \\eta \\mu v\_{t-1})$$

$$\\theta\_{t+1} \= \\theta\_t \- \\eta v\_t$$

#### **2.3.1 Implementation Divergence: PyTorch vs. Theory**

A critical detail for practitioners is that deep learning frameworks often do not implement the exact Sutskever/Nesterov formulation due to the inconvenience of calculating the gradient at a shifted position ($\\theta\_t \- \\eta \\mu v\_{t-1}$).  
PyTorch Implementation: PyTorch approximates NAG by applying the momentum correction to the update rule rather than the gradient calculation.  
The PyTorch update is:

$$v\_t \= \\mu v\_{t-1} \+ g\_t$$

$$\\theta\_{t+1} \= \\theta\_t \- \\eta (g\_t \+ \\mu v\_t)$$

Notice that the update includes the current gradient $g\_t$ plus the scaled velocity. This mathematically approximates the lookahead behavior without requiring a separate forward pass at the shifted position. While subtly different from the pure derivation, empirical performance is comparable, and it integrates seamlessly into the backpropagation engine.16

## **3\. The Adaptive Learning Rate Revolution**

The primary limitation of SGD and Momentum is the reliance on a global learning rate $\\eta$. In neural networks, parameters exist in vastly different scales. For example, in a word embedding matrix, the vector for the word "the" might receive gradients in every batch, while the vector for "idiosyncratic" might receive a gradient once every million samples. A global learning rate that is safe for "the" would be too slow for "idiosyncratic," while a rate that is fast enough for "idiosyncratic" would cause "the" to diverge.18

Adaptive methods solve this by maintaining a per-parameter learning rate.

### **3.1 Adagrad: Handling Sparsity**

Adagrad (Adaptive Gradient Algorithm) was the pioneer of this family. It scales the learning rate inversely by the sum of squared past gradients.19

$$G\_{t, ii} \= \\sum\_{\\tau=1}^t g\_{\\tau, i}^2$$

$$\\theta\_{t+1, i} \= \\theta\_{t, i} \- \\frac{\\eta}{\\sqrt{G\_{t, ii} \+ \\epsilon}} g\_{t, i}$$

* **Mechanism:** Parameters with frequent, large gradients accumulate a large $G$, resulting in a small effective learning rate. Rarely updated parameters have a small $G$, maintaining a large effective learning rate.  
* **The Flaw:** The accumulation $\\sum g^2$ is monotonically increasing. As training progresses, the denominator grows indefinitely, causing the learning rate to vanish effectively to zero. This makes Adagrad unsuitable for training deep networks that require many epochs, though it remains useful for convex sparse problems (e.g., logistic regression on TF-IDF features).20

### **3.2 RMSprop: The Leaky Integrator**

RMSprop (Root Mean Square Propagation) was introduced not in a paper, but in a Coursera lecture by Geoffrey Hinton (Lecture 6e). It resolves Adagrad's vanishing gradient problem by replacing the infinite sum with an **Exponential Moving Average (EMA)**.22

$$E\[g^2\]\_t \= \\beta E\[g^2\]\_{t-1} \+ (1 \- \\beta) g\_t^2$$

$$\\theta\_{t+1} \= \\theta\_t \- \\frac{\\eta}{\\sqrt{E\[g^2\]\_t \+ \\epsilon}} g\_t$$

By decaying old history (controlled by $\\beta$, typically 0.9), RMSprop ensures the denominator does not grow infinitely. It adapts to the recent curvature of the loss surface.

* **Physics of Preconditioning:** RMSprop acts as a diagonal preconditioner. In directions of high curvature (large gradients), it divides by a large number, reducing the step size. In flat directions, it divides by a small number, increasing the step size. This effectively transforms the elliptical contours of a ravine into a more spherical landscape, allowing the optimizer to move directly toward the minimum.24

### **3.3 Adadelta: Dimensional Consistency**

Adadelta was developed simultaneously with RMSprop to address two issues: the vanishing learning rate of Adagrad and the "unit mismatch" problem of SGD.25  
The Unit Mismatch: In standard gradient descent, the update $\\Delta \\theta$ has units proportional to the gradient $\\nabla L$. Since $\\nabla L \\approx \\frac{\\partial L}{\\partial \\theta}$, the units are $1/\\text{units of } \\theta$.

$$\\theta\_{new} \= \\theta\_{old} \- \\eta \\cdot \\text{gradient}$$

Equation units: $\[\\theta\] \= \[\\theta\] \- \[\\eta\] \\cdot \[1/\\theta\]$.  
For the units to match, the learning rate $\\eta$ must have units of $\[\\theta\]^2$. This is theoretically unsatisfying. Newton's method ($\\Delta \\theta \= H^{-1} g$) has the correct units because the inverse Hessian ($\[\\theta\]^2$) multiplies the gradient ($1/\[\\theta\]$), resulting in units of $\[\\theta\]$.26  
Adadelta approximates this second-order behavior by accumulating a window of past updates ($\\Delta \\theta$):

1. Compute gradient RMS: $RMS\[g\]\_t \= \\sqrt{E\[g^2\]\_t \+ \\epsilon}$  
2. Compute update: $\\Delta \\theta\_t \= \- \\frac{RMS\_{t-1}}{RMS\[g\]\_t} g\_t$  
3. Accumulate update RMS: $E\_t \= \\rho E\_{t-1} \+ (1-\\rho) \\Delta \\theta\_t^2$

Note that there is **no global learning rate** $\\eta$ in Adadelta. The step size is entirely determined by the ratio of past updates to past gradients.28

### **3.4 Adam: The Modern Standard**

Adam (Adaptive Moment Estimation) synthesizes the benefits of Momentum (first moment) and RMSprop (second moment).30  
Algorithm:

1. Update biased first moment (Momentum): $m\_t \= \\beta\_1 m\_{t-1} \+ (1 \- \\beta\_1) g\_t$  
2. Update biased second moment (RMSprop): $v\_t \= \\beta\_2 v\_{t-1} \+ (1 \- \\beta\_2) g\_t^2$  
3. Bias Correction: Since $m$ and $v$ are initialized to vectors of zeros, they are biased toward zero, especially in early steps.

   $$\\hat{m}\_t \= \\frac{m\_t}{1 \- \\beta\_1^t}, \\quad \\hat{v}\_t \= \\frac{v\_t}{1 \- \\beta\_2^t}$$  
4. Update parameters: $\\theta\_{t+1} \= \\theta\_t \- \\eta \\frac{\\hat{m}\_t}{\\sqrt{\\hat{v}\_t} \+ \\epsilon}$

#### **3.4.1 The Epsilon ($\\epsilon$) Controversy**

The $\\epsilon$ parameter is theoretically just for numerical stability (to avoid division by zero), but in practice, it acts as a clamp on the maximum learning rate.

* **TensorFlow vs. PyTorch:** PyTorch defaults to $\\epsilon=10^{-8}$. TensorFlow's default in tf.keras.optimizers.Adam is $10^{-7}$. While this seems trivial, in reduced precision training (FP16), $10^{-8}$ can underflow, causing instability.  
* **Tuning:** For certain architectures like Inception or EfficientNet, researchers have found that a much larger $\\epsilon$ (e.g., $1.0$ or $0.1$) improves stability significantly. This effectively transforms Adam closer to pure SGD (since the denominator becomes dominated by $\\epsilon$ rather than $\\sqrt{v\_t}$).32

## **4\. The AdamW Revolution: Decoupling Weight Decay**

For years, the deep learning community struggled with the observation that Adaptive methods (Adam) generalized worse than SGD with Momentum, particularly on image classification tasks (CIFAR, ImageNet).34 The breakthrough came with the identification of a flaw in how $L\_2$ regularization was implemented in adaptive optimizers.

### **4.1 The Flaw in Coupled $L\_2$ Regularization**

In standard SGD, $L\_2$ regularization is implemented by adding a penalty term $\\frac{\\lambda}{2} ||\\theta||^2$ to the loss function. The gradient then becomes $g\_t \+ \\lambda \\theta\_t$.  
$$ \\theta\_{t+1} \= \\theta\_t \- \\eta (g\_t \+ \\lambda \\theta\_t) \= \\theta\_t \- \\eta g\_t \- \\eta \\lambda \\theta\_t $$  
The term $\\eta \\lambda \\theta\_t$ linearly decays the weights. This is mathematically equivalent to Weight Decay.  
However, in Adam, the update is scaled by $1/\\sqrt{v\_t}$. If we add the regularization to the gradient:

$$\\theta\_{t+1} \= \\theta\_t \- \\frac{\\eta}{\\sqrt{v\_t}} (g\_t \+ \\lambda \\theta\_t)$$

The decay term is now $\\frac{\\eta \\lambda}{\\sqrt{v\_t}} \\theta\_t$.  
The Insight: The effective weight decay is scaled by the inverse of the gradient magnitude. Parameters with large gradients (large $v\_t$) experience less regularization, while parameters with small gradients experience more regularization. This is the opposite of what is typically desired; we often want to regularize the parameters that are moving the most to prevent overfitting.35

### **4.2 AdamW Implementation**

Loshchilov and Hutter (2017) proposed **AdamW**, which decouples the weight decay from the gradient update.37

1. Compute Adam step (without regularization in gradients): $\\Delta \\theta\_t \= \\frac{\\eta \\hat{m}\_t}{\\sqrt{\\hat{v}\_t} \+ \\epsilon}$  
2. Apply weight decay separately: $\\theta\_{t+1} \= \\theta\_t \- \\Delta \\theta\_t \- \\eta \\lambda \\theta\_t$

This simple change allows AdamW to match the generalization performance of SGD on many benchmarks, making it the default optimizer for Transformers (BERT, GPT, ViT).35

### **4.3 Implementation Trap in PyTorch**

A common mistake in PyTorch is assuming that setting weight\_decay in the torch.optim.Adam constructor enables AdamW behavior. It does not; it enables the flawed $L\_2$ implementation. One must explicitly use torch.optim.AdamW to get the decoupled behavior. Furthermore, when using AdamW, the optimal weight decay value is typically different (often larger) than what would be used for SGD or standard Adam.39

## **5\. Second-Order and Structure-Aware Methods**

While first-order methods dominate, they ignore the rich curvature information contained in the Hessian matrix ($H$). Second-order methods theoretically offer quadratic convergence but face the $O(N^2)$ storage and $O(N^3)$ inversion bottleneck.

### **5.1 Hessian-Free Optimization (Conjugate Gradient)**

Hessian-Free optimization bypasses the explicit computation of $H$. It attempts to solve the Newton system $H \\Delta \\theta \= \-g$ using the **Conjugate Gradient (CG)** algorithm.

* The Trick: CG only requires the computation of the matrix-vector product $Hv$ for some vector $v$. This can be computed exactly using standard backpropagation operators (the Pearlmutter trick) in $O(N)$ time, without ever forming $H$.

  $$Hv \= \\lim\_{\\epsilon \\to 0} \\frac{\\nabla L(\\theta \+ \\epsilon v) \- \\nabla L(\\theta)}{\\epsilon}$$

  This method allows for "curvature-aware" updates that can navigate pathological landscapes much more efficiently than SGD, though the overhead of running the inner CG loop makes it slower in wall-clock time for many problems.41

### **5.2 K-FAC (Kronecker-Factored Approximate Curvature)**

K-FAC approximates the Fisher Information Matrix (FIM) $F$ (which approximates the Hessian) by assuming a specific structure in the network's layers.  
For a linear layer $y \= Wx$, the gradient is $\\nabla\_W L \= \\delta x^T$ (where $\\delta$ is the backpropagated error). The FIM block for this layer is $E$.  
K-FAC approximates this as the Kronecker product of the covariance of inputs and the covariance of gradients:

$$F \\approx E \\otimes E \= A \\otimes G$$

The Gain: The inverse of a Kronecker product is the Kronecker product of inverses: $(A \\otimes G)^{-1} \= A^{-1} \\otimes G^{-1}$.  
Inverting two small matrices $A$ and $G$ is computationally feasible. K-FAC is widely used in large-scale distributed training and Reinforcement Learning (e.g., ACKTR) where sample efficiency is paramount.43

### **5.3 Shampoo: Tensor Preconditioning**

Shampoo generalizes K-FAC to tensor parameters (like Convolutional kernels). It maintains preconditioner matrices for each dimension of the tensor. For a rank-4 tensor (e.g., a Conv2D kernel), Shampoo computes 4 smaller matrices.

$$\\text{Update} \= g \\times\_1 L\_1^{-1/4} \\times\_2 L\_2^{-1/4} \\dots$$

Shampoo enables training with massive batch sizes (e.g., 32k) without the degradation in convergence seen with diagonal adaptive methods (like Adam). It captures correlations within spatial dimensions and channel dimensions separately.46

## **6\. The Frontier: AI-Discovered and Matrix-Specific Optimizers**

The 2023-2025 era has seen a shift from manually derived optimizers to those discovered via symbolic search or designed for specific matrix properties.

### **6.1 Lion (EvoLved Sign Momentum)**

Discovered by Google Brain using evolution-based symbolic search, Lion (Evolved Sign Momentum) is radically simple and memory efficient.48

**Update Rule (Program 1):**

1. Interpolate gradient: $c\_t \= \\beta\_1 m\_{t-1} \+ (1-\\beta\_1) g\_t$  
2. **Sign Update:** $u\_t \= \\text{sign}(c\_t)$  
3. Update Momentum: $m\_t \= \\beta\_2 m\_{t-1} \+ (1-\\beta\_2) g\_t$  
4. Apply Update: $\\theta\_{t+1} \= \\theta\_t \- \\eta u\_t \- \\eta \\lambda \\theta\_t$

**Analysis:**

* **Memory:** Lion tracks only momentum ($m\_t$), saving 50% memory compared to Adam (which tracks $m\_t$ and $v\_t$).  
* **Sign Operation:** Taking the sign makes the update magnitude uniform ($\\eta$) for all parameters. This acts as a strong regularizer and ensures the update norm is $\\eta \\sqrt{d}$.  
* **Hyperparameters:** Because the update norm $\\sqrt{d}$ is much larger than a standard gradient, the learning rate $\\eta$ must be set **3-10x lower** than AdamW, and weight decay $\\lambda$ must be **3-10x higher**.50  
* **Performance:** Lion outperforms AdamW on ImageNet (ViT) and diffusion models, primarily due to its regularization effects and efficiency.48

### **6.2 Sophia: Diagonal Hessian estimates**

Sophia (Second-order Clipped Stochastic Optimization) targets LLM pre-training. It uses a lightweight estimate of the diagonal Hessian ($h\_t$) to precondition the gradients.51  
Estimator: It typically uses a Hutchinson estimator or a Gauss-Newton-Bartlett estimator to approximate curvature without full backprop.  
Clipping: A key innovation is element-wise clipping of the update:

$$\\theta\_{t+1} \= \\theta\_t \- \\eta \\cdot \\text{clip}\\left( \\frac{m\_t}{\\max(h\_t, \\epsilon)}, \\rho \\right)$$

This clipping handles the instability of second-order updates in non-convex regions (where Hessian might be negative or near zero), preventing the "exploding update" problem typical of Newton's method. Sophia reports a 2x speedup in wall-clock time for training GPT-2 sized models.52

### **6.3 Muon: Newton-Schulz Orthogonalization**

Muon (MomentUm Orthogonalized by Newton-Schulz) is a specialized optimizer for the internal linear layers (matrices) of neural networks.53  
Mechanism: It forces the weight updates to be orthogonal. Standard adaptive methods scale updates element-wise. Muon treats the update matrix $U$ as a whole and attempts to "snap" its singular values to 1\.  
Instead of performing an expensive SVD to orthogonalize ($U \\to UV^T$), it uses Newton-Schulz iteration, an iterative matrix method:

$$X\_{k+1} \= \\frac{1}{2} X\_k (3I \- X\_k^T X\_k) \\quad \\text{(Example quadratic iteration)}$$

Or the specific quintic iteration used in the Muon paper:

$$X\_{k+1} \= a X\_k \+ b X\_k X\_k^T X\_k \+ c X\_k X\_k^T X\_k X\_k^T X\_k$$

with specific coefficients $a, b, c$ derived to optimize convergence to the orthogonal group.54  
Why? Orthogonal updates preserve the gradient norm distribution through deep networks, allowing for training significantly deeper models without normalization layers or careful initialization hacks. It is particularly effective for large-scale transformers.56

## **7\. Optimization Dynamics: Beyond Convergence**

### **7.1 The Edge of Stability (EoS)**

Classical convex optimization theory states that gradient descent is stable only if the learning rate $\\eta \< 2/\\lambda\_{max}$, where $\\lambda\_{max}$ is the sharpness (largest eigenvalue of Hessian).  
Cohen et al. (2021) observed that modern neural networks routinely violate this condition. During training, the sharpness $\\lambda\_{max}$ rises until it hits the stability threshold $2/\\eta$.

* **The EoS Mechanism:** Instead of diverging, the loss oscillates. The optimizer takes a step that is "too large" for the sharp curvature. This large step bounces the parameters *out* of the sharp minimum and onto the "walls" of the valley, where curvature is lower. The sharpness drops, the process repeats.  
* **Implication:** High learning rates force the model to find flatter minima because they physically cannot settle into minima sharper than $2/\\eta$. This explains why LR annealing is crucial: early high LR explores flat regions (Edge of Stability), and late low LR allows the model to finally settle into the bottom of the basin.9

### **7.2 Grokking: The Slingshot Mechanism**

Grokking refers to the phenomenon where a model achieves perfect training accuracy but random validation accuracy (memorization), and then, after many thousands of further steps, sudden generalization occurs.58  
The Slingshot Mechanism: Thilak et al. (2022) linked this to optimization instability. Just before grokking occurs, there is often a spike in loss (a "slingshot") caused by the adaptive optimizer entering a region of instability. This spike ejects the model from the "memorization manifold" and allows it to find a more generalizable solution (the "circuit efficiency" hypothesis). Without this instability (or without weight decay driving the model toward efficient representations), the model might remain in the memorization phase indefinitely.59

### **7.3 Warmup and Schedules**

**Linear Warmup:** Increasing LR from 0 to $\\eta\_{max}$ over the first few thousand steps is mandatory for Transformers.

* **Reason 1 (Curvature):** At initialization, the loss landscape is extremely rugged. A large step leads to immediate divergence. Warmup allows the model to traverse this initial "chaos" to a smoother region.61  
* **Reason 2 (Adam Variance):** The variance estimate $v\_t$ in Adam is initialized at 0\. In the first few steps, the bias correction is large, and the variance estimate is noisy. A full LR would amplify this noise, leading to bad early updates that are hard to recover from.62

## **8\. Technical Summary Tables**

### **Table 1: Hyperparameter Sensitivity and Implementation Defaults**

| Optimizer | Parameter | Default (PyTorch) | Default (TF) | Recommended Tuning Range | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **SGD** | Momentum | 0 | 0 | 0.9 \- 0.99 | High momentum (0.99) helps with large batch training. |
| **Adam** | $\\beta\_1$ | 0.9 | 0.9 | 0.9 | Rarely changed. |
| **Adam** | $\\beta\_2$ | 0.999 | 0.999 | 0.95 \- 0.999 | Lowering to 0.95 or 0.98 can improve stability in GANs/RL. |
| **Adam** | $\\epsilon$ | $1e-8$ | $1e-7$ | $1e-8$ to $1.0$ | **Crucial:** Use $1e-4$ or larger for FP16 training to prevent NaN. |
| **AdamW** | Weight Decay | 0.01 | 0.004 | 0.01 \- 0.5 | Requires much higher values than L2 regularization in SGD. |
| **Lion** | LR | N/A | N/A | $0.1\\times$ AdamW LR | Due to sign update, effective step size is large. |
| **Lion** | Weight Decay | N/A | N/A | $10\\times$ AdamW WD | Must be scaled up to compensate for lower LR. |

### **Table 2: Optimizer Complexity and Memory Footprint**

| Algorithm | Memory (per param) | Compute per step | Best Use Case |
| :---- | :---- | :---- | :---- |
| **SGD** | 0 (bufferless) | $1\\times$ | Simple convex problems, massive data where memory is tight. |
| **SGD+Mom** | 1 ($v\_t$) | $1\\times$ | **Computer Vision (ResNet)**. Often generalizes best. |
| **Adam/AdamW** | 2 ($m\_t, v\_t$) | $\\approx 1.5\\times$ | **NLP / Transformers / General Purpose**. Robust default. |
| **Lion** | 1 ($m\_t$) | $\\approx 1.2\\times$ | **Large ViTs / Diffusion**. Memory constrained training. |
| **Shampoo** | Variable (Rank-dependent) | High (SVD/Inverse) | **Large Batch Training**. Distributed systems (TPUs). |
| **Adafactor** | Sub-linear | Low | **Huge Models (PaLM)**. Factorizes $v\_t$ to save memory. |

## **9\. Practical Recommendations for Engineering**

1. **The "Safe" Baseline:** Start with **AdamW**. Use the default $\\beta$ values. Tune Learning Rate (usually $1e-4$ to $1e-3$) and Weight Decay ($0.01$ to $0.1$). Always use Linear Warmup (first 5-10% of steps) followed by Cosine Decay.  
2. **Vision Tasks:** Do not discard **SGD with Nesterov**. For ResNet-style architectures, it frequently beats AdamW in final top-1 accuracy by 0.5-1.0%, provided you tune the LR schedule (step decay or cosine).  
3. **Memory Bottlenecks:** If you are OOM (Out of Memory) on GPU, switch to **Lion** (saves 50% optimizer state) or use **8-bit Adam** (available in libraries like bitsandbytes).  
4. **Instability:** If loss is spiking or NaN:  
   * Check epsilon. Increase to $1e-6$ or $1e-4$.  
   * Enable **Gradient Clipping** (clip\_grad\_norm\_, max norm 1.0).  
   * Verify Warmup is sufficient.  
5. **Evaluation Mode:** Ensure model.eval() is called during validation. While not an optimizer setting, Batch Normalization statistics update during forward passes in train() mode, which can distort validation metrics if omitted.

## **10\. Conclusion**

The field of optimization has matured from simple hill-climbing to a sophisticated discipline intersecting geometry, probability, and symbolic AI. We have moved from asking "will it converge?" to "what kind of minimum will it find?". The transition from Adam to AdamW corrected a fundamental flaw in regularization. The transition to Lion and Sophia shows that we are now tailoring optimizers to the hardware (memory efficiency) and the model architecture (Transformer-specific dynamics). For the practitioner, the key is no longer just implementing the algorithm, but understanding the *implicit biases* these algorithms introduce into the learning process. Optimization *is* the learning algorithm; the architecture is merely the vessel.

#### **Works cited**

1. Deep Learning Optimization Algorithms \- Neptune.ai, accessed November 23, 2025, [https://neptune.ai/blog/deep-learning-optimization-algorithms](https://neptune.ai/blog/deep-learning-optimization-algorithms)  
2. Escaping Local Minima and Saddle Points in High-Dimensional Non-Convex Optimization Problems \- arXiv, accessed November 23, 2025, [https://arxiv.org/html/2409.12604v1](https://arxiv.org/html/2409.12604v1)  
3. Gradient descent \- Wikipedia, accessed November 23, 2025, [https://en.wikipedia.org/wiki/Gradient\_descent](https://en.wikipedia.org/wiki/Gradient_descent)  
4. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization \- Neural Dynamics and Computation Lab, accessed November 23, 2025, [https://ganguli-gang.stanford.edu/pdf/14.SaddlePoint.NIPS.pdf](https://ganguli-gang.stanford.edu/pdf/14.SaddlePoint.NIPS.pdf)  
5. Functions in higher dimensional space don't tend to have local minima / maxima? \- Reddit, accessed November 23, 2025, [https://www.reddit.com/r/math/comments/10nwq96/functions\_in\_higher\_dimensional\_space\_dont\_tend/](https://www.reddit.com/r/math/comments/10nwq96/functions_in_higher_dimensional_space_dont_tend/)  
6. IFT 6085 \- Lecture 27 SGD Escapes Saddle Points \- Ioannis Mitliagkas, accessed November 23, 2025, [http://mitliagkas.github.io/ift6085-2019/ift-6085-bonus-lecture-saddle-points-notes.pdf](http://mitliagkas.github.io/ift6085-2019/ift-6085-bonus-lecture-saddle-points-notes.pdf)  
7. gradient descent \- How can it be trapped in a saddle point? \- Stats StackExchange, accessed November 23, 2025, [https://stats.stackexchange.com/questions/278104/how-can-it-be-trapped-in-a-saddle-point](https://stats.stackexchange.com/questions/278104/how-can-it-be-trapped-in-a-saddle-point)  
8. on large-batch training for deep learning: generalization gap and sharp minima \- arXiv, accessed November 23, 2025, [https://arxiv.org/pdf/1609.04836](https://arxiv.org/pdf/1609.04836)  
9. What is Edge of Stability? \- About, accessed November 23, 2025, [https://eregis.github.io/blog/2025/09/08/edge-of-stability.html](https://eregis.github.io/blog/2025/09/08/edge-of-stability.html)  
10. Difference between Batch Gradient Descent and Stochastic Gradient Descent \- GeeksforGeeks, accessed November 23, 2025, [https://www.geeksforgeeks.org/machine-learning/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/](https://www.geeksforgeeks.org/machine-learning/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/)  
11. Gradient Descent vs. Mini-Batch Gradient Descent vs. Stochastic Gradient Descent: An Expert Comparison \- LUNARTECH, accessed November 23, 2025, [https://www.lunartech.ai/blog/gradient-descent-vs-mini-batch-gradient-descent-vs-stochastic-gradient-descent-an-expert-comparison](https://www.lunartech.ai/blog/gradient-descent-vs-mini-batch-gradient-descent-vs-stochastic-gradient-descent-an-expert-comparison)  
12. Train longer, generalize better: closing the generalization gap in large batch training of neural networks \- NIPS papers, accessed November 23, 2025, [http://papers.neurips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf](http://papers.neurips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf)  
13. accessed November 23, 2025, [https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/\#:\~:text=Nesterov%20Accelerated%20Gradient%20(NAG)\&text=NAG%20is%20considered%20more%20efficient,better%20performance%20in%20some%20cases.](https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/#:~:text=Nesterov%20Accelerated%20Gradient%20\(NAG\)&text=NAG%20is%20considered%20more%20efficient,better%20performance%20in%20some%20cases.)  
14. Nesterov Accelerated Gradient and Momentum, accessed November 23, 2025, [https://jlmelville.github.io/mize/nesterov.html](https://jlmelville.github.io/mize/nesterov.html)  
15. On the importance of initialization and momentum in deep learning, accessed November 23, 2025, [https://proceedings.mlr.press/v28/sutskever13.pdf](https://proceedings.mlr.press/v28/sutskever13.pdf)  
16. Is PyTorch's Nesterov Momentum Implementation Wrong? \- Towards Data Science, accessed November 23, 2025, [https://towardsdatascience.com/is-pytorchs-nesterov-momentum-implementation-wrong-5dc5f5032008/](https://towardsdatascience.com/is-pytorchs-nesterov-momentum-implementation-wrong-5dc5f5032008/)  
17. SGD — PyTorch 2.9 documentation, accessed November 23, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)  
18. A Deep Dive into Optimizers in Deep Learning: Roles, Mathematics, Applications and Pseudo Python Code | by Aashish Singh | Medium, accessed November 23, 2025, [https://medium.com/@aashish.singh2k8/a-deep-dive-into-optimizers-in-deep-learning-roles-mathematics-applications-and-pseudo-python-5d8ea1b566dd](https://medium.com/@aashish.singh2k8/a-deep-dive-into-optimizers-in-deep-learning-roles-mathematics-applications-and-pseudo-python-5d8ea1b566dd)  
19. Adagrad Optimizer in Deep Learning \- GeeksforGeeks, accessed November 23, 2025, [https://www.geeksforgeeks.org/machine-learning/intuition-behind-adagrad-optimizer/](https://www.geeksforgeeks.org/machine-learning/intuition-behind-adagrad-optimizer/)  
20. Adaptive Learning Rates, Inference, and Algorithms other than SGD \- CS@Cornell, accessed November 23, 2025, [https://www.cs.cornell.edu/courses/cs6787/2019fa/lectures/Lecture8.pdf](https://www.cs.cornell.edu/courses/cs6787/2019fa/lectures/Lecture8.pdf)  
21. Adagrad and Adadelta Optimizer: In-Depth Explanation | by Vijay Sharma \- Medium, accessed November 23, 2025, [https://visharma1.medium.com/adagrad-and-adadelta-optimizer-in-depth-explanation-6d0ad2fdf22](https://visharma1.medium.com/adagrad-and-adadelta-optimizer-in-depth-explanation-6d0ad2fdf22)  
22. Stabilizing Backpropagation in 16-bit Neural Training with Modified Adam Optimizer \- arXiv, accessed November 23, 2025, [https://arxiv.org/html/2307.16189v8](https://arxiv.org/html/2307.16189v8)  
23. RMSProp \- Cornell University Computational Optimization Open Textbook \- Optimization Wiki, accessed November 23, 2025, [https://optimization.cbe.cornell.edu/index.php?title=RMSProp](https://optimization.cbe.cornell.edu/index.php?title=RMSProp)  
24. Intro to optimization in deep learning: Momentum, RMSProp and Adam | DigitalOcean, accessed November 23, 2025, [https://www.digitalocean.com/community/tutorials/intro-to-optimization-momentum-rmsprop-adam](https://www.digitalocean.com/community/tutorials/intro-to-optimization-momentum-rmsprop-adam)  
25. ADADELTA: An Adaptive Learning Rate Method \- ML Explained, accessed November 23, 2025, [https://ml-explained.com/blog/adadelta-explained](https://ml-explained.com/blog/adadelta-explained)  
26. \[1212.5701\] ADADELTA: An Adaptive Learning Rate Method \- ar5iv \- arXiv, accessed November 23, 2025, [https://ar5iv.labs.arxiv.org/html/1212.5701](https://ar5iv.labs.arxiv.org/html/1212.5701)  
27. A short note on the AdaDelta algorithm. \- Anastasios Kyrillidis, accessed November 23, 2025, [https://akyrillidis.github.io/notes/AdaDelta](https://akyrillidis.github.io/notes/AdaDelta)  
28. AdaDelta Explained. Let's start with a simple question: Why… | by Amit Yadav | Medium, accessed November 23, 2025, [https://medium.com/@amit25173/adadelta-explained-89a0eecc85cb](https://medium.com/@amit25173/adadelta-explained-89a0eecc85cb)  
29. Gradient Descent With Adadelta from Scratch \- MachineLearningMastery.com, accessed November 23, 2025, [https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/](https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/)  
30. The Math behind Adam Optimizer | Towards Data Science, accessed November 23, 2025, [https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b/](https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b/)  
31. Learn Adam Optimizer and Bias Correction | Adaptive Methods \- Codefinity, accessed November 23, 2025, [https://codefinity.com/courses/v2/c72b49e0-c32c-4728-9036-593d70229393/1052ec71-bcce-4a89-a85f-f3d83d8fc9b2/9a027a57-aa2f-4bc1-9604-94eed7f12cff](https://codefinity.com/courses/v2/c72b49e0-c32c-4728-9036-593d70229393/1052ec71-bcce-4a89-a85f-f3d83d8fc9b2/9a027a57-aa2f-4bc1-9604-94eed7f12cff)  
32. epsilon parameter in Adam opitmizer \- Stack Overflow, accessed November 23, 2025, [https://stackoverflow.com/questions/57824804/epsilon-parameter-in-adam-opitmizer](https://stackoverflow.com/questions/57824804/epsilon-parameter-in-adam-opitmizer)  
33. ε, A Nuisance No More | Zack Nado, accessed November 23, 2025, [https://zna.do/epsilon](https://zna.do/epsilon)  
34. \[PDF\] Adam vs. SGD: Closing the generalization gap on image classiﬁcation | Semantic Scholar, accessed November 23, 2025, [https://www.semanticscholar.org/paper/2bd382601b02bb8ccc84e6023a03d4b3f952652f](https://www.semanticscholar.org/paper/2bd382601b02bb8ccc84e6023a03d4b3f952652f)  
35. Why is AdamW Often Superior to Adam with L2-Regularization in Practice? \- GeeksforGeeks, accessed November 23, 2025, [https://www.geeksforgeeks.org/deep-learning/why-is-adamw-often-superior-to-adam-with-l2-regularization-in-practice/](https://www.geeksforgeeks.org/deep-learning/why-is-adamw-often-superior-to-adam-with-l2-regularization-in-practice/)  
36. Adam vs. AdamW: Understanding Weight Decay and Its Impact on Model Performance, accessed November 23, 2025, [https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1](https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1)  
37. \[1711.05101\] Decoupled Weight Decay Regularization \- arXiv, accessed November 23, 2025, [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)  
38. Decoupled Weight Decay Regularization: Bye Bye Adam Optimizer \- Origins AI, accessed November 23, 2025, [https://originshq.com/blog/decoupled-weight-decay-regularization-bye-bye-adam-optimizer/](https://originshq.com/blog/decoupled-weight-decay-regularization-bye-bye-adam-optimizer/)  
39. Why Pytorch does not correct regularization in it's optimizers? \- Site Feedback, accessed November 23, 2025, [https://discuss.pytorch.org/t/why-pytorch-does-not-correct-regularization-in-its-optimizers/41262](https://discuss.pytorch.org/t/why-pytorch-does-not-correct-regularization-in-its-optimizers/41262)  
40. LR not decaying for pytorch AdamW even after hundreds of epochs \- Stack Overflow, accessed November 23, 2025, [https://stackoverflow.com/questions/78752899/lr-not-decaying-for-pytorch-adamw-even-after-hundreds-of-epochs](https://stackoverflow.com/questions/78752899/lr-not-decaying-for-pytorch-adamw-even-after-hundreds-of-epochs)  
41. Hessian Free Optimization \- Andrew Gibiansky, accessed November 23, 2025, [https://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/](https://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/)  
42. Deep learning via Hessian-free optimization \- Department of Computer Science, University of Toronto, accessed November 23, 2025, [https://www.cs.toronto.edu/\~jmartens/docs/Deep\_HessianFree.pdf](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)  
43. Introducing K-FAC. A Second-Order Optimization Method for… | by Kazuki Osawa | TDS Archive | Medium, accessed November 23, 2025, [https://medium.com/data-science/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414](https://medium.com/data-science/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414)  
44. KRONECKER-FACTORED CURVATURE APPROXIMA- TIONS FOR RECURRENT NEURAL NETWORKS \- OpenReview, accessed November 23, 2025, [https://openreview.net/pdf?id=HyMTkQZAb](https://openreview.net/pdf?id=HyMTkQZAb)  
45. KFAC explained \- Felix Dangel, accessed November 23, 2025, [https://fdangel.com/posts/kfac\_explained.html](https://fdangel.com/posts/kfac_explained.html)  
46. Shampoo: Efficient Tensor-Preconditioned Optimizer \- Emergent Mind, accessed November 23, 2025, [https://www.emergentmind.com/topics/shampoo-optimizer](https://www.emergentmind.com/topics/shampoo-optimizer)  
47. Demystifying ML Optimizers: Understanding SGD, Shampoo, and Beyond | by Afaf EL, accessed November 23, 2025, [https://medium.com/@afafel/demystifying-ml-optimizers-understanding-sgd-shampoo-and-beyond-037d94b58239](https://medium.com/@afafel/demystifying-ml-optimizers-understanding-sgd-shampoo-and-beyond-037d94b58239)  
48. Symbolic Discovery of Optimization Algorithms, accessed November 23, 2025, [https://arxiv.org/abs/2302.06675](https://arxiv.org/abs/2302.06675)  
49. \[2310.05898\] Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts \- arXiv, accessed November 23, 2025, [https://arxiv.org/abs/2310.05898](https://arxiv.org/abs/2310.05898)  
50. Lion, new optimizer discovered by Google Brain using genetic algorithms that is purportedly better than Adam(w), in Pytorch \- GitHub, accessed November 23, 2025, [https://github.com/lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch)  
51. Paper Summary \#9 \- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training | Shreyansh Singh, accessed November 23, 2025, [https://shreyansh26.github.io/post/2023-05-28\_sophia\_scalable\_second\_order\_optimizer\_llms/](https://shreyansh26.github.io/post/2023-05-28_sophia_scalable_second_order_optimizer_llms/)  
52. SOPHIA: ASCALABLE STOCHASTIC SECOND-ORDER OPTIMIZER FOR LANGUAGE MODEL PRE-TRAINING \- ICLR Proceedings, accessed November 23, 2025, [https://proceedings.iclr.cc/paper\_files/paper/2024/file/06960915ba8674c7a898ec0b472b80ff-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/06960915ba8674c7a898ec0b472b80ff-Paper-Conference.pdf)  
53. Deriving Muon \- Jeremy Bernstein, accessed November 23, 2025, [https://jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon)  
54. Muon: An optimizer for hidden layers in neural networks \- Keller Jordan blog, accessed November 23, 2025, [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)  
55. Newton-Schulz \- docs.modula.systems, accessed November 23, 2025, [https://docs.modula.systems/algorithms/newton-schulz/](https://docs.modula.systems/algorithms/newton-schulz/)  
56. Understanding the Muon Optimizer: A Game-Changer for Large Language Model Training, accessed November 23, 2025, [https://jehillparikh.medium.com/understanding-the-muon-optimizer-a-game-changer-for-large-language-model-training-15316975f0bc](https://jehillparikh.medium.com/understanding-the-muon-optimizer-a-game-changer-for-large-language-model-training-15316975f0bc)  
57. Understanding Edge-of-Stability Training Dynamics with a Minimalist Example \- arXiv, accessed November 23, 2025, [https://arxiv.org/abs/2210.03294](https://arxiv.org/abs/2210.03294)  
58. Grokking Phase Transition in Neural Nets \- Emergent Mind, accessed November 23, 2025, [https://www.emergentmind.com/topics/grokking-phase-transition](https://www.emergentmind.com/topics/grokking-phase-transition)  
59. \[2206.04817\] The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon \- arXiv, accessed November 23, 2025, [https://arxiv.org/abs/2206.04817](https://arxiv.org/abs/2206.04817)  
60. Thesis (B.Sc. / M.Sc.) Investigating Grokking Phenomena: Neural Network Learning Dynamics through Interpretable \- etit.tu-darmstadt.de, accessed November 23, 2025, [https://www.etit.tu-darmstadt.de/media/bcs/masterarbeiten/documents\_1/ThesisGrokking.pdf](https://www.etit.tu-darmstadt.de/media/bcs/masterarbeiten/documents_1/ThesisGrokking.pdf)  
61. Why Warmup the Learning Rate? Underlying Mechanisms and Improvements \- arXiv, accessed November 23, 2025, [https://arxiv.org/html/2406.09405v1](https://arxiv.org/html/2406.09405v1)  
62. In the context of Deep Learning, what is training warmup steps, accessed November 23, 2025, [https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps](https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps)
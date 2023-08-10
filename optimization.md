---
title: 'Optimization Methods in Machine Learning'
date: 2023-08-25
permalink: /posts/2023/08/Optim/
tags:
  - Gradient Descent
  - Newton Method
  - Lagrange
---

We can divide optimization problems into two categories: conditional optimization problems and unconditional optimization problems. For unconditional optimization problems, three commonly used methods are:
1. Gradient descent: Solving the first-order derivative is actually using Taylor's first-order expansion to approximate the optimal solution
2. L-BFGS: Solving the second order derivative is actually using Taylor's second order expansion approximation.
3. Improved iterative scaling (IIS).
For conditional optimization problems, we use Lagrange Multiplier when the condition cannot be directly substitute into the optimization function.

Gradient Descent
======
<details><summary>CLICK ME</summary>

Assume $f(x)$ has a continuous first derivative on $R^n$, the target is to find $x^*$ that minimize $f(x)$:
$$
\min_{x \in R^n} f(x)
$$
Gradient descent starts from initial value $x_0$, iteratively update $x$ in the direction of the negative gradient, which is the direction that makes the function value decrease the fastest. First order Taylor expansion of $f(x)$ at k-th iteration value $x_k$:
$$
f(x)=f(x_k)+g^T_k(x-x_k)
$$
where $g_k=\nabla f(x_k)$ is the gradient of $f(x)$ at $x_k$. The (k+1)-th iteration:
$$
x_{k+1} \leftarrow x_k - \lambda_k g_k
$$
where $\lambda_k$ is update step size, determined by
$$
f(x_k-\lambda_k g_k)=\min_{\lambda \geq 0} f(x_k-\lambda g_k)
$$
$\lambda_k$ is learning rate in deep learning, which updates according to learning rate schedule given the initial learning rate. <br>In deep neural networks, we use backpropagation to calculate gradient, which follows the chain rule of differentiation. $x$ refers to the weights of a neural network. $f(x)$ is the error function betweem predictions and ground truth. When the dataset is big, standard gradient descent takes too much time since one update needs predictions of the whole training set examples. **Batch gradient descent** randomly draw $M$ samples from the training set to update iteration,
which yields similar results but much faster than the standard GD. When batch size $M=1$, update happens whenever a new sample comes in, we call it **Stochastic Gradient Gescent (SGD)**. It is not effcient if we use SGD at the beginning of training. Usually we use SGD at the last training stage when the loss is low already and we want to further reduce it, try to jump out of the local minimum. Note that the SGD optimizer in Pytorch is SGD only if the batch size is set to 1, or it is actually mini-batch gradient descent. 
<br>However, SGD suffers in the following scenarios: 
* Error surface has high curvature, many deep ravines on the hyper surface of $f(x)$ trap the result in a local optimum.
* When it comes to the flat area of the hyper surface, we get small but consistent gradients, which slows down the model convergence speed.
* The gradients are very noisy, $x$ is updated along a tortuous route and hard to reach convergence.

In order to suppress the problems listed above, **gradient descent with Momentum** believes that adding inertia can help. It introduces a first-order momentum:
$$
m_k= \beta_1 m_{k-1} - \lambda_k g_k\\
x_{k+1}=x_k+ m_k
$$
the current update is more affected by the previous gradients when $\lambda_k$ is smaller than the momentum coefficient $\beta_1$, which is usually set high $\beta_1 = 0.9$. **Nesterov Momentum** applies a lookahead step: first take a step in the direction of the accumulated gradient, then calculate the gradient and make a correction. This helps because while the gradient term always points in the right direction, the momentum term may not. If the momentum term points in the wrong direction or overshoots, the 
gradient can still "go back" and correct it in the same update step:
$$
m_k= \beta_1 m_{k-1}- \lambda_k \nabla f(x_k+\beta_1 m_{k-1})\\
x_{k+1}=x_k+ m_k
$$
So far we have assigned the same learning rate to all features, which is not a good choice if the features vary in importance and frequency. **Adaptive Learning Rate Methods** assign different learning rate to updating model parameters. For weights that are updated frequently, we have accumulated a lot of knowledge about it, and we don't want to be affected too much by a single sample, so the learning rate is smaller; for parameters that are updated occasionally, we know too little information, and we hope to learn more from the samples that appear, so the learning rate is larger. **AdaGrad** downscales a model parameter by square-root of sum of squares of all its historical gradients, decrease the learning rate when parameters updated frequently (larger square-root):
$$
V_{k}=V_{k-1}+g_k^2\\
x_{k+1}=x_k-\frac{\lambda_k}{\delta+\sqrt V_k}g_k
$$
where $V_k$ is the sum of squares of all its historical gradients. Add a small positive value $\delta$ in case of the denominator being zero when $V_k=0$. AdaGrad is good when the objective is convex, but it shrink the learning rate too aggressively. Since $\sqrt V_k$ is monotonically increasing, learning rate of some parameters may be reduced to zero before model parameters approaching the global optimum $x^*$. **RMSProp** performs better than AdaGrad in non-convex settings by accumulating an exponentially decaying average of the squares of recent gradients instead of all history gradients:
$$
V_k=\rho V_{k-1}+(1-\rho)g_k^2\\
x_{k+1}=x_k-\frac{\lambda_k}{\delta+\sqrt V_k}g_k
$$
where $\rho \in [0,1)$. We can further add momentum to RMSProp, here is RMSProp with Nesterov Momentum:
$$
\hat g_k= \nabla f(x_k+\beta_2 m_{k-1})\\
V_k=\rho V_{k-1}+(1-\rho) \hat g_k^2\\
m_k= \beta_2 m_{k-1}- \frac{\lambda_k}{\delta + \sqrt V_k} \hat g_k\\
x_{k+1}=x_k+ m_k
$$
Here we just form momentum $m_k$ the same way RMSProp updates weights. Adam is like RMSProp with Momentum but with bias correction terms for the first and second moments. The bias correction compensates for the fact that the first 
and second moments are initialized at zero and need some time to “warm up”. Adam performs exponentially moving average on momentum:
$$
m_k=\beta_2 m_{k-1}+(1-\beta_2)g_k
$$
$x$ is updated according to approximately the mean of last $\frac{\beta_2}{1-\beta_2}$ gradients. Follow RMSProp:
$$
V_k=\rho V_{k-1}+(1-\rho) g_k^2
$$ 
and add bias correction:
$$
\hat m_k = \frac{m_k}{1-t_1}\\
\hat V_k = \frac{V_k}{1-t_2}\\
$$
update the weights:
$$
x_{k+1}=x_k-\lambda_k \frac{\hat m_k}{\delta + \sqrt{\hat V_k}}
$$

Apart from method mentioned above, there are many other adaptive learning rate methods such as NAdam, AdamW not included in this blog.
</details>
<br>

Newton Method
======
<details><summary>CLICK ME</summary>

</details>
<br>

Lagrange Multiplier
======
<details><summary>CLICK ME</summary>

</details>
<br>
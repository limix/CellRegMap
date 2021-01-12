K = GGt, Sigma = EEt
M = K*Sigma

Let & be the Hadamard product.

We known that:

  rank(A&B) <= rank(A) * rank(B)

Let

  K     = U @ Dk @ Ut
  Sigma = V @ Ds @ Vt

for diagonal matrices Dk and Ds.

K & Sigma = K & (Sum_i vi * lambda_i * vi^T ) = Sum_i (K & (vi @ lambda_i @ vi^T))
                                              = Sum_i ( Lki @ Lki^T )

Let e = |1 1 ... 1|. We have

  K & (vi @ lambda_i @ vi^T) = K & (ui @ ui^T)
                             = K & (Diag(ui) @ e @ e^T @ Diag(ui))
                             = Diag(ui) @ (K & (e @ e^T)) @ Diag(ui) (See (2.10), [1])
                             = Diag(ui) @ K @ Diag(ui)
                             = Diag(ui) @ G @ Gt @ Diag(ui)
                             = Lki @ Lki^T

for Lki = Diag(ui) @ G.

Trial
-----

```python
  import numpy as np

  K = np.random.randn(3, 3)
  K = K @ K.T

  # rank-2 symmetric matrix
  u0 = np.random.randn(3, 1)
  u1 = np.random.randn(3, 1)
  Sigma = u0 @ u0.T + u1 @ u1.T
  sigma, U = np.linalg.eigh(Sigma)

  tmp = sigma[1] * U[:, [1]] @ U[:, [1]].T + sigma[2] * U[:, [2]] @ U[:, [2]].T
  # Close to zero
  print(tmp - Sigma)

  u1 = np.sqrt(sigma[1]) * U[:, [1]]
  u2 = np.sqrt(sigma[2]) * U[:, [2]]

  tmp = u1 @ u1.T + u2 @ u2.T - Sigma
  # Close to zero
  print(tmp)

  e = np.ones(3)

  G = np.linalg.cholesky(K)

  Lk1 = np.diag(u1.ravel()) @ G
  Lk2 = np.diag(u2.ravel()) @ G

  # Close to zero
  print(Lk1 @ Lk1.T + Lk2 @ Lk2.T - K * Sigma)
```

Goal
----

- tr(K & Sigma) (there is a property)
- det(K & Sigma)
- (K & Sigma)^{-1} @ x

References
----------

[1] Hadamard Products and Multivariate Statistical Analysis.
    https://core.ac.uk/download/pdf/82272966.pdf

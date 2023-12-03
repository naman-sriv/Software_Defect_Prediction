import numpy as np
import numpy.linalg as la

def soft_thresholding(y: np.ndarray, mu: float):
    return np.sign(y) * np.clip(np.abs(y) - mu, a_min=0, a_max=None)

def svd_shrinkage(y: np.ndarray, tau: float):
    U, s, Vh = np.linalg.svd(y, full_matrices=False)
    s_t = soft_thresholding(s, tau)
    return U.dot(np.diag(s_t)).dot(Vh)

class RobustPCA:
    def __init__(self, lmb: float, mu_0: float=1e-5, rho: float=2, tau: float=10, 
                 max_iter: int=10, tol_rel: float=1e-3):
        assert mu_0 > 0
        assert lmb > 0
        assert rho > 1
        assert tau > 1
        assert max_iter > 0
        assert tol_rel > 0
        self.mu_0_ = mu_0
        self.lmb_ = lmb
        self.rho_ = rho
        self.tau_ = tau
        self.max_iter_ = max_iter
        self.tol_rel_ = tol_rel
        
    def fit(self, X: np.ndarray):

        assert X.ndim == 2
        mu = self.mu_0_
        Y = X / self._J(X, mu)
        S = np.zeros_like(X)
        S_last = np.empty_like(S)
        for k in range(self.max_iter_):
            # Solve argmin_L ||X - (L + S) + Y/mu||_F^2 + (lmb/mu)*||L||_*
            L = svd_shrinkage(X - S + Y/mu, 1/mu)
            
            # Solve argmin_S ||X - (L + S) + Y/mu||_F^2 + (lmb/mu)*||S||_1
            S_last = S.copy()
            S = soft_thresholding(X - L + Y/mu, self.lmb_/mu)
            
            # Update dual variables Y <- Y + mu * (X - S - L)
            Y += mu*(X - S - L)
            r, h = self._get_residuals(X, S, L, S_last, mu)
            
            # Check stopping cirteria
            tol_r, tol_h = self._update_tols(X, L, S, Y)
            if r < tol_r and h < tol_h:
                break
                
            # Update mu
            mu = self._update_mu(mu, r, h)
            
        return L, S
            
    def _J(self, X: np.ndarray, lmb: float):
        return max(np.linalg.norm(X), np.max(np.abs(X))/lmb)
    
    @staticmethod
    def _get_residuals(X: np.ndarray, S: np.ndarray, L: np.ndarray, S_last: np.ndarray, mu: float):
        primal_residual = la.norm(X - S - L, ord="fro")
        dual_residual = mu*la.norm(S - S_last, ord="fro")
        return primal_residual, dual_residual
    
    def _update_mu(self, mu: float, r: float, h: float):
        if r > self.tau_ * h:
            return mu * self.rho_
        elif h > self.tau_ * r:
            return mu / self.rho_
        else:
            return mu
        
    def _update_tols(self, X, S, L, Y):
        tol_primal = self.tol_rel_ * max(la.norm(X), la.norm(S), la.norm(L))
        tol_dual = self.tol_rel_ * la.norm(Y)
        return tol_primal, tol_dual
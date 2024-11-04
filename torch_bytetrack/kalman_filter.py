import torch
from typing import *


class KalmanFilter:
    def __init__(
            self,
            A: torch.Tensor,
            H: torch.Tensor,
            Q: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]=None,
            R: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]=None,
            B: Optional[torch.Tensor]=None,
            device: str="cpu"
    ):
        r"""
            A (torch.Tensor): 
                Model matrix that describe the (linear) system dynamics
            H (torch.Tensor):
                Measurement / observation matrix to map state space to measurement / observation
            R (Callable | torch.Tensor):
                Measurement / observation uncertainty / covariance matrix
            Q (Callable | torch.Tensor):
                Process noise covariance matrix
            B (torch.Tensor):
                Control input transition matrix (optional)
        """
        self.A = A.to(device)
        self.H = H.to(device)
        self.Q = Q.to(device) if torch.is_tensor(Q) else Q
        self.R = R.to(device) if torch.is_tensor(R) else R
        self.B = B.to(device) if torch.is_tensor(B) else B
        self.I = torch.eye(self.A.shape[0], dtype=self.A.dtype, device=device)

    def predict(
            self, 
            x: torch.Tensor, 
            p: torch.Tensor, 
            u: Optional[torch.Tensor]=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        params:
        ----------------------
        x:(torch.Tensor)
            corrected state space value from the previous timstep
        p:(torch.Tensor)
            corrected state covariance matrix
        u:(torch.Tensor)
            control input at the current timestep (optional)
        
        return
        ----------------------
        newly predicted x (state space) and p covariance matrix values Tuple[torch.Tensor, torch.Tensor]
        """
        Q = self.Q if isinstance(self.Q, torch.Tensor) else self.Q(x)
        if x.device != self.A.device:
            x.to(self.A.device)
        if p.device != self.A.device:
            p.to(self.A.device)
        x = self.A @ x
        if torch.is_tensor(u) and torch.is_tensor(self.B):
            x += (self.B @ u)
        p = (self.A @ p @ self.A.T) + Q
        return x, p
    
    def update(self, x: torch.Tensor, p: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        params
        ----------------------
        x:(torch.Tensor)
            predicted state value
        p:(torch.Tensor)
            predicted state covariance matrix
        z:(torch.Tensor)
            actual observation at current timestep

        return
        ----------------------
        corrected x (state space) and p (covariance matrix) values from the prediction step 
        Tuple[torch.Tensor, torch.Tensor]
        """
        R = self.R if isinstance(self.R, torch.Tensor) else self.R(x)
        Kgain = (p @ self.H.T) @ torch.linalg.tensorinv(((self.H @ p @ self.H.T) + R), ind=1)
        x = x + Kgain @ (z - self.project_to_observation_space(x))
        p = (self.I - (Kgain @ self.H)) @ p
        return x, p

    def project_to_observation_space(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        params
        ----------------------
        H:(torch.Tensor)
            matrix to map from state-space to measurement-space
        x:(torch.Tensor)
            state value
        
        return
        ----------------------
        corresponding observation (torch.Tensor)
        """
        return self.H @ x
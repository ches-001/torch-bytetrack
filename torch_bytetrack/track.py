import torch
from enum import Enum
from .kalman_filter import KalmanFilter
from .utils import xywh2x1y1x2y2
from deep_sort_realtime.deep_sort import track
from typing import *

class TrackState(Enum):
    """
    Newly created tracks are Tentative
    When the enough trajectory samples have been collected the mean becomes Conformed
    When the track has not been matched to a given detection at a given iteration, it is
    Lost
    """
    Tentative = 1
    Confirmed = 2
    Lost = 3

class Track:
    def __init__(
            self, 
            track_id: int,
            mean: torch.Tensor,
            covar: torch.Tensor,
            det_conf: float,
            det_class: Union[str, int],
            max_age: int,
            n_init: int,
        ):
        r"""
        params
        ----------------------
        track_id(int):
            unique id associated with track
        mean(torch.Tensor):
            initial mean of the state distribution (x, y, a, h, vx, vy, va, vh)
        covar(torch.Tensor):
            initial covariance matrix of the state distribution
        det_conf(float):
            detection confidence score
        det_class(int | str):
            detection class
        max_age(int):
            max number of steps that a trajectory can be marked as lost before getting deleted
        n_init(int):
            number of steps before track gets confirmed
        """
        self.track_id = track_id
        # mean: (x, y, a, h, vx, vy, va, vh)
        # (x, y) is the center of the bbox, a is the aspect ratio, h is the height, and (vx, vy, va, vh)
        # are corresponding velocity components
        self.mean = mean.clone()
        self.covar = covar
        self.det_conf = det_conf
        self.det_class = det_class
        self.hits_count = 0
        self.count_until_last_update = 0
        self.status = TrackState.Tentative
        self.max_age = max_age
        self.n_init = n_init

    def predict(self, kalman_filter: KalmanFilter):
        r"""
        predict next state of the track

        params
        ----------------------
        kalman_filter(KalmanFilter):
            kalman filter object used to predict next mean and covar of the trajectory
        """
        self.mean, self.covar = kalman_filter.predict(self.mean, self.covar)
        self.det_conf = None
        self.det_class = None
        self.count_until_last_update += 1

    def update(self, kalman_filter: KalmanFilter, detection: torch.Tensor):
        r"""
        update track

        params
        ----------------------
        kalman_filter(KalmanFilter):
            kalman filter object used to predict next mean and covar of the trajectory
        detection(torch.Tensor):
            new detection used to update the track
        """
        # boxes format (confidence, class_idx, x, y, a, h)
        self.mean, self.covar = kalman_filter.update(self.mean, self.covar, detection[2:])
        self.det_conf = detection[0].item()
        self.det_class = int(detection[1].item())
        self.count_until_last_update = 0
        self.hits_count += 1
        if self.status == TrackState.Tentative and self.hits_count > self.n_init:
            self.status = TrackState.Confirmed

    def mark_as_lost(self):
        self.status = TrackState.Lost

    def is_tentative(self): return self.status == TrackState.Tentative

    def is_confirmed(self): return self.status == TrackState.Confirmed

    def is_lost(self): return self.status == TrackState.Lost

    def to_ltrb(self, device: Union[str, torch.device]="cpu") -> torch.Tensor:
        r"""
        get (x1, y1, x2, y2) box detections, corresponding to (left top and right bottom corners)
        
        params
        ----------------------
        device(str | torch.device)
            device to copy box tensor to

        return
        ----------------------
        box (torch.Tensor)
        """
        box = self.to_xywh(device).unsqueeze(dim=0)
        box = xywh2x1y1x2y2(box).squeeze()
        return box
    
    def to_xywh(self, device: Union[str, torch.device]="cpu") -> torch.Tensor:
        r"""
        get (x, y, w, h) box detections, corresponding to (center coordinates, width and height)
        
        params
        ----------------------
        device(str | torch.device)
            device to copy box tensor to

        return
        ----------------------
        box (torch.Tensor)
        """
        # convert from (x, y, a, h) -> (x, y, w, h)
        box = self.mean[:4].clone()
        box[2] *= box[3]
        return box.to(device)
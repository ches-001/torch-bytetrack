import torch
from .track import Track
from .kalman_filter import KalmanFilter
from .assignment_solver import hungarian_solver
from .utils import x1y1x2y22xywh, compute_iou
from typing import *

class ByteTrack:
    def __init__(
            self, 
            max_age: int=30,
            n_init: int=3,
            tau: float=0.6,
            kf_kwargs: Optional[Dict[str, Any]]=None,
            std_weight_pos: float=1/20,
            std_weight_vel: float=1/160,
            use_ciou: bool=False
        ):
        self.max_age = max_age
        self.n_init = n_init
        self.tau = tau

        self._tracks: List[Track] = []
        self._last_track_id = -1
        self.frame_count = -1
        self.std_weight_pos = std_weight_pos
        self.std_weight_vel = std_weight_vel
        self.use_ciou = use_ciou
        kf_kwargs = kf_kwargs or {}
        self.kalman_filter = self.create_kf(**kf_kwargs)


    def _Q(self, mean: torch.Tensor):
        # create process noise covariance matrix for kalman filter
        # this matrix (according to this function) is designed to change according to the mean at every iteration
        std_pos = [self.std_weight_pos*mean[3], self.std_weight_pos*mean[3], 1e-2, self.std_weight_pos*mean[3]]
        std_vel = [self.std_weight_vel*mean[3], self.std_weight_vel*mean[3], 1e-5, self.std_weight_vel*mean[3]]
        covar = torch.tensor(std_pos+std_vel, device=mean.device).pow(2).diag()
        return covar


    def _R(self, mean: torch.Tensor):
        # create measurement covariance matrix for kalman filter
        # this matrix (according to this function) is designed to change according to the mean at every iteration
        std_pos = [self.std_weight_pos*mean[3], self.std_weight_pos*mean[3], 1e-2, self.std_weight_pos*mean[3]]
        covar = torch.tensor(std_pos, device=mean.device).pow(2).diag()
        return covar

    
    def create_kf(self, **kf_kwargs) -> KalmanFilter:
        # given the measurements [x, y, a, h], thats would include velocity components
        # so that it would be [x, y, a, h, vx, vy, va, vh]. We use a simple linear constant
        # velocity model, denoted by the expression x(t+1) = x(t) + (vx.t), where vx is constant
        # inotherwords, aceleration is 0. We apply this to all values in the state
        measurement_dim = 4
        dt = 1.0
        A = torch.eye(measurement_dim*2)
        for i in range(0, measurement_dim):
            A[i, measurement_dim+i] = dt
        # matrix to map from state space to measurement space
        H = torch.eye(measurement_dim, measurement_dim*2)
        default_kf_kwargs = {"A":A, "H":H, "Q":self._Q, "R":self._R}
        default_kf_kwargs.update(kf_kwargs)
        kalman_filter = KalmanFilter(**default_kf_kwargs)
        return kalman_filter


    def update_tracks(self, detections: torch.Tensor, format: str="xywh") -> List[Track]:
        # ByteTrack paper: https://arxiv.org/pdf/2110.06864
        # check out the paper, and the the pseudo code to follow through with this implementation
        # boxes format (confidence, class_idx, x, y, w, h) or (confidence, class_idx, x1, y1, x2, y2)
        self.frame_count += 1
        assert format in ["xywh", "xyxy"]
        assert detections.ndim == 2
        detections = detections.clone()
        if format == "xyxy":
            detections[:, 2:] = x1y1x2y22xywh(detections[:, 2:])
        device = detections.device

        # seperate detections according to high and low confidences
        DHigh_mask = detections[:, 0] > self.tau
        DHigh = detections[DHigh_mask]
        if self.frame_count == 0:
            self.__add_new_tracks(DHigh)
            return self._tracks
        DLow = detections[~DHigh_mask]
        # init DRemain and indexes for TRemain (currently initialized to be all indexes of tracks), 
        # DRemain are detections in DHigh that were not assigned to any track, similarly TRemains 
        # are tracks that were not assigned to any detections from DHigh. TRemain it will eventually
        # be trimmed to be the indexes of tracks not matched to a corresponding detection from DHigh
        DRemain = torch.tensor([], device=device)
        TRemain_indexes = torch.arange(0, len(self._tracks), 1, device=device)

        # DHigh_ious will eventually be a matrix where the rows correspond to tracks and columns correspond
        # to DHigh detections, here we predict the future state of each track via the kalman filter, then 
        # compute the IoU (CIoU) between each track and all DHigh detections
        DHigh_ious = []
        for i in range(0, len(self._tracks)):
            self._tracks[i].predict(self.kalman_filter)
            ious = compute_iou(self._tracks[i].to_xywh(device).unsqueeze(dim=0), DHigh[:, 2:], use_ciou=self.use_ciou)
            DHigh_ious.append(ious)
        
        if len(DHigh_ious) > 0:
            DHigh_ious = torch.stack(DHigh_ious, dim=0)
            # The Hungarian algorithm is used for performing the assignment between the tracks and detections
            # from DHigh, as such, it expects a square matrix, so a square matrix we shall provide.
            DHigh_ious = self.__max_dim_pad(DHigh_ious)
            DHigh_match_indexes, _ = hungarian_solver(-DHigh_ious)

            # discard assignments where either the rows or cols index is more than the row and col size of the
            # DHigh_ious since the the DHigh_ious matrix has the potential of being non-square.
            DHigh_match_indexes = self.__filter_invalid_assignments(
                DHigh_match_indexes, len(self._tracks), DHigh.shape[0]
            )

            # update the track[i] with matching detections[j] from DHigh
            for i, j in DHigh_match_indexes:
                self._tracks[i].update(self.kalman_filter, self.__measurement_to_state_format(DHigh[j]))

            DRemain = DHigh[~torch.isin(torch.arange(0, DHigh.shape[0], 1, device=device), DHigh_match_indexes[:, 1])]
            TRemain_indexes = TRemain_indexes[~torch.isin(TRemain_indexes, DHigh_match_indexes[:, 0])]

        # compute the IoU (CIoU) of tracks in TRemain with detections in DLow
        DLow_ious = []
        for i in TRemain_indexes:
            ious = compute_iou(self._tracks[i].to_xywh(device).unsqueeze(dim=0), DLow[:, 2:], use_ciou=self.use_ciou)
            DLow_ious.append(ious)
            
        if len(DLow_ious) > 0:
            DLow_ious = torch.stack(DLow_ious, dim=0)
            DLow_ious = self.__max_dim_pad(DLow_ious)
            # match tracks from TRemain to DLow
            DLow_match_indexes, _ = hungarian_solver(-DLow_ious)
            DLow_match_indexes = self.__filter_invalid_assignments(
                DLow_match_indexes, TRemain_indexes.shape[0], DLow.shape[0]
            )

            # Update tracks from TRemain with assigned detections from DLow
            for i, j in DLow_match_indexes:
                self._tracks[TRemain_indexes[i]].update(self.kalman_filter, self.__measurement_to_state_format(DLow[j]))

            # for tracks that were neither associated with detections from DHigh or DLow, we mark them
            # as lost
            lost_indexes = TRemain_indexes[~torch.isin(TRemain_indexes, DLow_match_indexes[:, 1])]
            self.__mark_tracks_as_lost(lost_indexes)

        self.__delete_untracked_tracks()
        # initialize new tracks with detections from DRemain
        self.__add_new_tracks(DRemain)
        return self._tracks


    def __max_dim_pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == x.shape[1]:
            return x
        if x.shape[0] < x.shape[1]:
            x = torch.cat([x, torch.zeros(x.shape[1]-x.shape[0], x.shape[1])], dim=0)
        elif x.shape[0] > x.shape[1]:
            x = torch.cat([x, torch.zeros(x.shape[0], x.shape[0]-x.shape[1])], dim=1)
        return x
    

    def __filter_invalid_assignments(self, match_indexes: torch.Tensor, row_size: int, col_size: int) -> torch.Tensor:
        match_indexes = match_indexes[match_indexes[:, 0] < row_size]
        match_indexes = match_indexes[match_indexes[:, 1] < col_size]
        return match_indexes
    

    def __mark_tracks_as_lost(self, indexes: torch.Tensor):
        for i in indexes:
            self._tracks[i].mark_as_lost()
        

    def __delete_untracked_tracks(self):
        _filter = lambda track : (not track.is_lost()) and track.count_until_last_update < self.max_age
        self._tracks = list(filter(_filter, self._tracks))


    def __measurement_to_state_format(self, measurement: torch.Tensor) -> torch.Tensor:
        # convert (x, y, w, h) -> (x, y, a, h)
        # convert (conf, cls_idx, x, y, w, h) -> (conf, cls_idx, x, y, a, h)
        measurement = measurement.clone()
        measurement[-2] /= measurement[-1]
        return measurement
    

    def __init_track(self, measurement: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        measurement = self.__measurement_to_state_format(measurement)
        mean = torch.concat([measurement, torch.zeros_like(measurement)], dim=0)
        std_pos = [
            self.std_weight_pos*measurement[3], 
            self.std_weight_pos*measurement[3], 
            1e-2, 
            self.std_weight_pos*measurement[3]
        ]
        std_pos = torch.tensor(std_pos, device=measurement.device) * 2
        std_pos[2] = 1e-2
        std_vel = [
            self.std_weight_vel*measurement[3], 
            self.std_weight_vel*measurement[3], 
            1e-5, 
            self.std_weight_vel*measurement[3]
        ]
        std_vel = torch.tensor(std_vel, device=measurement.device) * 10
        std_vel[2] = 1e-5
        covar = torch.concat([std_pos, std_vel]).pow(2).diag()
        return mean, covar
    

    def __add_new_tracks(self, detections: torch.Tensor):
        for i in range(0, detections.shape[0]):
            self._last_track_id += 1
            det = detections[i]
            mean, covar = self.__init_track(det[2:])
            new_track = Track(
                track_id=self._last_track_id,
                mean=mean,
                covar=covar,
                det_conf=det[0].item(),
                det_class=int(det[1].item()),
                max_age=self.max_age,
                n_init=self.n_init,
            )
            self._tracks.append(new_track)
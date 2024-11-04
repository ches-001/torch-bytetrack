# Torch-ByteTrack

The [ByteTrack Algorithm](https://arxiv.org/pdf/2110.06864) is a pretty interesting algorithm for tracking objects across frames of a continuous video feed. This module contains a pytorch implementation for ByteTrack and it is fairly simple to follow through with the procedures involved

1. Initialize the tracks buffer `T`, this buffer stores all our tracked detections

2. Get the list of detections for the given frame via an object detection model.

3. Seperate the low confidence detections `(DLow)` from the high confidence ones `(DHigh)` based on a threshold value $\tau$. If the iteration step is at the very first frame of the video, `T` is initialized with detections from `DHigh` and returned.

4. Using the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter), predict the next trajectory of each track. The Kalman filter used in this implementation uses a linear constant velocity model to project the current state in time, here the state is represented as $(x, y, a, h, vx, vy, va, vh)$ where $x$ and $y$ are box centers, $a$ is the aspect ratio and $h$ is height, and the rest are corresponding velocity components

5. Compute the IoU (CIoU in this implementation) between each track and all current detections, this will result in an $m \mathbf{x} n$ matrix, where $m$ is the number of tracks in `T` and $n$ is the number of detections in the current frame

6. Perform an optimal assignment with the negative of the IoU matrix to match each track to their corresponding detections with the [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm). The Hungarian (Kuhn) algorithm expects a square matrix, as such we pad the axis of our matrix with the least size with a zero matrix of size $max(m, n) - min(m, n)$ along the least sized dimension. We use the negative of the IoU matrix because the Hungarian algorithm aims to find the best possible assignment to minimise cost, our goal is the maximise the IoU scores for the matching.

7. From the optimal assignment solution, remove all invalid indexes caused due to padding.

8. Update each track with with the corresponding detections from the optimal assignment (this also involves the update/correction step of the Kalman Filter)

9. The tracks that were not assigned to any detections in `DHigh` will be initialized as `TRemain`, and the detections from `DHigh` not assigned to any track will be initialized as `DRemain`

10. Compute the IoUs of each track in `TRemain` with detections in `DLow`, this will produce a matrix of size $a \mathbf{x} b$ where $a$ is the number of tracks in `TRemain` and $b$ is the number of detections in `DLow`

11. Perform an optimal assignment with the negtive of the newly computed IoU matrix to match each track in `TRemain` to a corresponding detection in `DLow`. 

12. Remove all invalid assignments caused by padding.

13. Update the tracks in `TRemain` with its assigned detection from `DLow`

14. For the tracks in `TRemain` not assigned to a detection in `DLow`, we mark them as "lost tracks", and only delete them when the track has not been updated after a certain number of iterations `(max_age)`.

15. For detections in `DRemain` we initialize them as new tracks to be added to `T`.

# How to Use
```python
from torch_bytetrack import ByteTrack

tracker = ByteTrack()
detections = mydetector(frame)

tracks = tracker.update_tracks(detections, format="xyxy") # format can be xywh or xyxy
for track in tracks:
    if track.is_confirmed():
        box = track.to_ltrb()
```
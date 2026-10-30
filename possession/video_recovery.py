"""Conservative local video candidates; never enabled by production inference.

OpenCV is optional and imported only when constructing a tracker. The detector
combines a compact bright component, an anchor appearance template, and physical
motion gates. It is not a pretrained football detector and can abstain completely.
"""
from dataclasses import dataclass
import numpy as np


def transform(matrix, points, inverse=False):
    """Map normalized pitch coordinates to pixels, or pixels back to pitch."""
    matrix = np.asarray(matrix, dtype=float)
    points = np.asarray(points, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("A finite 3 by 3 homography is required")
    if points.shape[-1:] != (2,):
        raise ValueError("Points must end in two coordinates")
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("Singular or ill-conditioned homography")
    if inverse:
        matrix = np.linalg.inv(matrix)
    flat = points.reshape(-1, 2)
    out = np.column_stack((flat, np.ones(len(flat)))) @ matrix.T
    valid = np.isfinite(out).all(axis=1) & (np.abs(out[:, 2]) > 1e-8)
    result = np.full((len(flat), 2), np.nan)
    result[valid] = out[valid, :2] / out[valid, 2, None]
    return result.reshape(points.shape)


@dataclass(frozen=True)
class RecoverySettings:
    max_gap_seconds: float = .8
    max_speed_mps: float = 35.
    max_reprojection_error: float = 2.
    minimum_score: float = .77
    minimum_margin: float = .06
    template_radius: int = 6
    maximum_roi_radius: int = 140


class LocalBallTracker:
    """Anchor-initialized high-resolution tracker with explicit abstention.

    Call seed on an observed ball frame, then recover on subsequent missing
    frames in the same continuous shot/half. A gap cannot extend anchor lifetime.
    A new seed is required after cuts, invalid calibration, or long gaps.
    """
    def __init__(self, pitch=(105.3, 68.), fps=25., settings=None):
        import cv2
        self.cv2 = cv2
        self.pitch = np.asarray(pitch, dtype=float)
        self.fps = float(fps)
        self.settings = settings or RecoverySettings()
        if self.pitch.shape != (2,) or (self.pitch <= 0).any() or not np.isfinite(self.pitch).all() or self.fps <= 0:
            raise ValueError("Positive finite pitch dimensions and FPS required")
        self.reset()

    def reset(self):
        self.template = None
        self.last_thumbnail = None
        self.anchor_frame = self.last_frame = None
        self.point = None

    def _usable(self, matrix, valid, error):
        return bool(valid and np.isfinite(error) and 0 <= error <= self.settings.max_reprojection_error
                    and np.asarray(matrix).shape == (3, 3) and np.isfinite(matrix).all()
                    and np.linalg.cond(matrix) <= 1e10)

    def seed(self, frame, image, matrix, point, valid=True, reprojection_error=0.):
        self.reset()
        if not self._usable(matrix, valid, reprojection_error):
            return False
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or (point < -.02).any() or (point > 1.02).any():
            return False
        pixel = transform(matrix, point)
        if not np.isfinite(pixel).all():
            return False
        x, y = np.rint(pixel).astype(int)
        radius = self.settings.template_radius
        if x-radius < 0 or y-radius < 0 or x+radius >= image.shape[1] or y+radius >= image.shape[0]:
            return False
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        patch = gray[y-radius:y+radius+1, x-radius:x+radius+1]
        if float(patch.std()) < 8:
            return False
        self.template = patch.copy()
        self.point = point.copy()
        self.anchor_frame = self.last_frame = int(frame)
        self.last_thumbnail = self.cv2.resize(gray, (64, 36))
        return True

    def recover(self, frame, image, matrix, valid=True, reprojection_error=0.):
        """Return accepted normalized point/score, or an explicit rejection reason."""
        def reject(reason, reset=False):
            if reset:
                self.reset()
            return {"accepted": False, "reason": reason}
        if self.template is None:
            return reject("no_anchor")
        if frame <= self.last_frame:
            return reject("nonmonotonic_frame", True)
        if (frame-self.anchor_frame)/self.fps > self.settings.max_gap_seconds:
            return reject("anchor_expired", True)
        if not self._usable(matrix, valid, reprojection_error):
            return reject("invalid_homography", True)
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        thumbnail = self.cv2.resize(gray, (64, 36))
        if np.mean(np.abs(thumbnail.astype(float)-self.last_thumbnail)) > 40:
            return reject("shot_cut", True)
        center = transform(matrix, self.point)
        if not np.isfinite(center).all():
            return reject("off_camera", True)
        elapsed = (frame-self.last_frame)/self.fps
        metres = min(12., self.settings.max_speed_mps*elapsed + .8)
        probes = self.point + np.array([[metres/self.pitch[0], 0], [0, metres/self.pitch[1]]])
        projected = transform(matrix, probes)
        radius = int(np.clip(np.max(np.linalg.norm(projected-center, axis=1))+12, 18,
                             self.settings.maximum_roi_radius))
        x, y = np.rint(center).astype(int)
        x0, y0 = max(0, x-radius), max(0, y-radius)
        x1, y1 = min(image.shape[1], x+radius+1), min(image.shape[0], y+radius+1)
        crop = image[y0:y1, x0:x1]
        if min(crop.shape[:2]) <= 2*self.settings.template_radius:
            return reject("off_camera", True)
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        local_gray = gray[y0:y1, x0:x1]
        top_hat = self.cv2.morphologyEx(local_gray, self.cv2.MORPH_TOPHAT, np.ones((11,11), np.uint8))
        white = ((hsv[:,:,1] < 110) & (hsv[:,:,2] > 145) & (top_hat > 18)).astype(np.uint8)
        count, _, stats, centroids = self.cv2.connectedComponentsWithStats(white, 8)
        matches = self.cv2.matchTemplate(local_gray, self.template, self.cv2.TM_CCOEFF_NORMED)
        candidates = []
        half = self.settings.template_radius
        for index in range(1, count):
            bx, by, width, height, area = stats[index]
            if not 2 <= area <= 100 or not .4 <= width/max(height,1) <= 2.5 or max(width,height) > 16:
                continue
            cx, cy = centroids[index]
            mx, my = int(round(cx))-half, int(round(cy))-half
            if mx < 0 or my < 0 or mx >= matches.shape[1] or my >= matches.shape[0]:
                continue
            point = transform(matrix, np.array([cx+x0, cy+y0]), inverse=True)
            if not np.isfinite(point).all() or (point < -.02).any() or (point > 1.02).any():
                continue
            distance = np.linalg.norm((point-self.point)*self.pitch)
            if distance > metres:
                continue
            appearance = max(0., float(matches[my,mx]))
            compactness = min(1., area/max(width*height,1))
            contrast = min(1., float(top_hat[int(round(cy)),int(round(cx))])/90)
            score = .65*appearance + .15*compactness + .20*contrast
            candidates.append((score, point, appearance))
        candidates.sort(key=lambda item: -item[0])
        if not candidates or candidates[0][0] < self.settings.minimum_score:
            return reject("weak_visual_evidence")
        if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < self.settings.minimum_margin:
            return reject("ambiguous_candidates")
        score, point, appearance = candidates[0]
        self.point = point
        self.last_frame = int(frame)
        self.last_thumbnail = thumbnail
        return {"accepted": True, "point": point.tolist(), "score": float(score),
                "appearance": float(appearance), "reason": "visual_and_motion_gates"}


class YoloBallTracker(LocalBallTracker):
    """Optional pretrained COCO sports-ball detector over two enlarged local ROIs.

    A locally loaded detector object is supplied by the research runner. No
    network or download occurs here. Model inference runs CPU-only on image
    arrays; detections never leave the local machine.
    """
    def __init__(self, detector, pitch=(105.3,68.), fps=25., confidence=.4):
        super().__init__(pitch, fps)
        self.detector = detector
        self.confidence = float(confidence)
        self.ball_class = next((int(k) for k,v in detector.names.items() if v == 'sports ball'),None)
        if self.ball_class is None or not 0 < self.confidence < 1:
            raise ValueError("Detector must contain sports ball and a valid confidence")

    def seed(self, frame, image, matrix, point, valid=True, reprojection_error=0.):
        self.reset()
        point=np.asarray(point,float)
        if not self._usable(matrix,valid,reprojection_error) or point.shape!=(2,) or not np.isfinite(point).all():
            return False
        pixel=transform(matrix,point)
        if not np.isfinite(pixel).all() or not (0<=pixel[0]<image.shape[1] and 0<=pixel[1]<image.shape[0]):
            return False
        self.point=point.copy();self.anchor_frame=self.last_frame=int(frame)
        self.template=True
        self.last_thumbnail=self.cv2.resize(self.cv2.cvtColor(image,self.cv2.COLOR_BGR2GRAY),(64,36))
        return True

    def recover(self, frame, image, matrix, valid=True, reprojection_error=0.):
        def reject(reason,reset=False):
            if reset:self.reset()
            return {'accepted':False,'reason':reason}
        if self.template is None:return reject('no_anchor')
        if frame<=self.last_frame:return reject('nonmonotonic_frame',True)
        if (frame-self.anchor_frame)/self.fps>self.settings.max_gap_seconds:return reject('anchor_expired',True)
        if not self._usable(matrix,valid,reprojection_error):return reject('invalid_homography',True)
        thumbnail=self.cv2.resize(self.cv2.cvtColor(image,self.cv2.COLOR_BGR2GRAY),(64,36))
        if np.mean(np.abs(thumbnail.astype(float)-self.last_thumbnail))>40:return reject('shot_cut',True)
        center=transform(matrix,self.point)
        if not np.isfinite(center).all():return reject('off_camera',True)
        x,y=np.rint(center).astype(int)
        crops=[];origins=[]
        for radius in (96,192):
            x0,y0=max(0,x-radius),max(0,y-radius)
            x1,y1=min(image.shape[1],x+radius),min(image.shape[0],y+radius)
            if x1-x0<32 or y1-y0<32:continue
            crops.append(image[y0:y1,x0:x1]);origins.append((x0,y0))
        if not crops:return reject('off_camera',True)
        results=self.detector.predict(crops,imgsz=960,conf=.1,classes=[self.ball_class],
                                      device='cpu',verbose=False,save=False,stream=False,max_det=30)
        maximum=min(12.,self.settings.max_speed_mps*(frame-self.last_frame)/self.fps+.8)
        candidates=[]
        for result,(x0,y0) in zip(results,origins):
            boxes=result.boxes
            if boxes is None:continue
            for box,score in zip(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy()):
                width,height=box[2]-box[0],box[3]-box[1]
                if not 1<=width<=28 or not 1<=height<=28 or not .4<=width/height<=2.5:continue
                pixel=np.array([(box[0]+box[2])/2+x0,(box[1]+box[3])/2+y0])
                point=transform(matrix,pixel,inverse=True)
                if not np.isfinite(point).all() or (point<-.02).any() or (point>1.02).any():continue
                if np.linalg.norm((point-self.point)*self.pitch)>maximum:continue
                candidates.append((float(score),point,pixel))
        candidates.sort(key=lambda row:-row[0])
        # Merge the same ball seen by both ROI scales before ambiguity checking.
        distinct=[]
        for candidate in candidates:
            if all(np.linalg.norm(candidate[2]-existing[2])>4 for existing in distinct):distinct.append(candidate)
        if not distinct or distinct[0][0]<self.confidence:return reject('weak_detector_evidence')
        if len(distinct)>1 and distinct[0][0]-distinct[1][0]<.1:return reject('ambiguous_detector_candidates')
        score,point,_=distinct[0]
        self.point=point;self.last_frame=int(frame);self.last_thumbnail=thumbnail
        return {'accepted':True,'point':point.tolist(),'score':score,'reason':'pretrained_detector_and_motion_gates'}


def align_tracking_window(reference, other, start, stop, offsets, pitch=(105.3,68.),
                          reference_valid=None, other_valid=None, minimum_support=40):
    """Conservative local broadcast alignment: other index = reference index+lag.

    Requires dynamic ball movement, valid calibration, low spatial residuals,
    and a unique lag separated from other plausible alignments. A caller must
    estimate each continuous shot separately; this never assumes one global lag.
    No alignment is returned for static, ambiguous or weakly supported windows.
    """
    a, b = np.asarray(reference, float), np.asarray(other, float)
    pitch = np.asarray(pitch, float)
    index = np.arange(max(0, start), min(len(a), stop))
    valid_a = np.isfinite(a).all(axis=1)
    valid_b = np.isfinite(b).all(axis=1)
    if reference_valid is not None:
        valid_a &= np.asarray(reference_valid, bool)
    if other_valid is not None:
        valid_b &= np.asarray(other_valid, bool)
    scored = []
    for lag in sorted(set(int(value) for value in offsets)):
        target = index+lag
        inside = (target >= 0)&(target < len(b))
        left, right = index[inside], target[inside]
        good = valid_a[left]&valid_b[right]
        left, right = left[good], right[good]
        if len(left) < minimum_support:
            continue
        if np.linalg.norm(np.ptp(a[left]*pitch, axis=0)) < 8:
            continue
        distances = np.linalg.norm((a[left]-b[right])*pitch,axis=1)
        if np.mean(distances < 3) < .8:
            continue
        scored.append((float(np.median(distances)), lag, len(left)))
    scored.sort()
    if not scored or scored[0][0] > 1.5:
        return {"accepted": False, "reason": "insufficient_consistent_dynamic_support"}
    best = scored[0]
    alternatives = [row for row in scored[1:] if abs(row[1]-best[1]) > 2]
    if alternatives and alternatives[0][0]-best[0] < .75:
        return {"accepted": False, "reason": "ambiguous_or_replayed_alignment"}
    return {"accepted": True, "offset": best[1], "median_error_m": best[0],
            "support": best[2], "reason": "unique_dynamic_window"}

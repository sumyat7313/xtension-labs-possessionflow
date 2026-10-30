"""Local experimental video-ball recovery with held-out silver-anchor diagnostics."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAMES, GAME
from possession.video_recovery import LocalBallTracker, YoloBallTracker, RecoverySettings, align_tracking_window

YOLO11N_SHA256='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def load_local_detector(path, config_dir):
    """Load pinned official public weights; disable network checks and telemetry."""
    if not path or not Path(path).is_file():raise ValueError('Download official YOLO11n weights locally first')
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest!=YOLO11N_SHA256:raise ValueError('YOLO11n checkpoint differs from pinned official artifact')
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(Path(config_dir).resolve())
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO, settings
    settings.update({'sync':False})
    return YOLO(str(path))


def read_ball(path):
    """Read only the ball observation, never a possession or human label file."""
    result = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for expected, line in enumerate(handle, 1):
            frame, _, ball = line.strip().split(":")
            if int(frame) != expected:
                raise ValueError("ATD must be sequential and video indexed")
            values = np.fromstring(ball.rstrip(";"), sep=",")
            if values.shape != (3,):
                raise ValueError("Malformed ATD ball")
            result.append(values[:2])
    return np.asarray(result)


def homographies(path):
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    if values.ndim != 2 or values.shape[1] != 12 or not np.array_equal(values[:,0], np.arange(len(values))):
        raise ValueError("Homography rows must be complete sequential video frames")
    return values[:,1:10].reshape(-1,3,3), values[:,10], values[:,11].astype(bool)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n",encoding="utf-8")


def metrics(rows):
    accepted = [row for row in rows if row["accepted"]]
    errors = [row["error_m"] for row in accepted]
    return {"masked_frames":len(rows), "accepted_frames":len(accepted),
            "acceptance_fraction":len(accepted)/max(1,len(rows)),
            "within_2m_fraction_all_masked":sum(e <= 2 for e in errors)/max(1,len(rows)),
            "false_acceptance_fraction_accepted":sum(e > 2 for e in errors)/len(errors) if errors else None,
            "median_error_m_accepted":float(np.median(errors)) if errors else None,
            "p90_error_m_accepted":float(np.quantile(errors,.9)) if errors else None,
            "rejections":dict(Counter(row["reason"] for row in rows if not row["accepted"]))}


def run(args):
    import cv2
    started = time.monotonic()
    out = Path(args.out)
    if out.exists():
        raise ValueError("Use a new output directory to preserve pilot evidence")
    out.mkdir(parents=True)
    settings = RecoverySettings()
    protocol = {"schema_version":1,"mode":"silver_masked_observations", "settings":asdict(settings),
                "mask_frames":args.mask_frames,"window_limit":args.windows,
                "sampling":"One fully observed span per fixed 120-second time block; earliest qualifying anchor after block+25 frames.",
                "split":"Alternating disjoint 120-second blocks, even development / odd validation; settings frozen before either is scored.",
                "reference":"Existing automatic ball tracking, not possession labels or independent ball ground truth.",
                "limitations":["Observed-ball spans are easier than genuine missing-ball spans.",
                               "An anchor-initialized compact-bright-blob tracker is not a trained football detector.",
                               "Neither silver accuracy nor video recovery demonstrates better possession spell F1."],
                "source_hashes":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                 ("possession/video_recovery.py","tools/recover_video_ball.py")},
                "production_enabled":False}
    protocol['backend']=args.backend
    if args.backend=='yolo':
        protocol['detector']={'name':'YOLO11n COCO pretrained sports ball', 'weights_sha256':YOLO11N_SHA256,
                              'weights_source':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
                              'upstream_license':'AGPL-3.0 / commercial enterprise alternatives',
                              'roi_radii_pixels':[96,192],'inference_size':960,'minimum_confidence':.4,
                              'minimum_margin':.1,'device':'cpu','fine_tuning':False}
    excluded_blocks=set()
    if args.exclude_sampling:
        previous=json.loads(Path(args.exclude_sampling).read_text())
        excluded_blocks={row['time_block'] for row in previous['windows']}
        protocol['excluded_previous_time_blocks']=sorted(excluded_blocks)
        protocol['excluded_sampling_sha256']=hashlib.sha256(Path(args.exclude_sampling).read_bytes()).hexdigest()
    write_json(out/"protocol.json",protocol)
    detector=load_local_detector(args.weights,out/'ultralytics-config') if args.backend=='yolo' else None
    if detector is not None:
        import importlib.metadata
        write_json(out/'detector-environment.json',{package:importlib.metadata.version(package) for package in
                   ('ultralytics','torch','torchvision','opencv-python','numpy')})
    ball = read_ball(args.atd)
    matrices, error, valid = homographies(args.homography)
    if len(ball)!=len(matrices):
        raise ValueError("ATD and homography frame counts differ")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open local video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if abs(fps-25)>1e-3 or count != len(ball):
        cap.release()
        raise ValueError(f"Video timeline mismatch: fps={fps}, frames={count}, tracking={len(ball)}")
    info = {"fps":fps,"frames":count,"width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),"video_bytes":Path(args.video).stat().st_size,
            "opencv_version":cv2.__version__}
    game=GAMES[GAME]
    blocks=[(a-game['offsets'][half],b-game['offsets'][half]) for half,(a,b) in game['halves'].items()]
    active=np.zeros(len(ball),bool)
    for a,b in blocks: active[a:b+1]=True
    known=np.isfinite(ball).all(axis=1)&(ball>=0).all(axis=1)&(ball<=1).all(axis=1)&active
    usable=known&valid&np.isfinite(error)&(error<=settings.max_reprojection_error)
    windows=[]
    block_frames=120*25
    for block in range((len(ball)+block_frames-1)//block_frames):
        if block in excluded_blocks:continue
        start=block*block_frames+25
        stop=min((block+1)*block_frames,len(ball))-args.mask_frames-1
        for anchor in range(start,stop):
            if usable[anchor:anchor+args.mask_frames+1].all():
                windows.append((anchor,"development" if block%2==0 else "validation",block))
                break
    # Evenly sample the eligible time blocks, rather than only early match time.
    if len(windows)>args.windows:
        windows=[windows[i] for i in np.linspace(0,len(windows)-1,args.windows).astype(int)]
    write_json(out/"sampling.json",{"windows":[{"anchor":a,"split":s,"time_block":b} for a,s,b in windows],
                                     "selected_from_observation_availability_only":True})
    rows=[]; recovered=[]; seed_failures=Counter()
    for index,(anchor,split,block) in enumerate(windows):
        tracker=LocalBallTracker(game['pitch'],fps,settings) if detector is None else YoloBallTracker(detector,game['pitch'],fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES,anchor)
        ok,image=cap.read()
        if not ok: raise ValueError("Cannot decode sampled anchor")
        seeded=tracker.seed(anchor,image,matrices[anchor],ball[anchor],valid[anchor],error[anchor])
        if not seeded: seed_failures[split]+=1
        for frame in range(anchor+1,anchor+args.mask_frames+1):
            ok,image=cap.read()
            if not ok: raise ValueError("Cannot decode masked frame")
            # Crucially, the withheld current/future tracking coordinates are not
            # passed to recover; they enter the metric only after prediction.
            result=tracker.recover(frame,image,matrices[frame],valid[frame],error[frame])
            row={"frame":frame,"split":split,"time_block":block,"accepted":result['accepted'],"reason":result['reason']}
            if result['accepted']:
                point=np.asarray(result['point'])
                row['error_m']=float(np.linalg.norm((point-ball[frame])*game['pitch']))
                row['score']=result['score']
                recovered.append([frame,*point,result['score']])
            rows.append(row)
        print(f"Video pilot {index+1}/{len(windows)} windows processed",flush=True)
    cap.release()
    summary={"schema_version":1,"protocol":protocol,"assets":info,"windows":len(windows),
             "seed_failures":dict(seed_failures),"overall":metrics(rows),
             "splits":{split:metrics([row for row in rows if row['split']==split]) for split in ('development','validation')},
             "elapsed_seconds":round(time.monotonic()-started,3),"production_enabled":False,
             "recommendation":"Research candidate only; do not merge into release tracking based on silver-anchor performance alone."}
    write_json(out/"summary.json",summary)
    write_json(out/"frame-diagnostics.json",rows)
    np.save(out/"recovered-local-only.npy",np.asarray(recovered).reshape(-1,4))
    print(json.dumps(summary['overall'],indent=2),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--atd',type=Path,required=True)
    parser.add_argument('--homography',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--windows',type=int,default=24)
    parser.add_argument('--mask-frames',type=int,default=12)
    parser.add_argument('--backend',choices=('classic','yolo'),default='classic')
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--exclude-sampling',type=Path,help='Exclude complete time blocks exposed in an earlier pilot')
    args=parser.parse_args()
    if args.windows<2 or not 1<=args.mask_frames<=20: parser.error('windows >= 2 and mask-frames in [1,20] required')
    run(args)

"""Local-video recovery safety checks; OpenCV tests skip in minimal runtime."""
import importlib.util
import unittest
import numpy as np
from possession.video_recovery import transform, LocalBallTracker, YoloBallTracker, align_tracking_window


class HomographyAlignmentTests(unittest.TestCase):
    def test_roundtrip_and_singular_rejection(self):
        matrix=np.array([[600.,20,4],[5,300,9],[.01,.02,1]])
        points=np.array([[.2,.3],[.8,.7]])
        np.testing.assert_allclose(transform(matrix,transform(matrix,points),inverse=True),points)
        with self.assertRaises(ValueError): transform(np.zeros((3,3)),points)

    def test_unique_dynamic_alignment_and_replay_rejection(self):
        n=400
        reference=np.column_stack((np.linspace(.1,.9,n),.4+.1*np.sin(np.arange(n)/20)))
        other=np.full((n+170,2),np.nan)
        other[20:120]=reference[50:150]
        result=align_tracking_window(reference,other,50,150,[-30,0,20,80,120])
        self.assertTrue(result['accepted'])
        self.assertEqual(result['offset'],-30)
        other[170:270]=reference[50:150]
        self.assertFalse(align_tracking_window(reference,other,50,150,[-30,0,20,80,120])['accepted'])

    def test_static_or_invalid_calibration_has_no_alignment(self):
        constant=np.full((200,2),.5)
        self.assertFalse(align_tracking_window(constant,constant,0,150,range(-20,21))['accepted'])
        moving=np.column_stack((np.linspace(0,1,200),np.full(200,.5)))
        self.assertFalse(align_tracking_window(moving,moving,0,150,range(-20,21),other_valid=np.zeros(200,bool))['accepted'])


@unittest.skipUnless(importlib.util.find_spec('cv2'),'Optional OpenCV research dependency is not installed')
class VideoTrackerTests(unittest.TestCase):
    def image(self,x=100,second=None):
        import cv2
        image=np.full((240,400,3),(30,100,30),np.uint8)
        cv2.circle(image,(x,100),2,(240,240,240),-1)
        if second is not None: cv2.circle(image,(second,100),2,(240,240,240),-1)
        return image

    def tracker(self):
        tracker=LocalBallTracker()
        matrix=np.diag([400.,240.,1.])
        self.assertTrue(tracker.seed(0,self.image(),matrix,np.array([.25,100/240])))
        return tracker,matrix

    def test_visual_recovery_and_no_input_mutation(self):
        tracker,matrix=self.tracker()
        image=self.image(102);saved=image.copy()
        result=tracker.recover(1,image,matrix)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['point'],[102/400,100/240])
        np.testing.assert_array_equal(image,saved)

    def test_ambiguous_candidates_abstain(self):
        tracker,matrix=self.tracker()
        result=tracker.recover(1,self.image(98,104),matrix)
        self.assertFalse(result['accepted'])

    def test_expiration_and_invalid_homography_reset(self):
        tracker,matrix=self.tracker()
        self.assertEqual(tracker.recover(21,self.image(),matrix)['reason'],'anchor_expired')
        self.assertEqual(tracker.recover(22,self.image(),matrix)['reason'],'no_anchor')
        tracker,matrix=self.tracker()
        self.assertEqual(tracker.recover(1,self.image(),matrix,valid=False)['reason'],'invalid_homography')
        self.assertEqual(tracker.recover(2,self.image(),matrix)['reason'],'no_anchor')

    def test_cut_and_untextured_anchor_rejected(self):
        tracker,matrix=self.tracker()
        self.assertEqual(tracker.recover(1,np.full((240,400,3),255,np.uint8),matrix)['reason'],'shot_cut')
        self.assertFalse(tracker.seed(2,np.full((240,400,3),60,np.uint8),matrix,[.25,.5]))

    def test_optional_detector_scale_deduplication_and_gating(self):
        from types import SimpleNamespace
        class Array:
            def __init__(self,values):self.values=np.asarray(values)
            def cpu(self):return self
            def numpy(self):return self.values
        class Detector:
            names={32:'sports ball'}
            confidence=.8
            def predict(self,crops,**kwargs):
                assert kwargs['device']=='cpu' and kwargs['save'] is False
                results=[]
                for crop in crops:
                    yy,xx=np.where(crop[:,:,0]>200)
                    cx,cy=xx.mean(),yy.mean()
                    boxes=SimpleNamespace(xyxy=Array([[cx-2,cy-2,cx+2,cy+2]]),conf=Array([self.confidence]))
                    results.append(SimpleNamespace(boxes=boxes))
                return results
        detector=Detector();tracker=YoloBallTracker(detector);matrix=np.diag([400.,240.,1.])
        self.assertTrue(tracker.seed(0,self.image(),matrix,[.25,100/240]))
        result=tracker.recover(1,self.image(102),matrix)
        self.assertTrue(result['accepted'],result)
        np.testing.assert_allclose(result['point'],[102/400,100/240])
        detector.confidence=.2
        self.assertFalse(tracker.recover(2,self.image(103),matrix)['accepted'])

    def test_optional_detector_requires_sports_ball_class(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError):YoloBallTracker(SimpleNamespace(names={0:'person'}))


if __name__=='__main__': unittest.main()

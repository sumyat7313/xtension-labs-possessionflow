import copy
import unittest
import numpy as np
from possession.contact_evidence import FEATURE_NAMES,feature_matrix,extract_native
from possession.io import Tracking
from tools.contact_restart_v7 import peaks,align


def cached(n=100):
    x=np.zeros((n,98));z=np.zeros((n,58))
    x[:,0]=.8;x[:,11]=.9;x[:,56]=.8
    z[:,[0,1,3,5,8,9]]=1;z[:,4]=.01;z[:,7]=.1
    z[50:,2]=np.log1p(10)/np.log(46);z[50:,11]=4/13
    return x,z


def tracking(kind='contact'):
    n=120;pitch=np.array([105.,68.]);t=np.arange(n)/25
    if kind=='contact':position=np.where(t<2,30+4*t,38-10*(t-2))
    else:position=np.where(t<2,30,30+10*(t-2))
    ball=np.column_stack((position,np.full(n,30)))/pitch
    players=np.stack((ball+[.001,0],ball+[.2,0]),axis=1)
    return Tracking(players,np.zeros((n,2)),ball,np.zeros(n),np.array([0,1]),('H','A'))


class ContactEvidenceTests(unittest.TestCase):
    def test_cached_features_bounded_even_and_nonzero_for_supported_launch(self):
        x,z=cached();blocks=[(1,0,99)]
        a=feature_matrix(x,z,blocks,25)
        self.assertEqual(a.shape,(100,len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(a).all());self.assertTrue(((a>=0)&(a<=1)).all())
        np.testing.assert_array_equal(a,feature_matrix(-x,z,blocks,25))
        self.assertGreater(a[50,3],.5)
        self.assertGreater(a[50,1],.5)

    def test_missing_or_interpolated_context_abstains(self):
        x,z=cached()
        for column,value in ((0,0),(1,.45),(3,0)):
            altered=z.copy();altered[49,column]=value
            a=feature_matrix(x,altered,[(1,0,99)],25)
            np.testing.assert_array_equal(a[50],0)

    def test_uncertain_or_distant_team_never_produces_contact(self):
        x,z=cached()
        for xx,zz in ((x*.1,z),(x,np.where(np.arange(58)[None,:]==4,.5,z))):
            result=feature_matrix(xx,zz,[(1,0,99)],25)
            np.testing.assert_array_equal(result[:,1:],0)

    def test_cache_half_edges_and_inactive_gaps_isolated(self):
        x,z=cached();blocks=[(1,0,44),(2,55,99)]
        a=feature_matrix(x,z,blocks,25);changed=x.copy();changed[55:]*=-100
        np.testing.assert_array_equal(a[:55],feature_matrix(changed,z,blocks,25)[:55])
        np.testing.assert_array_equal(a[40:60],0)

    def test_native_contact_and_restart_are_candidates_not_verified_events(self):
        contact=extract_native(tracking(),[(1,0,119)],(105,68),25)
        restart=extract_native(tracking('restart'),[(1,0,119)],(105,68),25)
        self.assertGreater(contact['contact_strength'][50],.5)
        self.assertGreater(restart['restart_strength'][50],.5)
        for result in (contact,restart):
            self.assertFalse(result['verified_contact']);self.assertFalse(result['verified_restart'])
            self.assertFalse(result['independent_video_evidence'])

    def test_native_ignores_provider_speed_fields_and_preserves_team_symmetry(self):
        base=tracking();a=extract_native(base,[(1,0,119)],(105,68),25)
        altered=copy.deepcopy(base);altered.ball_speed[:]=np.nan;altered.player_speed[:]=10000;altered.teams=1-altered.teams
        b=extract_native(altered,[(1,0,119)],(105,68),25)
        np.testing.assert_array_equal(a['contact_strength'],b['contact_strength'])
        np.testing.assert_array_equal(a['signed_team_support'],-b['signed_team_support'])

    def test_native_missing_and_implausible_steps_block_contact_context(self):
        for value in (np.nan,1.1):
            altered=tracking();altered.ball[49]=value
            result=extract_native(altered,[(1,0,119)],(105,68),25)
            self.assertEqual(result['contact_strength'][50],0)
            self.assertEqual(result['restart_strength'][50],0)

    def test_native_half_isolation_and_permutation(self):
        base=tracking();blocks=[(1,0,55),(2,65,119)]
        a=extract_native(base,blocks,(105,68),25)
        altered=copy.deepcopy(base);altered.players=altered.players[:,::-1];altered.teams=altered.teams[::-1]
        altered.ball[65:]=np.nan
        b=extract_native(altered,blocks,(105,68),25)
        np.testing.assert_array_equal(a['contact_strength'][:65],b['contact_strength'][:65])
        np.testing.assert_array_equal(a['contact_strength'][56:65],0)

    def test_invalid_inputs_fail_closed(self):
        x,z=cached()
        for blocks in ([(1,0,99),(2,2,4)],[(1,-1,90)]):
            with self.assertRaises(ValueError):feature_matrix(x,z,blocks,25)
        for bad in (z[:,:57],np.full_like(z,np.nan)):
            with self.assertRaises(ValueError):feature_matrix(x,bad,[(1,0,99)],25)
        with self.assertRaises(ValueError):feature_matrix(x,z,[(1,0,99)],0)
        bad=tracking();bad.teams[:]=0
        with self.assertRaises(ValueError):extract_native(bad,[(1,0,119)],(105,68),25)

    def test_peak_selection_is_strength_ranked_and_half_local(self):
        values=np.zeros(60);values[[10,11,20,35,36]]=[.6,.9,.8,.7,.8]
        self.assertEqual(peaks(values,[(1,0,34),(2,35,59)],25),[11,36])
        values[:]=0;values[[34,35]]=.8
        self.assertEqual(peaks(values,[(1,0,34),(2,35,59)],25),[34,35])

    def test_alignment_cannot_bridge_unknown_reference_spans(self):
        reference=np.array(list('DDHHXXDDHH'))
        result=align(reference,[2,5,8],[(1,0,9)],25,'restart')
        self.assertEqual(result['n_reference_boundaries'],2)
        self.assertEqual(result['n_candidates'],2)
        self.assertEqual(result['n_matched'],2)
        self.assertIsNone(result['physical_contact_precision'])
        self.assertFalse(result['reference_is_independent_contact_truth'])


if __name__=='__main__':unittest.main()

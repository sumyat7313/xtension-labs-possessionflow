import unittest
import numpy as np
from tools.temporal_diagnostics_v6 import diagnostic


class TemporalDiagnosticsTests(unittest.TestCase):
    def test_perfect_dense_and_short_spells_at_native_rates(self):
        for fps in (10,25):
            y=np.array(list('H'*fps+'D'*(3*fps)+'A'*fps))
            d=diagnostic(y,y,fps)
            for key in ('dead_segments','dead_onsets','dead_offsets'):
                self.assertEqual(d[key]['f1'],1)
            self.assertEqual(d['short_possessions']['recall'],1)
            self.assertEqual(d['dead_duration_recall']['2-5s']['n_gt'],1)

    def test_half_edges_and_unknown_runs_never_create_or_match_boundaries(self):
        y=np.array(list('HHDDXXDDHH'))
        d=diagnostic(y,y,10,[(1,0,1),(2,2,9)])
        self.assertEqual(d['dead_onsets']['n_gt'],0)
        self.assertEqual(d['dead_offsets']['n_gt'],1)
        self.assertEqual(d['dead_segments']['n_gt'],2)

    def test_spurious_stoppages_reduce_precision(self):
        y=np.array(list('H'*20+'D'*20+'A'*20))
        p=y.copy();p[5:10]='D'
        d=diagnostic(y,p,10)
        self.assertEqual(d['dead_segments']['precision'],.5)
        self.assertEqual(d['dead_onsets']['n_pred'],2)

    def test_unknown_prediction_is_not_filled_for_stoppages(self):
        y=np.array(list('H'*10+'D'*20+'A'*10));p=y.copy();p[10:30]='X'
        d=diagnostic(y,p,10)
        self.assertEqual(d['dead_segments']['recall'],0)
        self.assertEqual(d['dead_onsets']['n_pred'],0)

    def test_timing_uses_seconds_and_short_bins_are_half_open(self):
        y=np.array(list('H'*20+'D'*20+'A'*20));p=np.array(list('H'*25+'D'*20+'A'*15))
        d=diagnostic(y,p,10)
        self.assertEqual(d['dead_onsets']['mean_signed_timing_error_seconds'],.5)
        self.assertIsNone(d['short_possessions']['recall'])
        y=np.array(list('H'*3+'A'*4+'H'*10+'D'*20))
        d=diagnostic(y,y,10)
        self.assertEqual([d['short_duration_recall'][k]['n_gt'] for k in ('<0.4s','0.4-1s','1-2s')],[1,1,1])

    def test_invalid_shapes_blocks_and_fps_fail(self):
        y=np.array(list('HAD'))
        for blocks in ([(1,0,2),(2,1,2)],[(1,-1,2)],[(1,0,3)]):
            with self.assertRaises(ValueError):diagnostic(y,y,10,blocks)
        with self.assertRaises(ValueError):diagnostic(y,y[:-1])
        with self.assertRaises(ValueError):diagnostic(y,y,0)


if __name__=='__main__':unittest.main()

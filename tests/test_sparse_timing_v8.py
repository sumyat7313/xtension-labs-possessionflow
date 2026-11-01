import copy,unittest
import numpy as np
from tools.sparse_timing_v8 import diagnostic
from tools.data_intake_v8 import ELASTIC_COMMIT

def document():
    return {'fps':25,'commit':ELASTIC_COMMIT,'fresh_holdout':False,'dense_had_truth':False,
      'frame_ranges':{'1':[10000,10099,0],'2':[100000,100099,101]},'points':[
        {'kind':'ball_out','period':1,'index':30,'frame_id':10030,'actor':'out_bottom','team':None,'event_type':'pass'},
        {'kind':'restart','period':1,'index':60,'frame_id':10060,'actor':'home_1','team':'H','event_type':'throw_in'}]}

class SparseTimingTests(unittest.TestCase):
    def test_sparse_events_report_recall_without_false_precision(self):
        labels=np.array(list('H'*30+'D'*30+'H'*40+'X'+'A'*100));d=diagnostic(labels,[(1,0,99),(2,101,200)],25,document())
        for kind in ('restart','ball_out'):
            self.assertEqual(d[kind]['n_matched'],1);self.assertEqual(d[kind]['recall_within_1s'],1.)
            self.assertIsNone(d[kind]['precision']);self.assertIsNone(d[kind]['f1'])
        self.assertFalse(d['fresh_match']);self.assertFalse(d['unlisted_frames_are_negatives'])

    def test_team_half_and_unknown_boundaries_cannot_match(self):
        labels=np.array(list('H'*30+'D'*30+'A'*40+'X'+'H'*100));d=diagnostic(labels,[(1,0,99),(2,101,200)],25,document())
        self.assertEqual(d['restart']['n_matched'],0)
        labels[29:31]='X';d=diagnostic(labels,[(1,0,99),(2,101,200)],25,document());self.assertEqual(d['ball_out']['n_matched'],0)

    def test_deduplicates_same_point_and_empty_scope_is_unknown(self):
        doc=document();doc['points']*=2;labels=np.array(list('H'*30+'D'*30+'H'*40))
        d=diagnostic(labels,[(1,0,99)],25,doc);self.assertEqual(d['restart']['n_reference_points'],1)
        doc['points']=[];d=diagnostic(labels,[(1,0,99)],25,doc);self.assertIsNone(d['restart']['recall_within_1s'])

    def test_provenance_alignment_semantics_fail_closed(self):
        labels=np.array(list('H'*100))
        for bad in ('commit','index','semantics','fresh'):
            doc=copy.deepcopy(document())
            if bad=='commit':doc['commit']='bad'
            elif bad=='index':doc['points'][0]['index']=29
            elif bad=='semantics':doc['points'][1]['event_type']='pass'
            else:doc['fresh_holdout']=True
            with self.assertRaises(ValueError):diagnostic(labels,[(1,0,99)],25,doc)

if __name__=='__main__':unittest.main()

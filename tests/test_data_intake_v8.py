import copy,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.data_intake_v8 import check_blob,frame_index,normalize_events,reviewed_intervals,validated_points,SOURCE_SHA256

RANGES={'1':[10000,10999,0],'2':[100000,100999,1001]}

def event(**changes):
    row={'period_id':1,'player_id':'home_7','spadl_type':'pass','frame_id':10020,
         'receiver_id':'away_2','receive_frame_id':10060,'synced_ts':'00:00.80','receive_ts':'00:02.40'}
    return dict(row,**changes)

def reviewed():
    return {'schema_version':1,'annotation_origin':'human','reviewer_ids':['rater-a','rater-b'],
            'reviewers_blinded_to_predictions':True,'disagreements_adjudicated':True,
            'source_video_sha256':'a'*64,'match':'unseen-fixture-01','n_frames':21,'fps':25,
            'blocks':[[1,0,9],[2,11,20]],'intervals':[{'period':1,'start':2,'end':7,'label':'H','boundary_uncertainty_frames':1}]}

class DataIntakeTests(unittest.TestCase):
    def test_pinned_blob_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample';p.write_bytes(b'abc')
            blob=hashlib.sha1(b'blob 3\0abc').hexdigest();check_blob(p,(3,blob))
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):check_blob(p,(3,blob))

    def test_exact_half_local_frame_mapping(self):
        self.assertEqual(frame_index(100000,2,RANGES),1001)
        for frame,period in [(9999,1),(11000,1),(10999,2),(10020,3),(True,1),(10000.5,1)]:
            with self.assertRaises(ValueError):frame_index(frame,period,RANGES)

    def test_sparse_receipts_never_fill_unlisted_frames(self):
        result=normalize_events([event()],'J03WMX',RANGES)
        self.assertEqual([p['index'] for p in result['points']],[20,60])
        self.assertEqual([p['team'] for p in result['points']],['H','A'])
        self.assertFalse(result['dense_had_truth']);self.assertFalse(result['fresh_holdout'])
        self.assertNotIn('labels',result)

    def test_restart_out_goal_are_distinct_and_foul_not_whistle(self):
        rows=[event(spadl_type='throw_in',receiver_id='out_bottom'),event(frame_id=10070,spadl_type='foul',receiver_id=None,receive_frame_id=None),
              event(frame_id=10080,spadl_type='shot',receiver_id='goal_left',receive_frame_id=10100)]
        points=normalize_events(rows,'J03WN1',RANGES)['points']
        self.assertEqual([p['kind'] for p in points],['restart','ball_out','event','event','goal'])
        self.assertIsNone(points[1]['team'])

    def test_negative_receipt_and_cross_half_rejected_not_shifted(self):
        for changes in ({'receive_frame_id':10019},{'receive_frame_id':100000},{'receiver_id':None},
                        {'receiver_id':'maybe_control'},{'player_id':'home'},{'spadl_type':'unknown'}):
            with self.assertRaises(ValueError):normalize_events([event(**changes)],'J03WPY',RANGES)
        with self.assertRaises(ValueError):normalize_events([event(),event(frame_id=10019)],'J03WPY',RANGES)
        with self.assertRaises(ValueError):normalize_events([event()],'new-match',RANGES)

    def test_valid_missing_receiver_and_inputs_unchanged(self):
        rows=[event(receiver_id=float('nan'),receive_frame_id=float('nan'))]
        result=normalize_events(rows,'J03WPY',RANGES)
        self.assertEqual(len(result['points']),1)
        self.assertTrue(np.isnan(rows[0]['receive_frame_id']))

    def test_review_unknown_gaps_uncertainty_and_half_separators(self):
        source=reviewed();old=copy.deepcopy(source);labels,proof=reviewed_intervals(source)
        self.assertEqual(''.join(labels),'XXXHHHH'+'X'*14)
        self.assertEqual(source,old);self.assertFalse(proof['promotion_eligible'])
        self.assertFalse(proof['human_review_independently_verified'])

    def test_pseudo_labels_and_unblinded_review_rejected(self):
        for changes in ({'annotation_origin':'machine'},{'reviewer_ids':['a','a']},{'reviewers_blinded_to_predictions':False},
                        {'disagreements_adjudicated':False},{'source_video_sha256':'not-a-hash'}):
            with self.assertRaises(ValueError):reviewed_intervals(dict(reviewed(),**changes))

    def test_reviewer_ids_requires_list_of_individual_ids(self):
        for value in ('abc',{'a':1,'b':2},None,23,[['a'],['b']]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                reviewed_intervals(dict(reviewed(),reviewer_ids=value))

    def test_review_overlap_invalid_values_and_cross_half_rejected(self):
        for changes in ({'end':12},{'boundary_uncertainty_frames':-1},{'label':'?'},{'period':3},{'start':True}):
            d=reviewed();d['intervals'][0].update(changes)
            with self.assertRaises(ValueError):reviewed_intervals(d)
        d=reviewed();d['intervals']*=2
        with self.assertRaises(ValueError):reviewed_intervals(d)
        d=reviewed();d['blocks']=[[1,0,9],[2,8,20]]
        with self.assertRaises(ValueError):reviewed_intervals(d)

    def test_reannotated_exposed_fixture_does_not_become_fresh(self):
        d=reviewed();d['match']='J03WMX';_,proof=reviewed_intervals(d)
        self.assertEqual(proof['role'],'exposed-development-only')
        d['match']='new-match';_,proof=reviewed_intervals(d)
        self.assertEqual(proof['role'],'candidate-holdout-pending-provenance-review')
        self.assertFalse(proof['promotion_eligible'])

    def test_full_sparse_wrapper_provenance_and_index_validation(self):
        d=normalize_events([event()],'J03WMX',RANGES)
        d.update(source_sha256=SOURCE_SHA256['J03WMX'],frame_ranges=RANGES,tracking_cache_metadata_sha256='a'*64,
                 tracking_input_provenance={'J03WMX_tracking.xml':{'bytes':10,'sha256':'b'*64}},quarantined_rows=[])
        for point in d['points']:point['source_event_id']=point['event_id']
        kwargs={'match':'J03WMX','fps':25,'blocks':[(1,0,999),(2,1001,2000)],'n_frames':2001}
        self.assertEqual(validated_points(d,**kwargs),d['points'])
        for key,value in [('commit','0'*40),('fresh_holdout',True),('dense_had_truth',True),('source_sha256','c'*64),('fps',10),('match','J03WPY')]:
            changed=copy.deepcopy(d);changed[key]=value
            with self.assertRaises(ValueError):validated_points(changed,**kwargs)
        for key,value in [('index',1000),('period',2),('kind','ball_out'),('team','A'),('source_event_id',-1)]:
            changed=copy.deepcopy(d);changed['points'][0][key]=value
            with self.assertRaises(ValueError):validated_points(changed,**kwargs)
        changed=copy.deepcopy(d);changed['quarantined_rows']=[{'source_event_id':0}]
        with self.assertRaises(ValueError):validated_points(changed,**kwargs)

if __name__=='__main__':unittest.main()

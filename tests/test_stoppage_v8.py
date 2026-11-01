import copy
import unittest
import numpy as np
from research import stoppage_v8 as sv
from tools.train_stoppage_v8 import fit,training_groups,_matches


def fixture():
    labels=np.array(list('H'*100+'D'*100+'A'*100));p=np.tile([.45,.45,.1],(300,1));p[100:200]=[.05,.05,.9]
    x=np.zeros((300,98));z=np.zeros((300,58));z[:,0]=1;z[:,9]=1
    return labels,p,x,z,[(1,0,299)],25


def invented_model():
    n=len(sv.FEATURE_NAMES);head={'mean':[0.]*n,'scale':[1.]*n,'coefficients':[0.]*n}
    return {'kind':'stoppage_boundary_ranker_v8','schema_version':1,'policy':dict(sv.POLICY),
            'feature_names':list(sv.FEATURE_NAMES),'heads':{'stop':copy.deepcopy(head),'restart':copy.deepcopy(head)}}


def human_item():
    from tools.data_intake_v8 import ELASTIC_COMMIT,SOURCE_SHA256
    labels,p,x,z,blocks,fps=fixture()
    document={'schema_version':1,'commit':ELASTIC_COMMIT,'match':'J03WMX','fps':25,'role':'exposed-development-only',
       'fresh_holdout':False,'dense_had_truth':False,'annotation_method':'provider-reported median of three human timestamp annotations',
       'individual_rater_uncertainty_available':False,'source_sha256':SOURCE_SHA256['J03WMX'],
       'tracking_cache_metadata_sha256':'a'*64,'tracking_input_provenance':{'J03WMX_synthetic.xml':{'sha256':'b'*64,'bytes':1}},
       'frame_ranges':{'1':[1000,1299,0]},'points':[
        {'index':110,'frame_id':1110,'period':1,'kind':'ball_out','actor':'out_top','event_type':'pass','team':None,'source_event_id':1,'event_id':1},
        {'index':190,'frame_id':1190,'period':1,'kind':'restart','actor':'away_1','event_type':'throw_in','team':'A','source_event_id':2,'event_id':2}]}
    return dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,y=labels.copy(),match='J03WMX',sparse_points=document)


class StoppageV8Tests(unittest.TestCase):
    def test_neutral_is_exact_identity_and_observations_not_mutated(self):
        args=fixture();original=[v.copy() if isinstance(v,np.ndarray) else copy.deepcopy(v) for v in args]
        np.testing.assert_array_equal(sv.refine(invented_model(),*args),args[0])
        for a,b in zip(args[:4],original[:4]):np.testing.assert_array_equal(a,b)

    def test_strong_shifts_preserve_topology_and_minimum_durations(self):
        labels,p,x,z,blocks,fps=fixture();z[[120,180],6]=1;model=invented_model()
        for head in model['heads'].values():head['coefficients'][sv.FEATURE_NAMES.index('outside_support')]=10
        result=sv.refine(model,labels,p,x,z,blocks,fps)
        self.assertEqual(np.flatnonzero(result=='D')[[0,-1]].tolist(),[120,179])
        spans=sv.segments(result,[(0,300)])[0]
        self.assertEqual([v[2] for v in spans],['H','D','A'])
        self.assertTrue(all(b-a>=round((.6 if c=='D' else .4)*fps) for a,b,c in spans))
        np.testing.assert_array_equal(labels,np.array(list('H'*100+'D'*100+'A'*100)))

    def test_half_gap_and_team_exchange_isolation(self):
        labels,p,x,z,_,fps=fixture();labels=np.r_[labels,np.array(list('X'*20)),labels]
        p=np.vstack((p,np.tile([.4,.4,.2],(20,1)),p));x=np.vstack((x,np.zeros((20,98)),x));z=np.vstack((z,np.zeros((20,58)),z))
        blocks=[(1,0,299),(2,320,619)];model=invented_model()
        for head in model['heads'].values():head['coefficients'][sv.FEATURE_NAMES.index('outside_support')]=10
        z[[120,180,440,500],6]=1;original=sv.refine(model,labels,p,x,z,blocks,fps)
        changed=z.copy();changed[320:]*=3
        np.testing.assert_array_equal(sv.refine(model,labels,p,x,changed,blocks,fps)[:320],original[:320])
        swapped=np.where(labels=='H','A',np.where(labels=='A','H',labels))
        actual=sv.refine(model,swapped,p[:,[1,0,2]],-x,z,blocks,fps)
        np.testing.assert_array_equal(actual,np.where(original=='H','A',np.where(original=='A','H',original)))
        np.testing.assert_array_equal(actual[300:320],'X')

    def test_missing_ball_abstains_and_cannot_invent_dead_segment(self):
        labels,p,x,z,blocks,fps=fixture();model=invented_model()
        for head in model['heads'].values():head['coefficients'][sv.FEATURE_NAMES.index('outside_support')]=100
        z[[120,180],6]=1;z[:,0]=0
        np.testing.assert_array_equal(sv.refine(model,labels,p,x,z,blocks,fps),labels)
        labels[:]='H';z[:,0]=1
        np.testing.assert_array_equal(sv.refine(model,labels,p,x,z,blocks,fps),labels)

    def test_unknown_prediction_break_is_context_barrier(self):
        labels,p,x,z,blocks,fps=fixture();labels[90:96]='X'
        proposals=sv.proposals(labels,p,x,z,blocks,fps)
        self.assertFalse(any(q['kind']=='stop' for q in proposals))
        self.assertTrue(any(q['kind']=='restart' for q in proposals))

    def test_unknown_reference_excluded_and_matching_one_to_one(self):
        labels,p,x,z,blocks,fps=fixture();y=labels.copy();y[110:115]='X'
        groups,counts=training_groups([dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,y=y)])
        self.assertGreater(counts['unknown_context_excluded'],0)
        pred=np.array(list('H'*20+'D'*40+'H'*10+'D'*40+'H'*20));truth=np.array(list('H'*20+'D'*90+'H'*20))
        pairs=_matches(pred,truth,[(0,len(pred))])
        self.assertEqual(len(set(pairs.values())),len(pairs))

    def test_fit_is_deterministic_and_learns_boundary_ranking(self):
        labels,p,x,z,blocks,fps=fixture();y=np.array(list('H'*110+'D'*80+'A'*110))
        z[[110,190],6]=1;item=dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,y=y)
        model=fit([item]);self.assertEqual(model,fit([item]))
        result=sv.refine(model,labels,p,x,z,blocks,fps)
        self.assertLess(np.sum(result!=y),np.sum(labels!=y))
        self.assertEqual(model['training']['matched_targets'],2)

    def test_reference_team_mismatch_retains_original_target(self):
        labels,p,x,z,blocks,fps=fixture();y=np.array(list('A'*110+'D'*80+'H'*110))
        groups,counts=training_groups([dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,y=y)])
        self.assertEqual(counts['matched_targets'],0);self.assertEqual(counts['retention_targets'],2)
        for kind,records in groups.items():
            self.assertEqual(len(records),1);self.assertEqual(int((records[0][1]>0).sum()),1)

    def test_invalid_shapes_values_policy_and_fps_fail_closed(self):
        args=fixture();model=invented_model()
        for value in (0,True,float('nan')):
            with self.assertRaises(ValueError):sv.refine(model,*args[:-1],value)
        broken=list(args);broken[3]=args[3][:,:57]
        with self.assertRaises(ValueError):sv.refine(model,*broken)
        broken=list(args);broken[1]=args[1]*2
        with self.assertRaises(ValueError):sv.refine(model,*broken)
        bad=copy.deepcopy(model);bad['policy']['maximum_shift_seconds']=10
        with self.assertRaises(ValueError):sv.validate_model(bad)
        bad=copy.deepcopy(model);bad['heads']['stop']['scale'][0]=0
        with self.assertRaises(ValueError):sv.validate_model(bad)
        with self.assertRaises(ValueError):fit([],l2=.1)

    def test_sparse_human_points_override_only_aligned_boundary_targets(self):
        item=human_item();groups,counts=training_groups([item])
        self.assertEqual(counts['human_stop_targets'],1);self.assertEqual(counts['human_restart_targets'],1)
        proposal=sv.proposals(*[item[k] for k in ('labels','p','x','z','blocks','fps')])
        for q in proposal:
            target=110 if q['kind']=='stop' else 190
            selected=q['indices'][groups[q['kind']][0][1]>0]
            self.assertTrue((np.abs(selected-target)<=3).all())
        item['sparse_points']['points']=[];_,counts=training_groups([item])
        self.assertEqual(counts['human_absent_dense_fallback'],2)

    def test_sparse_ambiguity_and_unknown_context_abstain(self):
        item=human_item();other=copy.deepcopy(item['sparse_points']['points'][1]);other.update(index=192,frame_id=1192,source_event_id=3)
        item['sparse_points']['points'].append(other);groups,counts=training_groups([item])
        self.assertEqual(len(groups['restart']),0);self.assertEqual(counts['human_ambiguous_groups_excluded'],1)
        item=human_item();item['y'][109]='X';groups,counts=training_groups([item])
        self.assertEqual(len(groups['stop']),0);self.assertGreater(counts['unknown_context_excluded'],0)

    def test_sparse_provenance_alignment_semantics_and_fold_fail_closed(self):
        for change in ('commit','frame','role','foul','match','duplicate'):
            item=human_item();doc=item['sparse_points']
            if change=='commit':doc['commit']='bad'
            elif change=='frame':doc['points'][0]['frame_id']+=1
            elif change=='role':doc['fresh_holdout']=True
            elif change=='foul':doc['points'][1]['event_type']='foul'
            elif change=='match':doc['match']='J03WN1'
            else:doc['points'].append(copy.deepcopy(doc['points'][0]))
            with self.assertRaises(ValueError):training_groups([item])
        with self.assertRaises(ValueError):fit([human_item()],metadata={'held_out_matches':['J03WMX']})


if __name__=='__main__':unittest.main()

import copy
import unittest
import numpy as np
from research import precision_v8 as pr
from tools.train_precision_v8 import examples,fit


def invented_model(score=.95):
    n=len(pr.FEATURE_NAMES)
    return {'schema_version':1,'kind':'short_island_quality_v8','feature_names':list(pr.FEATURE_NAMES),
        'policy':dict(pr.POLICY),'mean':[0.]*n,'scale':[1.]*n,'coefficients':[0.]*n,
        'intercept':float(np.log(score/(1-score))),'metadata':{}}


def observations(labels='H'*30+'A'*20+'H'*30):
    labels=np.array(list(labels));n=len(labels);p=np.tile([.7,.2,.1],(n,1));p[labels=='A']=[.2,.7,.1]
    x=np.zeros((n,98));z=np.zeros((n,58));z[:,[0,1,9]]=1
    return labels,p,x,z,[(1,0,n-1)],25


class PrecisionTests(unittest.TestCase):
    def test_conservative_merge_and_no_duration_only_deletion(self):
        args=observations();before=[v.copy() for v in args[:4]]
        actual,diagnostic=pr.refine(invented_model(),*args,return_diagnostics=True)
        self.assertTrue((actual=='H').all());self.assertEqual(diagnostic['merged_islands'],1)
        np.testing.assert_array_equal(pr.refine(invented_model(.7),*args),args[0])
        for value,old in zip(args[:4],before):np.testing.assert_array_equal(value,old)
        # Strong observed evidence in favor of the island is an explicit veto.
        args[2][30:50,0]=-.2
        np.testing.assert_array_equal(pr.refine(invented_model(),*args),args[0])

    def test_observation_support_in_each_window_and_complete_context(self):
        for a,b in ((20,30),(30,50),(50,60)):
            args=observations();args[3][a:b,0]=0
            np.testing.assert_array_equal(pr.refine(invented_model(),*args),args[0])
        args=observations('H'*9+'A'*10+'H'*20)
        self.assertEqual(pr.segments(*args)[0],[])

    def test_team_exchange_is_equivariant(self):
        args=observations();args[2][:,0]=np.linspace(-.05,.05,len(args[0]));args[2][:,11]=.02
        swapped=list(args);swapped[0]=np.where(args[0]=='H','A','H');swapped[1]=args[1][:,[1,0,2]];swapped[2]=-args[2]
        np.testing.assert_array_equal(pr.segments(*args)[1],pr.segments(*swapped)[1])
        original=pr.refine(invented_model(),*args);other=pr.refine(invented_model(),*swapped)
        np.testing.assert_array_equal(other,np.where(original=='H','A','H'))

    def test_dead_unknown_and_half_boundaries_preserved(self):
        args=observations('H'*30+'A'*20+'D'*30+'H'*30+'A'*20+'X'*30)
        np.testing.assert_array_equal(pr.refine(invented_model(),*args),args[0])
        args=observations();args=list(args);args[4]=[(1,0,39),(2,40,79)]
        np.testing.assert_array_equal(pr.refine(invented_model(),*args),args[0])

    def test_later_half_cannot_change_first_half(self):
        args=list(observations(('H'*30+'A'*20+'H'*30)*2));args[4]=[(1,0,79),(2,80,159)]
        before=pr.refine(invented_model(),*args);args[2][80:]=100;args[3][80:]=0;args[1][80:]=[.1,.8,.1]
        after=pr.refine(invented_model(),*args);np.testing.assert_array_equal(before[:80],after[:80])

    def test_two_second_island_and_endpoint_spells_abstain(self):
        for text in ('H'*30+'A'*50+'H'*30,'A'*20+'H'*40,'H'*40+'A'*20):
            args=observations(text);np.testing.assert_array_equal(pr.refine(invented_model(),*args),args[0])

    def test_conflicting_islands_do_not_both_consume_the_same_context(self):
        args=observations('H'*30+'A'*12+'H'*12+'A'*12+'H'*30)
        actual,diag=pr.refine(invented_model(),*args,return_diagnostics=True)
        self.assertEqual(diag['eligible_islands'],3);self.assertEqual(diag['merged_islands'],1)
        self.assertTrue((actual[30:42]=='H').all());self.assertTrue((actual[54:66]=='A').all())

    def test_training_targets_unknown_and_ambiguous_are_excluded(self):
        labels,p,x,z,blocks,fps=observations();base=dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps)
        safe=dict(base,y=np.array(list('H'*80)));keep=dict(base,y=labels.copy())
        unknown=copy.deepcopy(safe);unknown['y'][35]='X'
        ambiguous=copy.deepcopy(safe);ambiguous['y'][30:35]='A'
        matrix,y,count=examples([safe,keep,unknown,ambiguous])
        np.testing.assert_array_equal(y,[1,0]);self.assertEqual(count['ambiguous_or_unknown'],2)
        self.assertEqual(matrix.shape,(2,len(pr.FEATURE_NAMES)))

    def test_deterministic_fit_and_unknown_reference_does_not_change_model(self):
        items=[]
        for i in range(20):
            labels,p,x,z,blocks,fps=observations();safe=i%2==0;x[30:50,0]=.5 if safe else -.5
            items.append(dict(labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,y=np.array(list('H'*80)) if safe else labels.copy()))
        model=fit(items);self.assertEqual(model,fit(items));matrix,y,_=examples(items)
        self.assertTrue(((pr.predict(model,matrix)>.8)==y).all())
        bad=copy.deepcopy(items[0]);bad['y'][:]='X';bad['x'][:]=100
        altered=fit(items+[bad])
        for key in ('mean','scale','coefficients','intercept'):self.assertEqual(model[key],altered[key])
        for unsupported in ([items[0]],[]):
            noop=fit(unsupported);self.assertFalse(noop['training']['supported'])
            self.assertTrue((pr.predict(noop,matrix)<pr.POLICY['merge_score_threshold']).all())

    def test_invalid_inputs_and_empty_arrays(self):
        args=list(observations());bad=copy.deepcopy(invented_model());bad['policy']['merge_score_threshold']=.1
        with self.assertRaises(ValueError):pr.refine(bad,*args)
        args[1][0]=[2,0,0]
        with self.assertRaises(ValueError):pr.refine(invented_model(),*args)
        args=list(observations());args[-1]=True
        with self.assertRaises(ValueError):pr.refine(invented_model(),*args)
        out=pr.refine(invented_model(),np.array([],dtype='<U1'),np.zeros((0,3)),np.zeros((0,98)),np.zeros((0,58)),[],25)
        self.assertEqual(len(out),0)

    def test_sparse_human_reception_conflict_excludes_only_safe_merge_target(self):
        from tools.data_intake_v8 import normalize_events,SOURCE_SHA256
        ranges={'1':[10000,10079,0]}
        event={'period_id':1,'player_id':'home_7','spadl_type':'pass','frame_id':10020,
            'receiver_id':'away_2','receive_frame_id':10040,'synced_ts':'00:00.80','receive_ts':'00:01.60'}
        doc=normalize_events([event],'J03WMX',ranges)
        doc.update(source_sha256=SOURCE_SHA256['J03WMX'],frame_ranges=ranges,tracking_cache_metadata_sha256='a'*64,
            tracking_input_provenance={'J03WMX_tracking.xml':{'bytes':10,'sha256':'b'*64}},quarantined_rows=[])
        for point in doc['points']:point['source_event_id']=point['event_id']
        labels,p,x,z,blocks,fps=observations()
        item=dict(match='J03WMX',labels=labels,p=p,x=x,z=z,blocks=blocks,fps=fps,
            y=np.array(list('H'*80)),sparse_points=doc)
        _,target,count=examples([item]);self.assertEqual(len(target),0);self.assertEqual(count['sparse_conflict_excluded'],1)
        keep=dict(item,y=labels.copy());self.assertEqual(examples([keep])[1].tolist(),[0])
        other=copy.deepcopy(item);other['sparse_points']['points'][1].update(actor='home_2',team='H')
        self.assertEqual(examples([other])[1].tolist(),[1])
        for bad in (doc['points'],dict(doc,source_sha256='c'*64),dict(doc,dense_had_truth=True)):
            with self.assertRaises(ValueError):examples([dict(item,sparse_points=bad)])


if __name__=='__main__':unittest.main()

import copy
import unittest
from unittest.mock import patch
import numpy as np
from tools.train_refinement_v8 import BASELINE,CANDIDATES,FOLDS,REGIMES,select,apply_models

def records():
    rows=[]
    for name in (BASELINE,*CANDIDATES):
        gain=.02 if name!=BASELINE else 0.
        for fold in FOLDS:
            for match in fold:
                for regime in REGIMES:
                    rows.append({'candidate':name,'match':match,'regime':regime,'spellF1':.6+gain,'macroF1':.9,
                      'dead_precision':.95,'dead_recall':.96,'temporal_diagnostics':{'dead_segments':{'f1':.9},
                      'dead_onsets':{'f1':.8+gain},'dead_offsets':{'f1':.8+gain},'short_possessions':{'recall':.28}},
                      'duration_precision_recall':{'bins':{k:{'n_pred':10,'n_matched_pred':5} for k in ('<0.4s','0.4-1s','1-2s')}}})
    return rows

class RefinementCampaignTests(unittest.TestCase):
    def test_complete_fixed_cells_deterministic_selection(self):
        rows=records();s=select(rows)
        self.assertEqual(s['selected'],'combined');self.assertEqual(s,select(rows[::-1]))
        self.assertTrue(s['gates']['combined']['passed']);self.assertFalse(s['fresh_audit_used_for_selection'])
        np.testing.assert_allclose(s['gates']['combined']['match_bootstrap_5_95'],[.02,.02])

    def test_precision_recall_or_stoppage_loss_blocks(self):
        for kind in ('precision','recall','dead_segment','dead_boundary','dead_precision'):
            rows=records()
            for r in rows:
                if r['candidate']!='precision':continue
                if kind=='precision':r['duration_precision_recall']['bins']['<0.4s']['n_matched_pred']=4
                elif kind=='recall':r['temporal_diagnostics']['short_possessions']['recall']=.279
                elif kind=='dead_segment':r['temporal_diagnostics']['dead_segments']['f1']=.899
                elif kind=='dead_boundary':r['temporal_diagnostics']['dead_onsets']['f1']=.7
                else:r['dead_precision']=.949
            self.assertFalse(select(rows)['gates']['precision']['passed'],kind)

    def test_match_bootstrap_does_not_treat_corruptions_as_independent(self):
        rows=records();first=FOLDS[0][0]
        for r in rows:
            if r['candidate']=='precision':r['spellF1']=.65 if r['match']==first else .597
        gate=select(rows)['gates']['precision']
        self.assertGreater(gate['mean_gains']['spell'],.005)
        self.assertLess(gate['match_bootstrap_5_95'][0],0);self.assertFalse(gate['passed'])

    def test_missing_duplicate_unknown_nonfinite_fail(self):
        rows=records()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):select(bad)
        for key,value in [('candidate','bogus'),('spellF1',np.nan),('spellF1',True)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):select(bad)

    def test_no_gain_retains_baseline(self):
        rows=records()
        for r in rows:r['spellF1']=.6
        s=select(rows);self.assertEqual(s['selected'],BASELINE);self.assertIn(s['audit_candidate'],CANDIDATES)

    def test_composition_order_and_annotations_never_pass_to_refiners(self):
        labels=np.array(list('HA'));item=dict(labels=labels,p=object(),x=object(),z=object(),blocks=[],fps=25,
          y=np.array(list('DD')),sparse_points={'secret_truth':True})
        seen=[]
        def precision(model,current,*args):
            seen.append(('precision',model,args));return current[::-1].copy()
        def stoppage(model,current,*args):
            seen.append(('stoppage',model,args));np.testing.assert_array_equal(current,['A','H']);return current
        with patch('research.precision_v8.refine',side_effect=precision),patch('research.stoppage_v8.refine',side_effect=stoppage):
            output=apply_models({'precision':'p','stoppage':'d'},item,'combined')
        self.assertEqual([s[0] for s in seen],['precision','stoppage'])
        self.assertTrue(all(len(s[2])==5 for s in seen));np.testing.assert_array_equal(labels,['H','A'])
        np.testing.assert_array_equal(output,['A','H'])

if __name__=='__main__':unittest.main()

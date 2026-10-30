import copy
from dataclasses import replace
from itertools import product
import unittest
import numpy as np
from possession.config import Config
from possession.decoder import decode
from tools.train_boundary_campaign_v7 import BASELINE,CANDIDATES,FOLDS,REGIMES,select


def records():
    rows=[]
    for name in (BASELINE,*CANDIDATES):
        gain=.02 if name!=BASELINE else 0
        for fold in FOLDS:
            for match in fold:
                for regime in REGIMES:
                    rows.append({'candidate':name,'match':match,'regime':regime,'spellF1':.6+gain,'macroF1':.9,
                        'dead_precision':.95,'dead_recall':.96,'temporal_diagnostics':{'dead_segments':{'f1':.9},
                        'dead_onsets':{'f1':.8+gain},'dead_offsets':{'f1':.8+gain},'short_possessions':{'recall':.28+gain}},
                        'duration_precision_recall':{'bins':{k:{'n_pred':10,'n_matched_pred':5} for k in ('<0.4s','0.4-1s','1-2s')}}})
    return rows


class BoundaryCampaignTests(unittest.TestCase):
    def test_fixed_cells_and_ties_select_eligible_deterministically(self):
        rows=records();out=select(rows)
        self.assertEqual(out['selected'],'typed-boundary')
        self.assertEqual(out,select(list(reversed(rows))))
        self.assertFalse(out['fresh_audit_used_for_selection'])

    def test_short_stoppage_and_precision_regressions_fail(self):
        for key in ('spellF1','macroF1','dead_precision','dead_recall','segment','boundary','short','precision'):
            rows=records()
            for row in rows:
                if row['candidate']!='typed-boundary':continue
                if key=='segment':row['temporal_diagnostics']['dead_segments']['f1']-=.01
                elif key=='boundary':row['temporal_diagnostics']['dead_onsets']['f1']-=.1
                elif key=='short':row['temporal_diagnostics']['short_possessions']['recall']-=.1
                elif key=='precision':row['duration_precision_recall']['bins']['<0.4s']['n_matched_pred']=0
                else:row[key]-=.03
            out=select(rows);self.assertFalse(out['gates']['typed-boundary']['passed'],key)

    def test_invalid_missing_duplicate_and_nonfinite_fail(self):
        rows=records()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):select(bad)
        bad=copy.deepcopy(rows);bad[0]['spellF1']=np.nan
        with self.assertRaises(ValueError):select(bad)

    def test_baseline_retained_but_one_experiment_reserved_if_gates_fail(self):
        rows=records()
        for row in rows:row['spellF1']=.6
        out=select(rows);self.assertEqual(out['selected'],BASELINE);self.assertIn(out['audit_candidate'],CANDIDATES)


class TypedCostDecoderTests(unittest.TestCase):
    def test_constant_vectors_are_exact_legacy(self):
        rng=np.random.default_rng(7);p=rng.dirichlet(np.ones(3),100);cfg=Config()
        np.testing.assert_array_equal(decode(p,25,cfg),decode(p,25,cfg,
            stop_costs=np.full(100,cfg.stop_cost),restart_costs=np.full(100,cfg.restart_cost)))

    def test_typed_costs_match_exhaustive_global_optimum(self):
        cfg=replace(Config(),min_team_seconds=.2,min_dead_seconds=.3);fps=10
        p=np.random.default_rng(13).dirichlet(np.ones(3),7);n=len(p)
        team=np.linspace(.1,.7,n);stop=np.linspace(.2,.8,n);restart=np.linspace(.9,.1,n)
        emission=-np.log(p)*cfg.effective_observations_per_second/fps
        def cost(path):
            edges=[0]+[i for i in range(1,n) if path[i]!=path[i-1]]+[n]
            for a,b in zip(edges[1:-2],edges[2:-1]):
                if b-a<(3 if path[a]==2 else 2):return np.inf
            total=sum(emission[i,k] for i,k in enumerate(path))
            for i in edges[1:-1]:total+=stop[i] if path[i]==2 else restart[i] if path[i-1]==2 else team[i]
            return total
        actual=decode(p,fps,cfg,team,stop_costs=stop,restart_costs=restart)
        self.assertAlmostEqual(cost(actual),min(cost(path) for path in product(range(3),repeat=n)),places=10)

    def test_each_malformed_typed_vector_rejected(self):
        p=np.full((6,3),1/3)
        for key in ('stop_costs','restart_costs'):
            for value in (np.zeros(5),np.zeros((6,1)),np.full(6,np.nan),np.full(6,-1)):
                with self.assertRaises(ValueError):decode(p,25,Config(),**{key:value})


if __name__=='__main__':unittest.main()

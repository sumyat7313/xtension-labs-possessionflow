import copy
from dataclasses import replace
from itertools import product
import unittest
import numpy as np
from possession.config import Config
from possession.decoder import decode
from tools.train_temporal_v6 import BASELINE,CANDIDATES,FOLDS,REGIMES,select


def records():
    rows=[]
    for name in (BASELINE,*CANDIDATES):
        gain=0 if name==BASELINE else .02
        for fold in FOLDS:
            for match in fold:
                for regime in REGIMES:
                    rows.append({'candidate':name,'match':match,'regime':regime,
                        'spellF1':.6+gain,'macroF1':.9,'dead_precision':.95,'dead_recall':.96,
                        'duration_precision_recall':{'bins':{k:{'n_pred':10,'n_matched_pred':5} for k in ('<0.4s','0.4-1s','1-2s')}},
                        'temporal_diagnostics':{'dead_segments':{'f1':.9},'dead_onsets':{'f1':.8+gain},
                            'dead_offsets':{'f1':.8+gain},'short_possessions':{'recall':.28+gain}}})
    return rows


class TemporalCampaignTests(unittest.TestCase):
    def test_complete_cells_select_deterministic_eligible_best(self):
        rows=records();d=select(rows)
        self.assertEqual(d['selected'],'combined')
        self.assertEqual(d,select(list(reversed(rows))))
        self.assertFalse(d['fresh_audit_used_for_selection'])

    def test_regressions_in_stoppages_or_short_spells_block_promotion(self):
        for key in ('spellF1','dead_recall','dead_precision','short','short_precision','dead_boundary','dead_segment'):
            rows=records()
            for r in rows:
                if r['candidate']!='combined':continue
                if key=='short':r['temporal_diagnostics']['short_possessions']['recall']=.27
                elif key=='short_precision':r['duration_precision_recall']['bins']['<0.4s']['n_matched_pred']=0
                elif key=='dead_boundary':r['temporal_diagnostics']['dead_onsets']['f1']=.7
                elif key=='dead_segment':r['temporal_diagnostics']['dead_segments']['f1']=.8
                else:r[key]-=.05
            d=select(rows)
            self.assertFalse(d['gates']['combined']['passed'],key)
            self.assertNotEqual(d['selected'],'combined')

    def test_missing_duplicate_invalid_scores_fail(self):
        rows=records()
        for bad in (rows[:-1],rows+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):select(bad)
        for value in (True,None,np.nan,1.1):
            bad=copy.deepcopy(rows);bad[0]['temporal_diagnostics']['short_possessions']['recall']=value
            with self.assertRaises(ValueError):select(bad)

    def test_no_improvement_keeps_baseline_but_audits_one_experiment(self):
        rows=records()
        for r in rows:
            r['spellF1']=.6;r['temporal_diagnostics']['short_possessions']['recall']=.28
            r['temporal_diagnostics']['dead_onsets']['f1']=r['temporal_diagnostics']['dead_offsets']['f1']=.8
        d=select(rows)
        self.assertEqual(d['selected'],BASELINE)
        self.assertIn(d['audit_candidate'],CANDIDATES)


class EarlyExitDecoderTests(unittest.TestCase):
    def test_disabled_flags_preserve_legacy_result_exactly(self):
        rng=np.random.default_rng(6);p=rng.dirichlet(np.ones(3),size=80)
        np.testing.assert_array_equal(decode(p,25,Config()),decode(p,25,Config(),early_exit_allowed=np.zeros(80,bool)))

    def test_decoder_matches_exhaustive_paths_with_early_exits(self):
        cfg=replace(Config(),min_team_seconds=.3,min_dead_seconds=.2,team_switch_cost=.2,stop_cost=.3,restart_cost=.2)
        fps=10;min_short=.1;penalty=.1
        rng=np.random.default_rng(10527)
        for case in range(3):
            p=rng.dirichlet(np.ones(3),size=7);early=np.array([False,True,True,False,True,True,True])
            costs=np.linspace(.1,.3,len(p));em=-np.log(p)*cfg.effective_observations_per_second/fps
            def cost(path):
                edges=[0]+[i for i in range(1,len(path)) if path[i]!=path[i-1]]+[len(path)]
                total=sum(em[i,label] for i,label in enumerate(path))
                for j,frame in enumerate(edges[1:-1],1):
                    old,new=path[frame-1],path[frame];duration=frame-edges[j-1]
                    extra=0.
                    if j>1 and duration<(3 if old<2 else 2):
                        if old>=2 or new>=2 or duration<1 or not early[frame]:return np.inf
                        extra=penalty
                    total+=(costs[frame] if old<2 and new<2 else cfg.stop_cost if new==2 else cfg.restart_cost)+extra
                return total
            result=decode(p,fps,cfg,costs,early_exit_allowed=early,min_short_team_seconds=min_short,early_exit_penalty=penalty)
            best=min(cost(path) for path in product(range(3),repeat=len(p)))
            self.assertAlmostEqual(cost(result),best,places=10)

    def test_early_exit_recovers_supported_brief_team_spell_only(self):
        p=np.tile([.999998,.000001,.000001],(40,1));p[15:20]=[.000001,.999998,.000001]
        cfg=replace(Config(),team_switch_cost=.1)
        early=np.zeros(40,bool);early[20]=True
        pred=decode(p,25,cfg,early_exit_allowed=early)
        np.testing.assert_array_equal(pred[15:20],1)
        # A short D segment cannot use the H/A early-exit route.
        p[15:20]=[.000001,.000001,.999998]
        ordinary=decode(p,25,cfg)
        np.testing.assert_array_equal(ordinary,decode(p,25,cfg,early_exit_allowed=early))

    def test_malformed_early_exit_controls_rejected(self):
        p=np.full((8,3),1/3)
        for bad in (np.ones(8),np.ones(7,bool)):
            with self.assertRaises(ValueError):decode(p,25,Config(),early_exit_allowed=bad)
        for kw in ({'min_short_team_seconds':0},{'min_short_team_seconds':.5},{'early_exit_penalty':-1}):
            with self.assertRaises(ValueError):decode(p,25,Config(),early_exit_allowed=np.ones(8,bool),**kw)

    def test_short_floor_rounds_up_at_noninteger_frame_duration(self):
        p=np.tile([.999998,.000001,.000001],(40,1));p[15:17]=[.000001,.999998,.000001]
        pred=decode(p,15,replace(Config(),team_switch_cost=.1),early_exit_allowed=np.ones(40,bool))
        edges=np.r_[0,np.flatnonzero(pred[1:]!=pred[:-1])+1,len(pred)]
        for start,stop in zip(edges[1:-2],edges[2:-1]):
            self.assertGreaterEqual(stop-start,3)  # ceil(.16*15), not round(.16*15)


if __name__=='__main__':unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import pandas as pd
from tools import fresh_benchmark_v6 as fresh


def invented(relative=False,mirror_y=True):
    halves=[]
    for half in range(2):
        rows=[]
        for frame in range(12):
            for team in (1,2):
                for number in (1,2):
                    sign=1 if team==1 else -1
                    scale=np.array([sign,sign if mirror_y else 1])
                    point=np.array([number*10+team,number*3+team])*scale
                    ball=np.array([frame,frame+2])*scale
                    if relative:ball-=point
                    rows.append(dict(x=point[0],y=point[1],x_ball=ball[0],y_ball=ball[1],
                        group_id=team,player_id=team*10+number,timestamp_ms=half*2700000+frame*100))
        halves.append(pd.DataFrame(rows)[fresh.OBS_COLUMNS])
    return halves


def paired_scores():
    dense={'n':100,'class_counts':{'H':30,'A':30,'D':40},'spellF1':.7,'macroF1':.8,
        'dead_precision':.9,'dead_recall':.8,'temporal':{k:{'n_gt':10,'f1':.6} for k in ('dead_segments','dead_onsets','dead_offsets')}}
    dense['temporal']['short_possessions']={'n_gt':10,'recall':.5}
    candidate=copy.deepcopy(dense);candidate['temporal']['short_possessions']['recall']=.6
    result={'wcsf2023':{'baseline':dense,'candidate':candidate}}
    for match in fresh.SPLIT['final_silver_matches']:
        result[f'skillcorner{match}']={'baseline':{'n':100,'accuracy':.8},'candidate':{'n':100,'accuracy':.8}}
    return result


def write_zip(path,frame):
    with ZipFile(path,'w') as z:z.writestr(path.stem,frame.to_csv(index=False))


class FreshV6Tests(unittest.TestCase):
    def test_mirroring_resolved_from_ball_copies_without_flags(self):
        for relative in (False,True):
            for mirrored in (False,True):
                t,blocks,pitch,alignment=fresh.wcsf_observations(invented(relative,mirrored))
                np.testing.assert_allclose(t.ball[3],[3/105+.5,.5-5/68])
                self.assertEqual(blocks,[(1,0,11),(2,12,23)])
                self.assertEqual(alignment['geometry'][0]['away_y_mirrored'],mirrored)
                self.assertEqual(alignment['geometry'][0]['ball_coordinates'],'player_offset' if relative else 'team_frame')
                self.assertAlmostEqual(t.ball_speed[4],np.sqrt(200))
                self.assertEqual(t.teams.tolist(),[0,0,1,1])

    def test_reference_columns_rejected_by_observation_api(self):
        frames=invented();frames[0]['AliveFlag']=1
        with self.assertRaisesRegex(ValueError,'whitelist'):fresh.wcsf_observations(frames)

    def test_ambiguous_or_inconsistent_geometry_fails_closed(self):
        frames=invented();frames[0]['y_ball']=0
        with self.assertRaisesRegex(ValueError,'Ambiguous'):fresh.wcsf_observations(frames)
        frames=invented();frames[0].loc[0,'x_ball']+=2
        with self.assertRaisesRegex(ValueError,'consistent ball'):fresh.wcsf_observations(frames)

    def test_timeline_duplicates_and_missing_frame_no_interpolation(self):
        frames=invented();frames[0]=pd.concat([frames[0],frames[0].iloc[[0]]])
        with self.assertRaisesRegex(ValueError,'Duplicate'):fresh.wcsf_observations(frames)
        frames=invented();frames[0]=frames[0][frames[0].timestamp_ms!=500]
        t,_,_,_=fresh.wcsf_observations(frames)
        self.assertTrue(np.isnan(t.ball[5]).all());self.assertTrue(np.isnan(t.players[5]).all())
        frames=invented();frames[0].loc[0,'timestamp_ms']=31
        with self.assertRaisesRegex(ValueError,'10 Hz'):fresh.wcsf_observations(frames)

    def test_relative_ball_requires_reconstructed_cross_team_support(self):
        frames=invented(relative=True)
        frames[0].loc[frames[0].group_id==2,['x','y']]=np.nan
        with self.assertRaisesRegex(ValueError,'consistent ball'):fresh.wcsf_observations(frames)

    def test_zip_reader_parses_only_explicit_column_whitelist(self):
        frames=invented();frames[0]['AliveFlag']='INVALID_REFERENCE_DO_NOT_PARSE';frames[0]['PossessionFlag']='NO'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'positions.csv.zip';write_zip(path,frames[0])
            observation=fresh._read_wcsf_columns(path,fresh.OBS_COLUMNS)
        self.assertEqual(set(observation.columns),set(fresh.OBS_COLUMNS))
        self.assertNotIn('AliveFlag',observation)

    def test_explicit_dead_flags_and_unknown_gaps(self):
        frames=invented();_,_,_,alignment=fresh.wcsf_observations(frames)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for half,name in enumerate(fresh.WCSF):
                frame=frames[half].copy();index=((frame.timestamp_ms-frame.timestamp_ms.min())//100)
                frame['AliveFlag']=np.where(index<5,1,0);frame['PossessionFlag']=np.where(index<3,1,2)
                frame=frame[index!=8]
                write_zip(folder/name,frame)
            with patch.object(fresh.old,'check_file'):
                labels,counts=fresh.wcsf_references(folder,alignment)
        self.assertEqual(''.join(labels),'HHHAADDDXDDD'*2)
        self.assertEqual(counts['missing_frames_unknown'],2)
        self.assertEqual(counts['class_counts']['D'],12)

    def test_reference_conflicts_do_not_turn_into_dead(self):
        frames=invented();_,_,_,alignment=fresh.wcsf_observations(frames)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for half,name in enumerate(fresh.WCSF):
                frame=frames[half].copy();frame['AliveFlag']=1;frame['PossessionFlag']=1
                if half==0:frame.loc[0,'PossessionFlag']=2
                write_zip(folder/name,frame)
            with patch.object(fresh.old,'check_file'):
                with self.assertRaisesRegex(ValueError,'contradictory'):fresh.wcsf_references(folder,alignment)

    def test_native_fps_metrics_and_half_barriers(self):
        reference=np.array(list('HHHHDDDAAAAAHHHHDDDAAAAA'))
        score=fresh.dense_score(reference,reference,[(1,0,11),(2,12,23)],10)
        self.assertEqual(score['spellF1'],1);self.assertEqual(score['dead_f1'],1)
        self.assertEqual(score['temporal']['short_possessions']['n_gt'],4)
        self.assertEqual(score['temporal']['dead_onsets']['n_gt'],2)
        self.assertFalse(score['dense_human_had_truth'])

    def test_gate_requires_improvement_and_rejects_each_regression(self):
        scores=paired_scores();self.assertTrue(fresh.decide(scores)['passed'])
        scores['wcsf2023']['candidate']['temporal']['short_possessions']['recall']=.5
        self.assertFalse(fresh.decide(scores)['passed'])
        for key,drop in [('spellF1',.006),('macroF1',.006),('dead_precision',.006),('dead_recall',.006)]:
            scores=paired_scores();scores['wcsf2023']['candidate'][key]-=drop
            self.assertFalse(fresh.decide(scores)['passed'])
        scores=paired_scores();scores['wcsf2023']['candidate']['temporal']['dead_segments']['f1']-=.011
        self.assertFalse(fresh.decide(scores)['passed'])
        scores=paired_scores();scores['skillcorner1925299']['candidate']['accuracy']-=.03;scores['skillcorner1927964']['candidate']['accuracy']+=.04
        self.assertFalse(fresh.decide(scores)['passed'])

    def test_gate_rejects_missing_support_and_invalid_scores(self):
        scores=paired_scores();scores['wcsf2023']['baseline']['temporal']['short_possessions'].update(n_gt=0,recall=None)
        scores['wcsf2023']['candidate']['temporal']['short_possessions'].update(n_gt=0,recall=None)
        self.assertFalse(fresh.decide(scores)['passed'])
        scores=paired_scores();scores['wcsf2023']['candidate']['spellF1']=float('nan')
        with self.assertRaises(ValueError):fresh.decide(scores)
        scores=paired_scores();scores.pop('skillcorner1925299')
        with self.assertRaises(ValueError):fresh.decide(scores)

    def test_reservation_and_policy_tampering_guard(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);fresh.freeze_split(folder)
            (folder/'final-reference-access.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already consumed'):fresh.evaluate_pair(folder,'no','no','no',folder/'out')
            (folder/'policy-before-reference-access.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'policy changed'):fresh.freeze_split(folder)

    def test_all_six_predictions_saved_before_any_reference_scoring(self):
        obs=fresh.wcsf_observations(invented());labels=np.array(list('HHHHDDDAAAAAHHHHDDDAAAAA'));calls=[]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);out=folder/'audit'
            for name in ['model.json','config.json',*fresh.WCSF,*[f'{m}_phases_of_play.csv' for m in fresh.SC]]:(folder/name).write_text('{}')
            def infer(tracking,events,*args):
                self.assertEqual(events,[]);calls.append('infer');return labels.copy(),None,None,None,[]
            def reference(*args):
                self.assertEqual(len(list(out.glob('*.csv'))),6);self.assertEqual(calls.count('infer'),6)
                calls.append('reference');return labels.copy(),{}
            def phase_reference(*args):
                value,counts=reference(*args);value[value=='D']='X';return value,counts
            with patch.object(fresh,'read_wcsf',return_value=obs),patch.object(fresh,'read_skillcorner',return_value=obs),patch.object(fresh.old,'check_file'),patch.object(fresh,'wcsf_references',side_effect=reference),patch.object(fresh.old,'phase_references',side_effect=phase_reference),patch('possession.learned.load_model',return_value={}),patch.object(fresh,'load_config',return_value={}),patch('possession.pipeline.infer',side_effect=infer):
                result=fresh.evaluate_pair(folder,folder/'model.json',folder/'model.json',folder/'config.json',out)
            self.assertEqual(calls,['infer']*6+['reference']*3)
            self.assertFalse(result['gate']['passed']) # identical models provide no weakness improvement
            self.assertFalse(result['reference_fields_used_for_inference'])
            self.assertTrue((folder/'final-reference-access.json').exists())


if __name__=='__main__':unittest.main()

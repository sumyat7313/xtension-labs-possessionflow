import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import fresh_benchmark as fresh


class ForbiddenFields(dict):
    def __getitem__(self,key):
        if key in {'possession','ball_state','phases','playing_time'}: raise AssertionError('Reference field was accessed')
        return super().__getitem__(key)
    def get(self,key,*args):
        if key in {'possession','ball_state','phases','playing_time'}: raise AssertionError('Reference field was accessed')
        return super().get(key,*args)


def invented():
    metadata={'players':[{'id':11,'team_id':1},{'id':22,'team_id':2}],
        'home_team':{'id':1,'name':'H'},'away_team':{'id':2,'name':'A'},'pitch_length':105,'pitch_width':68}
    rows=[ForbiddenFields(frame=i,period=1 if i<6 else 2,
        player_data=[{'player_id':11,'x':i,'y':0,'is_detected':True},{'player_id':22,'x':10,'y':2,'is_detected':False}],
        ball_data={'x':i,'y':1,'is_detected':i!=3},possession={'group':'home team'}) for i in range(12)]
    return metadata,rows


def scores(accuracy=.8):
    def pair(quality):
        a={'accuracy':accuracy,'balanced_team_recall':accuracy,'n':100,'dense_had_truth':False,'reference_quality':quality}
        return {'baseline':a,'candidate':dict(a)}
    return {'metrica3':pair('provider_sparse_events'),**{f'skillcorner{x}':pair('provider_silver') for x in (1886347,1899585)}}


class FreshBenchmarkTests(unittest.TestCase):
    def test_detected_only_observations_never_access_reference_fields(self):
        metadata,rows=invented();t,blocks,pitch,a=fresh._skillcorner_observations(metadata,rows)
        self.assertTrue(np.isnan(t.players[:,1]).all());self.assertTrue(np.isnan(t.ball[3]).all())
        self.assertEqual(a['fps'],10);self.assertEqual(blocks,[(1,0,5),(2,6,11)])
        np.testing.assert_allclose(t.players[2,0],[2/105+.5,.5])
        np.testing.assert_allclose(t.ball[2],[2/105+.5,.5-1/68])
        self.assertAlmostEqual(t.player_speed[2,0],10)

    def test_bad_tracking_timeline_and_duplicate_players_rejected(self):
        metadata,rows=invented();rows[2]['frame']=9
        with self.assertRaises(ValueError):fresh._skillcorner_observations(metadata,rows)
        metadata,rows=invented();rows[2]['player_data'].append(rows[2]['player_data'][0])
        with self.assertRaises(ValueError):fresh._skillcorner_observations(metadata,rows)

    def test_phase_gaps_are_unknown_and_conflicting_intervals_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'phases.csv'
            path.write_text('frame_start,frame_end,period,team_in_possession_id\n1,4,1,1\n3,5,1,2\n7,9,2,2\n')
            refs,counts=fresh.phase_references(path,{'periods':np.r_[np.ones(6),np.full(6,2)],'team_ids':[1,2]})
            self.assertEqual(''.join(refs),'XHHXAXXAAXXX')
            self.assertNotIn('D',refs);self.assertEqual(counts['contradictory_frames_excluded'],1)

    def test_misaligned_phase_coverage_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'phases.csv';path.write_text('frame_start,frame_end,period,team_in_possession_id\n1,7,1,1\n')
            with self.assertRaises(ValueError):fresh.phase_references(path,{'periods':np.r_[np.ones(6),np.full(6,2)],'team_ids':[1,2]})

    def test_sparse_pass_origins_exact_frames_no_interval_fill(self):
        def event(frame,team,kind=1,period=1):return {'start':{'frame':frame},'team':{'id':team},'type':{'id':kind},'period':period}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.json';path.write_text(json.dumps({'data':[event(1,'h'),event(1,'h'),event(3,'a',7),event(3,'h'),event(5,'a',2),event(7,'a',16,2)]}))
            refs,counts=fresh.metrica3_references(path,{'frame_ids':np.arange(1,13),'periods':np.r_[np.ones(6),np.full(6,2)],'team_ids':['h','a']})
            self.assertEqual(''.join(refs),'HXXXXXAXXXXX');self.assertEqual(counts['duplicates'],1);self.assertEqual(counts['conflicting_frames'],1)

    def test_false_dead_predictions_count_as_errors(self):
        r=fresh.score_reference(np.array(list('HAHX')),np.array(list('HDAX')),'provider_silver')
        self.assertAlmostEqual(r['accuracy'],1/3);self.assertIsNone(r['spellF1']);self.assertFalse(r['dense_had_truth'])
        self.assertEqual(r['metric'],'phase_present_team_accuracy')

    def test_strict_fresh_gate_rejects_human_regression(self):
        value=scores();self.assertTrue(fresh.decide(value)['passed'])
        value['metrica3']['candidate']['accuracy']-=.001
        self.assertFalse(fresh.decide(value)['passed'])
        value=scores();value['skillcorner1886347']['candidate']['accuracy']-=.03;value['skillcorner1899585']['candidate']['accuracy']+=.04
        self.assertFalse(fresh.decide(value)['passed'])
        value=scores();value['skillcorner1886347']['candidate']['accuracy']-=.01
        self.assertFalse(fresh.decide(value)['passed'])

    def test_split_tampering_and_repeated_final_audit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=fresh.freeze_split(folder);self.assertEqual(path,fresh.freeze_split(folder))
            (folder/'final-reference-access.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already consumed'):fresh.evaluate_pair(folder,'absent','absent','absent',folder/'out')
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'split changed'):fresh.freeze_split(folder)

    def test_epts_substitution_spec_preserves_global_player_slots(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            def fmt(a,b,players):
                groups=''.join(f'<SplitRegister><PlayerChannelRef playerChannelId="{v}x"/><PlayerChannelRef playerChannelId="{v}y"/></SplitRegister>' for v in players)
                return f'<DataFormatSpecification startFrame="{a}" endFrame="{b}"><StringRegister/><SplitRegister>{groups}</SplitRegister><SplitRegister/></DataFormatSpecification>'
            xml='<main><Score idLocalTeam="h" idVisitingTeam="a"/><Team id="h"><Name>H</Name></Team><Team id="a"><Name>A</Name></Team><Player id="p" teamId="h"/><Player id="q" teamId="a"/><FrameRate>25</FrameRate><FieldSize><Width>105</Width><Height>68</Height></FieldSize>'
            xml+=''.join(f'<PlayerChannel id="{p}{c}" playerId="{p}" channelId="{c}"/>' for p in 'pq' for c in 'xy')
            xml+=''.join(f'<ProviderParameter><Name>{k}</Name><Value>{v}</Value></ProviderParameter>' for k,v in [('first_half_start',1),('first_half_end',2),('second_half_start',3),('second_half_end',4)])
            xml+=fmt(1,2,'pq')+fmt(3,4,'qp')+'</main>'
            (folder/'Sample_Game_3_metadata.xml').write_text(xml)
            (folder/'Sample_Game_3_tracking.txt').write_text('1:0.1,0.2;0.8,0.9:NaN,NaN\n2:0.2,0.2;0.7,0.9:0.5,0.5\n3:0.6,0.9;0.3,0.2:0.5,0.5\n4:0.5,0.9;0.4,0.2:0.5,0.5\n')
            with patch.object(fresh,'check_file'):
                t,blocks,pitch,a=fresh.read_metrica3(folder)
            np.testing.assert_allclose(t.players[:,0,0],[.1,.2,.3,.4]);self.assertEqual(blocks,[(1,0,1),(2,2,3)])
            self.assertEqual(a['team_ids'],['h','a'])

    def test_all_predictions_saved_before_any_separate_reference_scoring(self):
        metadata,rows=invented();observations=fresh._skillcorner_observations(metadata,rows)
        labels=np.array(list('HAHAHAHAHAHA'));calls=[]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);out=folder/'audit'
            for name in ('model.json','config.json','Sample_Game_3_events.json','1886347_phases_of_play.csv','1899585_phases_of_play.csv'):
                (folder/name).write_text('{}')
            def infer(tracking,events,*args):
                self.assertEqual(events,[]);calls.append('infer');return labels.copy(),None,None,None,[]
            def reference(*args):
                self.assertEqual(len(list(out.glob('*.csv'))),6)
                self.assertEqual(calls.count('infer'),6);calls.append('reference');return labels.copy(),{}
            with patch.object(fresh,'read_metrica3',return_value=observations),patch.object(fresh,'read_skillcorner',return_value=observations),patch.object(fresh,'check_file'),patch.object(fresh,'metrica3_references',side_effect=reference),patch.object(fresh,'phase_references',side_effect=reference),patch('possession.learned.load_model',return_value={}),patch.object(fresh,'load_config',return_value={}),patch('possession.pipeline.infer',side_effect=infer):
                result=fresh.evaluate_pair(folder,folder/'model.json',folder/'model.json',folder/'config.json',out)
            self.assertEqual(calls,['infer']*6+['reference']*3)
            self.assertEqual(result['reference_read_count'],3);self.assertTrue(result['gate']['passed'])
            self.assertFalse(result['reference_fields_used_for_inference'])
            self.assertTrue((folder/'final-reference-access.json').exists())


if __name__=='__main__':unittest.main()

import copy
import unittest
from unittest.mock import patch

import numpy as np

from tools import pff_adapter as pff


def metadata():
    return [{'id':'3812','fps':29.97,'homeTeam':{'id':'1','name':'home'},'awayTeam':{'id':'2','name':'away'},
             'stadium':{'pitches':[{'length':105,'width':68}]}}]


def frames():
    return [{'gameRefId':3812,'frameNum':i,'period':1,'videoTimeMs':1000*i/29.97,
             'homePlayers':[{'jerseyNum':1,'visibility':'VISIBLE','x':-20.,'y':5.}],
             'awayPlayers':[{'jerseyNum':1,'visibility':'VISIBLE','x':20.,'y':-5.}],
             'balls':[{'visibility':'VISIBLE','x':i*.02,'y':0.,'z':0.}],
             'ballsSmoothed':{'x':-20.,'y':5.},'game_event':{'home_ball':True,'game_event_type':'OTB'}}
            for i in range(60)]


class PFFAdapterTests(unittest.TestCase):
    def test_compact_records_match_original_tuple_storage_exactly(self):
        rows=frames()
        for row in rows:
            row['homePlayers'][0]['x']=float(np.nextafter(row['frameNum']*.137, np.inf))
        with patch.object(pff,'_pack_players',side_effect=lambda values:values), \
             patch.object(pff,'_players',side_effect=lambda record:iter(record['players'])):
            original=pff.observations(metadata(),rows)
        compact=pff.observations(metadata(),rows)
        for name in ('players','player_speed','ball','ball_speed','teams'):
            np.testing.assert_array_equal(getattr(original[0],name),getattr(compact[0],name))
        self.assertEqual(original[1:],compact[1:])

    def test_raw_geometry_ignores_smoothed_and_all_reference_fields(self):
        rows=frames();a=pff.observations(metadata(),rows)
        altered=copy.deepcopy(rows)
        for row in altered:
            row['ballsSmoothed']={'x':9999,'y':9999};row['homePlayersSmoothed']=[]
            row['game_event']={'home_ball':False,'game_event_type':'OUT'}
            row['possession_event']={'team_id':999}
        b=pff.observations(metadata(),altered)
        np.testing.assert_array_equal(a[0].ball,b[0].ball)
        np.testing.assert_array_equal(a[0].players,b[0].players)
        self.assertEqual(a[3],b[3]);self.assertEqual(a[3]['fps'],25)

    def test_estimated_and_ambiguous_ball_are_missing(self):
        rows=frames()
        for row in rows[:15]:row['balls'][0]['visibility']='ESTIMATED'
        for row in rows[15:30]:row['balls'].append({'visibility':'VISIBLE','x':0,'y':2})
        tracking,_,_,align=pff.observations(metadata(),rows)
        source=np.asarray(align['source_frames'])
        self.assertTrue(np.isnan(tracking.ball[source<30]).all())
        self.assertTrue(np.isfinite(tracking.ball[source>=30]).all())

    def test_estimated_player_does_not_become_observed(self):
        rows=frames()
        for row in rows[:30]:row['homePlayers'][0]['visibility']='ESTIMATED'
        tracking,_,_,align=pff.observations(metadata(),rows)
        self.assertTrue(np.isnan(tracking.players[np.asarray(align['source_frames'])<30,0]).all())

    def test_event_duplicate_frames_collapsed_but_conflicting_geometry_rejected(self):
        rows=frames();duplicate=copy.deepcopy(rows[15]);duplicate['game_event']={'home_ball':False}
        rows.insert(16,duplicate)
        a=pff.observations(metadata(),rows)
        self.assertEqual(a[-1]['duplicates_collapsed'],1)
        rows[16]['balls'][0]['x']+=1
        with self.assertRaises(ValueError):pff.observations(metadata(),rows)

    def test_missing_native_frames_never_interpolated(self):
        rows=[r for r in frames() if r['frameNum'] not in range(20,30)]
        tracking,_,_,align=pff.observations(metadata(),rows)
        missing=np.array(align['source_frames'])<0
        self.assertTrue(missing.any());self.assertTrue(np.isnan(tracking.ball[missing]).all())

    def test_coordinate_origin_and_y_axis(self):
        tracking,_,pitch,_=pff.observations(metadata(),frames())
        self.assertEqual(pitch,(105,68))
        np.testing.assert_allclose(tracking.players[0,0],[-20/105+.5,.5-5/68],rtol=1e-6)

    def test_speed_does_not_cross_periods(self):
        rows=frames()
        for row in rows[30:]:row['period']=2;row['balls'][0]['x']=40
        tracking,blocks,_,_=pff.observations(metadata(),rows)
        self.assertEqual(len(blocks),2)
        self.assertTrue(np.isnan(tracking.ball_speed[blocks[1][1]]))
        separator=blocks[0][2]+1
        self.assertEqual(blocks[1][1],separator+1)
        self.assertTrue(np.isnan(tracking.ball[separator]).all())

    def test_invalid_identifiers_order_and_fps_fail_closed(self):
        for bad in ('id','order','fps','coordinate','clock'):
            rows=frames();meta=metadata()
            if bad=='id':rows[0]['gameRefId']=9
            elif bad=='order':rows[0],rows[1]=rows[1],rows[0]
            elif bad=='fps':meta[0]['fps']=0
            elif bad=='coordinate':rows[0]['balls'][0]['x']=float('inf')
            elif bad=='clock':rows[0]['videoTimeMs']=1000
            with self.assertRaises(ValueError):pff.observations(meta,rows)

    def test_optional_repeated_header_identity_matches_native_schema(self):
        rows=frames()
        for row in rows[1:]:row['gameRefId']=None
        self.assertGreater(len(pff.observations(metadata(),rows)[0].ball),0)
        rows[0]['gameRefId']=None
        with self.assertRaises(ValueError):pff.observations(metadata(),rows)

    def test_reference_carry_missing_flags_and_period_reset(self):
        alignment={'match':'3812','source_frames':list(range(12))+[-1]+list(range(20,25)),
                   'video_seconds':list(np.arange(12)/25)+[None]+list(10+np.arange(5)/25),
                   'periods':[1]*12+[0]+[2]*5,'reference_fields_used_for_inference':False}
        def event(i,t,kind,owner=None,**flags):
            return {'gameId':3812,'gameEventId':i,'startTime':t,'endTime':t,
                    'gameEvents':{'period':1,'gameEventType':kind,'homeTeam':owner,**flags}}
        events=[event(1,0,'FIRSTKICKOFF',True),event(2,.12,'OUT'),event(3,.24,'OTB',False),
                event(4,.32,'OTB',True,videoMissing=True),event(5,.4,'OTB',True)]
        y,c=pff.references(events,alignment)
        self.assertEqual(''.join(y),'HHHDDDAAXXHHXXXXXX')
        self.assertFalse(c['dense_human_had_truth']);self.assertEqual(c['uncertain_game_events'],1)

    def test_reference_duplicates_collapse_and_conflicts_fail_closed(self):
        alignment={'match':'3812','source_frames':[0,1,2], 'video_seconds':[0,.04,.08],
                   'periods':[1]*3,'reference_fields_used_for_inference':False}
        event={'gameId':3812,'gameEventId':1,'startTime':0.,'endTime':0.,
               'gameEvents':{'period':1,'gameEventType':'OTB','homeTeam':True}}
        y,c=pff.references([event,copy.deepcopy(event)],alignment)
        self.assertEqual(''.join(y),'HHH');self.assertEqual(c['duplicate_event_rows_collapsed'],1)
        conflict=copy.deepcopy(event);conflict['gameEvents']['homeTeam']=False
        with self.assertRaises(ValueError):pff.references([event,conflict],alignment)
        conflict['gameEventId']=2
        self.assertEqual(''.join(pff.references([event,conflict],alignment)[0]),'XXX')

    def test_nullable_uncertainty_flags_reset_until_known_control(self):
        alignment={'match':'3812','source_frames':list(range(6)),
                   'video_seconds':[0,.04,.08,.12,.16,.2],
                   'periods':[1]*6,'reference_fields_used_for_inference':False}
        def event(i,t,kind,owner=None,**flags):
            return {'gameId':3812,'gameEventId':i,'startTime':t,'endTime':None,
                    'gameEvents':{'period':1,'gameEventType':kind,'homeTeam':owner,
                                  'videoMissing':False,'initialNonEvent':False,**flags}}
        for flag in ('videoMissing','initialNonEvent'):
            with self.subTest(flag=flag):
                events=[event(1,0,'OTB',True),event(2,.04,'FOUL',**{flag:None}),
                        event(3,.12,'FOUL'),event(4,.16,'OTB',False)]
                y,c=pff.references(events,alignment)
                self.assertEqual(''.join(y),'HXXXAA')
                self.assertEqual(c['uncertain_game_events'],1)
                self.assertEqual(c['class_counts']['D'],0)

    def test_non_boolean_non_null_uncertainty_flags_fail_closed(self):
        alignment={'match':'3812','source_frames':[0], 'video_seconds':[0.],
                   'periods':[1],'reference_fields_used_for_inference':False}
        for flag in ('videoMissing','initialNonEvent'):
            for value in (0,1,'false',[],{}):
                with self.subTest(flag=flag,value=value):
                    event={'gameId':3812,'gameEventId':1,'startTime':0.,'endTime':None,
                           'gameEvents':{'period':1,'gameEventType':'FOUL',flag:value}}
                    with self.assertRaisesRegex(ValueError,'Invalid event uncertainty flag'):
                        pff.references([event],alignment)

    def test_native_reused_id_preserves_control_and_uncertain_foul(self):
        alignment={'match':'3812','source_frames':list(range(5)),
                   'video_seconds':[0,.04,.08,.12,.16],
                   'periods':[1]*5,'reference_fields_used_for_inference':False}
        control={'gameId':3812,'gameEventId':1,'startTime':0.,'endTime':0.,
                 'gameEvents':{'period':1,'gameEventType':'OTB','homeTeam':True,
                               'videoMissing':False,'initialNonEvent':False}}
        foul={'gameId':3812,'gameEventId':1,'startTime':.08,'endTime':None,
              'gameEvents':{'period':1,'gameEventType':'FOUL','homeTeam':None,
                            'videoMissing':None,'initialNonEvent':None}}
        y,c=pff.references([control,foul,copy.deepcopy(foul)],alignment)
        self.assertEqual(''.join(y),'HHXXX')
        self.assertEqual(c['unique_game_events'],2)
        self.assertEqual(c['reused_game_event_ids'],1)
        self.assertEqual(c['duplicate_event_rows_collapsed'],1)
        np.testing.assert_array_equal(y,pff.references([foul,control],alignment)[0])

    def test_reference_unknown_events_and_missing_observations_do_not_make_dead(self):
        alignment={'match':'3812','source_frames':[0,1,-1,4,5], 'video_seconds':[0,.04,.08,.12,.16],
                   'periods':[1]*5,'reference_fields_used_for_inference':False}
        event={'gameId':3812,'gameEventId':1,'startTime':0.,'endTime':0.,
               'gameEvents':{'period':1,'gameEventType':'OTB','homeTeam':True}}
        self.assertEqual(''.join(pff.references([event],alignment)[0]),'HHXXX')
        event['gameEvents']['gameEventType']='NEW_UNDOCUMENTED'
        self.assertEqual(''.join(pff.references([event],alignment)[0]),'XXXXX')


if __name__=='__main__':unittest.main()

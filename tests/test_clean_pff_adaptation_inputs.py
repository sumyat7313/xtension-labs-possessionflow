"""Synthetic filesystem checks; never open actual provider records or weights."""
import copy
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools import clean_pff_adaptation_inputs as bridge


class AdaptationInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.patch = patch.object(bridge, 'ROOT', self.root)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.mid = bridge.protocol.INNER_TRAIN_IDS[0]
        self.match = f'pff{self.mid}'
        self.sportec = bridge.protocol.SPORTEC_IDS[0]
        self.context = dict(snapshot={'source_hashes': {'source.py': self.write('source.py', b'pass\n')}},
                            producer_sources={'source.py': self.digest('source.py')},
                            clean_sources={'source.py': self.digest('source.py')},
                            pff={}, sportec_seals={}, sportec_inputs={})

    def write(self, relative, data):
        p = self.root / relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data); return self.digest(relative)

    def digest(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def json(self, relative, value):
        return self.write(relative, (json.dumps(value, indent=2, sort_keys=True)+'\n').encode())

    def array(self, relative, value):
        p=self.root/relative; p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, value, allow_pickle=False); return self.digest(relative)

    def pff_fixture(self, *, item_changes=None, x=None, z=None, y=None):
        folder=f'{bridge.PFF_CACHE}/{self.mid}'
        item=dict(match=self.match, blocks=[[1,0,5]], fps=25., usable=[True], pitch=[105.,68.],
                  reference_fields_used_for_inference=False)
        item.update(item_changes or {})
        alignment=dict(match=self.mid, fps=25., source_frames=list(range(6)),
                       video_seconds=[i/25 for i in range(6)], periods=[1]*6,
                       reference_fields_used_for_inference=False)
        files={'item.json':self.json(f'{folder}/item.json',item),
               'alignment.json':self.json(f'{folder}/alignment.json',alignment),
               'x.npy':self.array(f'{folder}/x.npy',np.zeros((6,98)) if x is None else x),
               'z.npy':self.array(f'{folder}/z.npy',np.zeros((6,58)) if z is None else z)}
        # Historical upstream files are intentionally unusable, and must stay unopened.
        for name in ('base_home.npy','base_dead.npy','baseline.npy'):
            files[name]=self.write(f'{folder}/{name}',b'forbidden historical upstream')
        event_hash=self.json(f'data/pff/Event Data/{self.mid}.json',[])
        manifest=dict(match_id=self.mid,identity=dict(events=event_hash,sources=self.context['producer_sources']),
                      files=files,rows=6)
        self.json(f'{folder}/complete.json',manifest)
        self.context['pff'][self.match]=manifest
        return np.array(list('HADHAX')) if y is None else y

    def sportec_fixture(self, *, metadata_changes=None, support_changes=None):
        folder=f'{bridge.REBUILD}/cache/{self.sportec}'
        inputs={f'{self.sportec}_{kind}.xml':dict(sha256=str(i)*64,bytes=10,published_md5='0'*32)
                for i,kind in enumerate(('metadata','tracking'),1)}
        self.context['sportec_inputs'].update(inputs)
        meta=dict(match=self.sportec,blocks=[[1,0,5]],pitch=[105.,68.],alignment={},
                  inputs={k:{name:v[name] for name in ('sha256','bytes')} for k,v in inputs.items()})
        meta.update(metadata_changes or {})
        seal={'metadata.json':self.json(f'{folder}/metadata.json',meta),
              'reference.npy':self.array(f'{folder}/reference.npy',np.array(list('HADHAX')))}
        for regime in ('clean','degraded','camera'):
            for key,width in (('x',98),('z',58)):
                relative=f'{folder}/{regime}-{key}.npy'
                seal[f'{regime}-{key}.npy']=(self.array(relative,np.zeros((6,width))) if regime=='clean'
                                           else self.write(relative,b'forbidden augmentation'))
            support={'usable':[True]}; support.update(support_changes or {})
            seal[f'{regime}-support.json']=self.json(f'{folder}/{regime}-support.json',support)
        self.context['sportec_seals'][self.sportec]=self.json(f'{folder}/complete.json',seal)
        return seal

    def factory(self):
        with patch.object(bridge,'_evidence',return_value=self.context):
            return bridge.make_inner_factory(fixture_ids=bridge.protocol.fit_ids(),
                                            protocol_sha256=bridge.protocol.fixed_protocol_sha256())

    @contextmanager
    def synthetic_evidence(self):
        """Only aggregate-semantic validators are stubbed; file/lineage checks run."""
        ancestor=self.json('models/synthetic-ancestor.json',{'synthetic':'ancestor'})
        candidate=self.json(f'{bridge.REBUILD}/refit/candidate.json',{'synthetic':'candidate'})
        lineage=dict(schema_version=1,candidate_sha256=candidate,
                     comparator_sha256=bridge.policy.original.BASELINE_SHA256,fit_inputs=['d'*64],artifacts=[
            dict(sha256=candidate,kind='model',scope='external',identifier='new external',parents=[ancestor,'d'*64]),
            dict(sha256=ancestor,kind='model',scope='external',identifier='external ancestor',parents=['d'*64]),
            dict(sha256='d'*64,kind='dataset',scope='external',identifier='synthetic dataset',parents=[])])
        lineage_hash=self.json(f'{bridge.REBUILD}/refit/lineage.json',lineage)
        config_hash=self.json(f'{bridge.REBUILD}/refit/config.json',{'synthetic':'config'})
        plan_hash=self.json(f'{bridge.REBUILD}/protocol.json',{})
        summary_hash=self.json(f'{bridge.REBUILD}/summary.json',{})
        snapshot_hash=self.json('models/clean-pff-adaptation-protocol.json',self.context['snapshot'])
        old={};old_hash=self.json('models/pff-research.json',old)
        clean=dict(protocol=dict(source_hashes=self.context['clean_sources'],inputs={}),
                   completed_stage_seals={f'prepare-{mid}':'e'*64 for mid in bridge.protocol.SPORTEC_IDS})
        clean_hash=self.json('models/external-rebuild-001-research.json',clean)
        ctx=dict(reservation=dict(protocol_sha256=bridge.protocol.RESERVATION_SHA256,
                                 split={'train':list(bridge.protocol.TRAIN_IDS)}),lineage=lineage,
                 model_paths={candidate:f'{bridge.REBUILD}/refit/candidate.json',ancestor:'models/synthetic-ancestor.json'},
                 cache_manifests=[dict(match_id=mid,identity={'sources':self.context['producer_sources']})
                                  for mid in bridge.protocol.INNER_TRAIN_IDS])
        with ExitStack() as stack:
            for obj,name,value in ((bridge,'ADAPTATION_SNAPSHOT_SHA256',snapshot_hash),
                (bridge.protocol,'CLEAN_SHA256',candidate),(bridge.protocol,'CONFIG_SHA256',config_hash),
                (bridge.dev,'PARENT_BUNDLE_SHA',old_hash),(bridge.dev,'REBUILD_BUNDLE_SHA',clean_hash),
                (bridge.dev,'LINEAGE_FILE_SHA',lineage_hash),(bridge.dev,'REBUILD_PROTOCOL_SHA',plan_hash),
                (bridge.dev,'REBUILD_SUMMARY_SHA',summary_hash)):
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(bridge.protocol,'verify_protocol'))
            stack.enter_context(patch.object(bridge.dev,'portable_context',return_value=ctx))
            stack.enter_context(patch.object(bridge.kernel,'validate_parent'))
            stack.enter_context(patch.object(bridge.prior,'sources',return_value=self.context['producer_sources']))
            yield ctx

    def test_bad_roster_and_core_rejected_before_any_evidence_io(self):
        roster=list(bridge.protocol.fit_ids())
        for ids in (roster[:-1],roster[::-1],['pff3814',*roster[1:]],roster+[roster[0]]):
            with patch.object(bridge,'_evidence') as evidence, self.assertRaises(ValueError):
                bridge.make_inner_factory(fixture_ids=ids,protocol_sha256=bridge.protocol.fixed_protocol_sha256())
            evidence.assert_not_called()
        with patch.object(bridge,'_evidence') as evidence, self.assertRaises(ValueError):
            bridge.make_inner_factory(fixture_ids=roster,protocol_sha256='a'*64)
        evidence.assert_not_called()

    def test_forbidden_fixture_rejected_before_any_filesystem_call(self):
        factory=self.factory()
        for match in ('pff3814','pff3813',f'pff{bridge.protocol.INNER_VALIDATION_IDS[0]}','../private','J03WMX-camera',self.mid):
            with patch.object(bridge,'_path') as path, self.assertRaises(ValueError): factory(match)
            path.assert_not_called()

    def test_construction_is_lazy_and_provenance_does_not_authorize_stages(self):
        with patch.object(bridge,'_array') as arrays, patch.object(bridge.prior.pff_adapter,'references') as labels:
            factory=self.factory()
        arrays.assert_not_called();labels.assert_not_called()
        v=factory.provenance()
        for key in ('stage_execution_authorized','refit_authorized','development_accessed','final_accessed',
                    'promoted','submission_eligible','artifact_delivery_rights_verified'):
            self.assertIs(v[key],False)
        self.assertEqual(v['sportec_regime'],'clean')

    def test_native_pff_returns_only_kernel_schema_and_never_old_upstream(self):
        y=self.pff_fixture(); loaded=[]
        original=bridge._array
        def load(name,sha):loaded.append(name);return original(name,sha)
        with patch.object(bridge,'_array',side_effect=load), \
             patch.object(bridge.prior.pff_adapter,'references',return_value=(y,{})) as labels:
            result=self.factory()(self.match)
        self.assertEqual(set(result),bridge.kernel.RECORD_KEYS)
        np.testing.assert_array_equal(result['y'],y)
        self.assertEqual([Path(v).name for v in loaded],['x.npy','z.npy'])
        labels.assert_called_once()

    def test_changed_cache_manifest_rejected_before_arrays_or_events(self):
        self.pff_fixture()
        path=f'{bridge.PFF_CACHE}/{self.mid}/complete.json'
        manifest=copy.deepcopy(self.context['pff'][self.match]);manifest['rows']=7
        self.json(path,manifest)
        with patch.object(bridge,'_array') as arrays,patch.object(bridge.prior.pff_adapter,'references') as labels:
            with self.assertRaisesRegex(ValueError,'manifest'):self.factory()(self.match)
        arrays.assert_not_called();labels.assert_not_called()

    def test_changed_feature_bytes_rejected_before_reference_read(self):
        self.pff_fixture();self.write(f'{bridge.PFF_CACHE}/{self.mid}/x.npy',b'changed')
        with patch.object(bridge.prior.pff_adapter,'references') as labels:
            with self.assertRaisesRegex(ValueError,'before materialization'):self.factory()(self.match)
        labels.assert_not_called()

    def test_changed_event_bytes_rejected_before_reference_parser(self):
        self.pff_fixture();self.write(f'data/pff/Event Data/{self.mid}.json',b'[{}]')
        with patch.object(bridge.prior.pff_adapter,'references') as labels:
            with self.assertRaisesRegex(ValueError,'bytes differ'):self.factory()(self.match)
        labels.assert_not_called()

    def test_invalid_observation_declaration_rejected_before_arrays(self):
        for change in ({'match':'pff3814'},{'reference_fields_used_for_inference':True},{'fps':24},
                       {'base_dead':'forbidden'},{'usable':[1]}):
            self.pff_fixture(item_changes=change)
            with patch.object(bridge,'_array') as arrays,self.assertRaises(ValueError):self.factory()(self.match)
            arrays.assert_not_called()

    def test_invalid_feature_dimensions_rejected_before_references(self):
        self.pff_fixture(x=np.zeros((6,97)))
        with patch.object(bridge.prior.pff_adapter,'references') as labels:
            with self.assertRaisesRegex(ValueError,'schema'):self.factory()(self.match)
        labels.assert_not_called()

    def test_invalid_reference_labels_are_rejected(self):
        self.pff_fixture()
        with patch.object(bridge.prior.pff_adapter,'references',return_value=(np.array(list('HADHQQ')),{})):
            with self.assertRaises(ValueError):self.factory()(self.match)

    def test_sportec_loads_only_clean_regime_and_pinned_reference(self):
        self.sportec_fixture();loaded=[];original=bridge._array
        def load(name,sha):loaded.append(Path(name).name);return original(name,sha)
        with patch.object(bridge,'_array',side_effect=load):result=self.factory()(self.sportec)
        self.assertEqual(loaded,['clean-x.npy','clean-z.npy','reference.npy'])
        self.assertEqual(set(result),bridge.kernel.RECORD_KEYS)

    def test_self_consistent_changed_sportec_seal_cannot_replace_published_identity(self):
        seal=self.sportec_fixture();folder=f'{bridge.REBUILD}/cache/{self.sportec}'
        seal['clean-x.npy']=self.array(f'{folder}/clean-x.npy',np.ones((6,98)))
        self.json(f'{folder}/complete.json',seal)
        with patch.object(bridge,'_array') as arrays,self.assertRaisesRegex(ValueError,'bytes differ'):
            self.factory()(self.sportec)
        arrays.assert_not_called()

    def test_sportec_wrong_match_or_input_ancestry_rejected_before_arrays(self):
        for change in ({'match':bridge.protocol.SPORTEC_IDS[1]},{'inputs':{}}):
            self.sportec_fixture(metadata_changes=change)
            with patch.object(bridge,'_array') as arrays,self.assertRaises(ValueError):self.factory()(self.sportec)
            arrays.assert_not_called()

    def test_modified_source_rejected_before_any_fixture_io(self):
        self.pff_fixture();factory=self.factory();self.write('source.py',b'changed source')
        with patch.object(bridge,'_pff_record') as loader,self.assertRaises(ValueError):factory(self.match)
        loader.assert_not_called()

    def test_repository_path_escape_rejected(self):
        for relative in ('../private','C:/private','/private','tools\\private'):
            with self.assertRaises(ValueError):bridge._path(relative)

    def test_linked_paths_rejected_before_array_materialization(self):
        target=self.root/'target.npy';np.save(target,np.zeros(2),allow_pickle=False)
        link=self.root/'link.npy'
        try:link.symlink_to(target)
        except (OSError,NotImplementedError):self.skipTest('Symlink creation unavailable')
        with patch.object(bridge.np,'load') as loader,self.assertRaises(ValueError):bridge._array('link.npy',self.digest('target.npy'))
        loader.assert_not_called()

    def test_array_mutation_during_load_is_rejected(self):
        identity=self.array('x.npy',np.zeros((2,98)));original=bridge.np.load
        def mutate(stream,**kwargs):
            result=original(stream,**kwargs)
            with (self.root/'x.npy').open('ab') as output:output.write(b'changed')
            return result
        with patch.object(bridge.np,'load',side_effect=mutate),self.assertRaisesRegex(ValueError,'during materialization'):
            bridge._array('x.npy',identity)

    def test_evidence_rejects_unpinned_snapshot_before_aggregate_context(self):
        self.json('models/clean-pff-adaptation-protocol.json',{})
        with patch.object(bridge.dev,'portable_context') as context,self.assertRaisesRegex(ValueError,'bytes differ'):
            bridge._evidence()
        context.assert_not_called()

    def test_parent_handoff_reauthenticates_content_and_file_identity(self):
        factory=self.factory();self.json(f'{bridge.REBUILD}/refit/candidate.json',{})
        with patch.object(bridge.kernel,'validate_parent') as validate,self.assertRaises(ValueError):factory.parent_model()
        validate.assert_not_called()

    def test_evidence_authenticates_synthetic_actual_ancestors_without_loading_records(self):
        with self.synthetic_evidence(), patch.object(bridge,'_array') as arrays:
            result=bridge._evidence()
        self.assertEqual(len(result['pff']),32)
        self.assertEqual(set(result['sportec_seals']),set(bridge.protocol.SPORTEC_IDS))
        arrays.assert_not_called()

    def test_evidence_rejects_changed_actual_ancestor_despite_consistent_declared_lineage(self):
        with self.synthetic_evidence(), patch.object(bridge.kernel,'validate_parent') as parent:
            self.write('models/synthetic-ancestor.json',b'changed ancestor')
            with self.assertRaisesRegex(ValueError,'Artifact bytes differ'):bridge._evidence()
        parent.assert_not_called()

    def test_simulated_linked_ancestor_rejected_without_symlink_privilege(self):
        identity=self.array('linked/x.npy',np.zeros(2))
        ancestor=self.root/'linked'
        with patch.object(Path,'is_symlink',lambda path:path==ancestor), \
             patch.object(bridge.np,'load') as loader, self.assertRaisesRegex(ValueError,'Linked filesystem'):
            bridge._array('linked/x.npy',identity)
        loader.assert_not_called()


if __name__=='__main__':unittest.main()

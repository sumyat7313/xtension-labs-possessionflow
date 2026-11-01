"""Tampering tests for research evidence; fixtures never change release artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from possession.io import sha256,write_json
from tools import verify_refinement_v8 as verifier
from tools.data_intake_v8 import MATCHES
from test_refinement_campaign_v8 import records
from test_precision_v8 import invented_model as precision_model
from test_stoppage_v8 import invented_model as stoppage_model


def heads(metadata):
    result={'precision':precision_model(),'stoppage':stoppage_model()}
    for model in result.values():model['metadata']=copy.deepcopy(metadata)
    return result


def sparse_report(match):
    document=json.loads((verifier.ROOT/f'research/annotations/elastic/{match}-points.json').read_text())
    first={int(k):v[2] for k,v in document['frame_ranges'].items()}
    report={'reference_scope':'sparse corrected human event timing on exposed development matches',
      'fresh_match':False,'dense_had_truth':False,'unlisted_frames_are_negatives':False}
    for kind in ('restart','ball_out'):
        count=len({(p['period'],p['team'] if kind=='restart' else None,p['index']) for p in document['points']
          if p['kind']==kind and p['index']!=first[p['period']]})
        report[kind]={'n_reference_points':count,'n_matched':0,'recall_within_1s':0.,
          'mean_absolute_error_matched_seconds':None,'mean_signed_error_matched_seconds':None,
          'precision':None,'f1':None,'tolerance_seconds':1.}
    return report


class RefinementVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        v=verifier;source=v.scientific_hashes()
        protocol={'schema_version':8,'source_hashes':source,'baseline_sha256':v.BASELINE_SHA256,
          'config_sha256':v.CONFIG_SHA256,'folds':v.FOLDS,'regimes':v.REGIMES,'primary_regimes':v.PRIMARY_REGIMES,
          'candidates':[v.BASELINE,*v.CANDIDATES],'gate':v.GATE,'precision_policy':v.precision_v8.POLICY,
          'stoppage_policy':v.stoppage_v8.POLICY,'combination_order':['precision','stoppage'],
          'competition_labels_used':False,'independent_test_claimed':False,'inference_receives_annotations':False,
          'fresh_audit_used_for_selection':False,'annotation_hashes':{p.name:sha256(p) for p in
          sorted((v.ROOT/'research/annotations/elastic').glob('*-points.json'))}}
        protocol=json.loads(json.dumps(protocol))
        write_json(cls.folder/'protocol.json',protocol);ph=sha256(cls.folder/'protocol.json')
        old=json.loads((v.ROOT/'models/joint-selection.json').read_text())
        old={(r['match'],r['regime']):r['prediction_sha256'] for r in old['records'] if r['candidate']=='joint-state'}
        rows=records();folds={m:i for i,held in enumerate(v.FOLDS) for m in held}
        sparse={m:sparse_report(m) for m in MATCHES}
        for row in rows:
            row['fold']=folds[row['match']]
            row['prediction_sha256']=old[row['match'],row['regime']] if row['candidate']==v.BASELINE else 'a'*64
            if row['match'] in sparse:row['human_sparse_timing']=copy.deepcopy(sparse[row['match']])
        selection=v.select(rows)
        cls.model={'schema_version':8,'kind':'isolated_refinement','base_model_sha256':v.BASELINE_SHA256,
          'candidate_id':selection['audit_candidate'],**heads({'protocol_sha256':ph,'training_matches':sorted(v.CATALOG),
          'training_regimes':list(v.REGIMES),'stage':'all-seven-refit','selected':selection['audit_candidate']})}
        cls.original_path=cls.folder/'original-model.json';write_json(cls.original_path,cls.model)
        fold_models={str(i):heads({'protocol_sha256':ph,'training_matches':sorted(set(v.CATALOG)-set(held)),
          'held_out_matches':list(held),'fold':i,'training_regimes':list(v.REGIMES)}) for i,held in enumerate(v.FOLDS)}
        cls.evidence={'protocol':protocol,'protocol_sha256':ph,'source_hashes':source,'records':rows,'selection':selection,
          'fold_models':fold_models,'competition_labels_used':False,'independent_test_claimed':False,
          'incumbent_historical_predictions_reproduced':True,'experimental_model_sha256':sha256(cls.original_path),
          'release_decision':{'promoted':False,'model_sha256':v.BASELINE_SHA256,'reason':'No independent dense truth.'}}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.e=copy.deepcopy(self.evidence);self.model_path=self.folder/'test-model.json'
        self.model_path.write_bytes(self.original_path.read_bytes())

    def check(self):return verifier.verify(self.e,self.model_path)

    def test_valid_development_evidence_keeps_release(self):
        self.assertEqual(self.check()['selected'],'combined')
        self.assertFalse(self.e['release_decision']['promoted'])

    def test_outer_fold_leaks_missing_models_and_scored_identity_rejected(self):
        for mutation in ('train_leak','held_out','fold_bool','regime','missing','scored_fold'):
            self.e=copy.deepcopy(self.evidence)
            meta=self.e['fold_models']['0']['precision']['metadata']
            if mutation=='train_leak':meta['training_matches'].append(verifier.FOLDS[0][0])
            elif mutation=='held_out':meta['held_out_matches']=[]
            elif mutation=='fold_bool':meta['fold']=False
            elif mutation=='regime':meta['training_regimes']=[]
            elif mutation=='missing':del self.e['fold_models']['2']
            else:self.e['records'][0]['fold']=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_sparse_support_scope_and_unsupported_precision_rejected(self):
        for mutation in ('fresh_match','dense_had_truth','unlisted_frames_are_negatives','precision','f1','support','recall','empty_error','tolerance'):
            self.e=copy.deepcopy(self.evidence)
            report=next(r['human_sparse_timing'] for r in self.e['records'] if 'human_sparse_timing' in r)
            value=report['restart']
            if mutation in ('fresh_match','dense_had_truth','unlisted_frames_are_negatives'):report[mutation]=True
            elif mutation in ('precision','f1'):value[mutation]=.99
            elif mutation=='support':value['n_reference_points']+=1
            elif mutation=='recall':value['recall_within_1s']=1.
            elif mutation=='empty_error':value['mean_absolute_error_matched_seconds']=0.
            else:value['tolerance_seconds']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_sparse_matched_error_bounds_and_unannotated_claim_rejected(self):
        for absolute,signed in ((float('nan'),0.),(.1,.2),(1.1,0.),(True,0.)):
            self.e=copy.deepcopy(self.evidence)
            value=next(r['human_sparse_timing']['restart'] for r in self.e['records'] if 'human_sparse_timing' in r)
            value.update(n_matched=1,recall_within_1s=1/value['n_reference_points'],
              mean_absolute_error_matched_seconds=absolute,mean_signed_error_matched_seconds=signed)
            with self.subTest(absolute=absolute,signed=signed),self.assertRaises(ValueError):self.check()
        self.e=copy.deepcopy(self.evidence)
        row=next(r for r in self.e['records'] if r['match'] not in MATCHES)
        row['human_sparse_timing']=sparse_report(MATCHES[0])
        with self.assertRaises(ValueError):self.check()

    def test_false_promotion_independent_truth_and_comparator_changes_rejected(self):
        for mutation in ('promoted','independent','competition','comparator','reproduced'):
            self.e=copy.deepcopy(self.evidence)
            if mutation=='promoted':self.e['release_decision']['promoted']=True
            elif mutation=='independent':self.e['independent_test_claimed']=True
            elif mutation=='competition':self.e['competition_labels_used']=True
            elif mutation=='reproduced':self.e['incumbent_historical_predictions_reproduced']=False
            else:next(r for r in self.e['records'] if r['candidate']==verifier.BASELINE)['prediction_sha256']='a'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_source_policy_selection_and_protocol_hash_tampering_rejected(self):
        for mutation in ('source','policy','selection','hash'):
            self.e=copy.deepcopy(self.evidence)
            if mutation=='source':self.e['source_hashes']['fake.py']='a'*64
            elif mutation=='policy':self.e['protocol']['combination_order'].reverse()
            elif mutation=='selection':self.e['selection']['selected']=verifier.BASELINE
            else:self.e['protocol_sha256']='a'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_refit_provenance_and_model_bytes_tampering_rejected(self):
        self.model_path.write_bytes(self.model_path.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.check()
        for mutation in ('training_matches','stage','selected'):
            self.e=copy.deepcopy(self.evidence);model=copy.deepcopy(self.model)
            model['stoppage']['metadata'][mutation]=[] if mutation=='training_matches' else 'wrong'
            write_json(self.model_path,model);self.e['experimental_model_sha256']=sha256(self.model_path)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_changed_active_release_or_annotation_detected_without_mutating_files(self):
        for target in ('models/submission.json','configs/default.json','submission/Xtension Labs.csv',
                       'research/annotations/elastic/J03WMX-points.json'):
            changed=verifier.ROOT/target
            def digest(path):return 'a'*64 if Path(path)==changed else sha256(path)
            with self.subTest(target=target),patch.object(verifier,'sha256',side_effect=digest),self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()

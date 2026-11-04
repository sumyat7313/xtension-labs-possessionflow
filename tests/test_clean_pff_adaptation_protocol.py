"""Synthetic receipt/gate tests; no provider files, labels, weights or fitting."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import clean_pff_adaptation_protocol as p


class AdaptationProtocolTests(unittest.TestCase):
    def setUp(self):
        support = dict(known_frames=1000, reference_live=600, reference_dead=400,
                       reference_dead_segments=20, reference_short_spells=20,
                       reference_short_frames=100)
        baseline = [dict(match_id=mid, condition='native', baseline_sha256=p.CLEAN_SHA256,
                         baseline_prediction_sha256=self.sha(f'baseline:{mid}'),
                         reference_sha256=self.sha(f'reference:{mid}'),
                         metrics={key: .5 for key in p.METRICS}, reference_support=copy.deepcopy(support))
                    for mid in p.INNER_VALIDATION_IDS]
        # Only the aggregate comparator identity is replaced, solely for this test.
        replacement = patch.object(p, 'BASELINE_RECORDS_SHA256', p.original.digest(baseline))
        replacement.start()
        self.addCleanup(replacement.stop)
        sources = {'tools/clean_pff_adaptation_protocol.py': self.sha('protocol'),
                   'tools/clean_pff_adaptation_fit.py': self.sha('kernel')}
        body = p._body(sources, baseline)
        self.protocol = dict(body, protocol_sha256=p.original.digest(body))
        self.candidates, self.predictions, self.records = {}, [], []
        for cid in p.CANDIDATES:
            receipt = dict(candidate_id=cid, model_sha256=self.sha('model:'+cid),
                protocol_sha256=self.protocol['protocol_sha256'], core_protocol_sha256=p.fixed_protocol_sha256(),
                phase='inner', parent_sha256=p.CLEAN_SHA256, config_sha256=p.CONFIG_SHA256,
                recipe_sha256=p.original.digest(p.recipe()), fitting_fixture_ids=list(p.fit_ids()),
                normalization_fixture_ids=list(p.fit_ids()), new_dead_fit_fixture_ids=list(p.fit_ids()) if cid=='B' else [],
                upstream_dead=self.protocol['candidates'][cid]['upstream_dead'], upstream_stacking='in-sample-fit-only')
            self.candidates[cid] = receipt
            for base in baseline:
                mid = base['match_id']; pred = self.sha(f'prediction:{cid}:{mid}')
                self.predictions.append(dict(candidate_id=cid, match_id=mid, prediction_sha256=pred))
                metrics = dict(base['metrics'], spell=.51)
                self.records.append(dict(candidate_id=cid, match_id=mid, condition='native',
                    protocol_sha256=self.protocol['protocol_sha256'], model_sha256=receipt['model_sha256'],
                    fit_receipt_sha256=p.original.digest(receipt), prediction_sha256=pred,
                    baseline_sha256=p.CLEAN_SHA256, baseline_prediction_sha256=base['baseline_prediction_sha256'],
                    reference_sha256=base['reference_sha256'], reference_support=copy.deepcopy(support), metrics=metrics,
                    short_diagnostics=dict(predicted_short_spells=10, matched_predicted_short_spells=5,
                        short_spells_with_dead=2, short_spells_erased_by_dead=1, short_false_dead_frames=10)))
        self.freeze = dict(schema_version=1, protocol_sha256=self.protocol['protocol_sha256'],
            model_sha256={cid:r['model_sha256'] for cid,r in self.candidates.items()},
            fit_receipt_sha256={cid:p.original.digest(r) for cid,r in self.candidates.items()},
            records=self.predictions, references_accessed=False, all_predictions_complete=True)

    @staticmethod
    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def score_receipt(self):
        return dict(schema_version=1, protocol_sha256=self.protocol['protocol_sha256'],
            prediction_freeze_sha256=p.original.digest(self.freeze), records_sha256=p.original.digest(self.records),
            scored_match_ids=list(p.INNER_VALIDATION_IDS), condition='native',
            reference_access_started_after_prediction_freeze=True, both_candidates_scored=True,
            development_accessed=False, final_accessed=False)

    def select(self, score_receipt=None):
        return p.select_candidate(self.records, self.protocol, candidates=self.candidates,
                                  prediction_freeze=self.freeze,
                                  score_receipt=self.score_receipt() if score_receipt is None else score_receipt)

    def a_rows(self):
        return [row for row in self.records if row['candidate_id']=='A']

    def rehash_protocol(self):
        body = {key:value for key,value in self.protocol.items() if key!='protocol_sha256'}
        self.protocol['protocol_sha256'] = p.original.digest(body)

    def test_tie_selects_a_but_never_authorizes_execution(self):
        result = self.select()
        self.assertEqual(result['selected_candidate_id'], 'A')
        self.assertEqual(result['ranking'], ['A','B'])
        for key in p.FLAGS:
            self.assertIs(result[key], False)

    def test_selects_best_qualifying_candidate_not_highest_failed_candidate(self):
        for row in self.a_rows():
            row['metrics'].update(spell=.7, dead_precision=.49)
        result = self.select()
        self.assertEqual(result['ranking'][0], 'A')
        self.assertFalse(result['candidates']['A']['passed'])
        self.assertEqual(result['selected_candidate_id'], 'B')

    def test_zero_spell_gain_never_qualifies(self):
        for row in self.records: row['metrics']['spell'] = .5
        self.assertIsNone(self.select()['selected_candidate_id'])

    def test_mean_stoppage_recall_regression_rejects_candidate(self):
        self.a_rows()[0]['metrics']['dead_recall'] = .499
        self.assertFalse(self.select()['candidates']['A']['passed'])

    def test_single_fixture_spell_guard_survives_positive_mean(self):
        self.a_rows()[0]['metrics']['spell'] = .489
        result = self.select()['candidates']['A']
        self.assertGreater(result['mean_gains']['spell'], 0)
        self.assertFalse(result['checks']['worst_fixture_spell'])

    def test_single_fixture_dead_recall_guard_survives_positive_mean(self):
        for row in self.a_rows(): row['metrics']['dead_recall'] = .51
        self.a_rows()[0]['metrics']['dead_recall'] = .494
        result = self.select()['candidates']['A']
        self.assertGreater(result['mean_gains']['dead_recall'], 0)
        self.assertFalse(result['checks']['worst_fixture_dead_recall'])

    def test_exact_per_fixture_guard_boundaries_are_inclusive(self):
        for row in self.a_rows(): row['metrics']['dead_recall'] = .51
        self.a_rows()[0]['metrics'].update(spell=.49, dead_recall=.495)
        result = self.select()['candidates']['A']
        self.assertTrue(result['checks']['worst_fixture_spell'])
        self.assertTrue(result['checks']['worst_fixture_dead_recall'])
        self.assertTrue(result['passed'])

    def test_lexicographic_ranking_uses_short_precision_after_spell(self):
        for row in self.records:
            if row['candidate_id']=='B':
                row['metrics']['short_precision']=.6
                row['short_diagnostics']['matched_predicted_short_spells']=6
        self.assertEqual(self.select()['selected_candidate_id'], 'B')

    def test_short_count_totals_include_unmatched_predictions(self):
        totals = self.select()['candidates']['A']['short_diagnostic_totals']
        self.assertEqual(totals['unmatched_predicted_short_spells'], 40)
        self.assertEqual(totals['short_spells_erased_by_dead'], 8)

    def test_missing_score_cannot_be_hidden_by_new_receipt(self):
        self.records.pop()
        with self.assertRaises(ValueError): self.select()

    def test_missing_candidate_cannot_be_selected_as_partial_family(self):
        del self.candidates['B']
        with self.assertRaises(ValueError): self.select()

    def test_identical_model_artifacts_are_rejected(self):
        self.candidates['B']['model_sha256']=self.candidates['A']['model_sha256']
        with self.assertRaisesRegex(ValueError, 'distinct'): self.select()

    def test_prediction_freeze_cannot_claim_reference_access(self):
        self.freeze['references_accessed']=True
        with self.assertRaises(ValueError): self.select()

    def test_boolean_schema_version_rejected(self):
        self.freeze['schema_version']=True
        with self.assertRaises(ValueError): self.select()

    def test_development_or_final_reference_access_is_rejected(self):
        for key in ('development_accessed','final_accessed'):
            receipt=self.score_receipt(); receipt[key]=True
            with self.subTest(key=key), self.assertRaises(ValueError): self.select(receipt)

    def test_claimed_scoring_before_prediction_freeze_rejected(self):
        receipt=self.score_receipt(); receipt['reference_access_started_after_prediction_freeze']=False
        with self.assertRaises(ValueError): self.select(receipt)

    def test_refit_roster_cannot_be_claimed_in_inner_receipt(self):
        self.candidates['A']['normalization_fixture_ids']=list(p.fit_ids('refit'))
        with self.assertRaises(ValueError): self.select()

    def test_contaminated_parent_identity_rejected(self):
        self.candidates['A']['parent_sha256']='3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0'
        with self.assertRaises(ValueError): self.select()

    def test_source_paths_cannot_escape_repository(self):
        for path in ('../private.json','C:/secret.py','/root/secret.py','tools\\other.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p._sources({'tools/clean_pff_adaptation_protocol.py':'a'*64,
                            'tools/clean_pff_adaptation_fit.py':'b'*64, path:'c'*64})

    def test_protocol_and_fit_kernel_source_identities_both_required(self):
        for missing in ('tools/clean_pff_adaptation_protocol.py','tools/clean_pff_adaptation_fit.py'):
            sources=dict(self.protocol['source_hashes']); del sources[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError): p._sources(sources)

    def test_protocol_permission_flag_cannot_be_enabled_even_after_rehash(self):
        self.protocol['final_access_authorized']=True; self.rehash_protocol()
        with self.assertRaises(ValueError): p.verify_protocol(self.protocol)

    def test_recipe_mutation_cannot_be_hidden_with_valid_digest(self):
        self.protocol['recipe']['epochs']=9; self.rehash_protocol()
        with self.assertRaises(ValueError): p.verify_protocol(self.protocol)

    def test_split_membership_cannot_be_mutated_even_after_rehash(self):
        self.protocol['inner_train_ids'][0]=p.FINAL_IDS[0]; self.rehash_protocol()
        with self.assertRaises(ValueError): p.verify_protocol(self.protocol)

    def test_zero_short_predictions_require_zero_short_precision(self):
        row=self.a_rows()[0]
        row['short_diagnostics'].update(predicted_short_spells=0,matched_predicted_short_spells=0)
        with self.assertRaises(ValueError): self.select()
        row['metrics']['short_precision']=0
        self.assertFalse(self.select()['candidates']['A']['passed'])

    def test_erased_short_spells_cannot_exceed_interrupted(self):
        self.a_rows()[0]['short_diagnostics']['short_spells_erased_by_dead']=3
        with self.assertRaises(ValueError): self.select()


class PublishedAggregateComparatorTests(unittest.TestCase):
    """Checks published aggregate identities only; no external record access."""
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.diagnosis=json.loads((root/'models/clean-pff-training-diagnosis-research.json').read_text(encoding='utf-8'))
        cls.reservation=json.loads((root/'models/pff-protocol.json').read_text(encoding='utf-8'))

    def test_exact_eight_full_model_comparators_match_published_diagnosis(self):
        rows=p.comparator_records(self.diagnosis)
        self.assertEqual(p.original.digest(rows), p.BASELINE_RECORDS_SHA256)
        self.assertEqual([row['match_id'] for row in rows], list(p.INNER_VALIDATION_IDS))
        for row in rows:
            source=next(record for record in self.diagnosis['records'] if record['match_id']==row['match_id'])
            detail=next(value for value in source['details'] if value['candidate']=='full')
            self.assertEqual(row['metrics'], source['metrics']['full'])
            self.assertEqual(row['baseline_prediction_sha256'], detail['prediction_sha256'])
            self.assertEqual(row['reference_sha256'], source['reference_support']['event_sha256'])
            self.assertEqual(row['reference_support']['known_frames'], source['diagnostics']['models']['full']['known_frames'])

    def test_modified_aggregate_evidence_is_rejected(self):
        diagnosis=copy.deepcopy(self.diagnosis)
        diagnosis['records'][0]['metrics']['full']['spell'] += .00001
        with self.assertRaises(ValueError): p.comparator_records(diagnosis)

    def test_build_and_verify_preserves_reservation_and_outer_gates(self):
        source={'tools/clean_pff_adaptation_protocol.py':'a'*64,'tools/clean_pff_adaptation_fit.py':'b'*64}
        study=p.build_protocol(self.reservation,self.diagnosis,source)
        self.assertEqual(p.verify_protocol(study),study)
        self.assertEqual(study['original_development_policy'],p.original.POLICY)
        self.assertEqual(study['additional_development_precision_margin'],.002)
        self.assertEqual(study['parent_gain_retention'],.9)
        self.assertEqual(set(p.INNER_TRAIN_IDS)|set(p.INNER_VALIDATION_IDS),set(self.reservation['split']['train']))
        self.assertFalse(set(p.fit_ids()) & {f'pff{mid}' for mid in p.EXCLUDED_INNER_IDS})


if __name__ == '__main__':
    unittest.main()

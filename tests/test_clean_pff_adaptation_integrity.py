"""Independent synthetic checks for clean adaptation; no provider inputs or fits."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools import clean_pff_adaptation_fit as kernel
from tools import clean_pff_adaptation_protocol as protocol


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Unexpected array materialization')


class AdaptationKernelIntegrityTests(unittest.TestCase):
    def record(self, match, number=0):
        n = 45 + 5*number
        x = np.full((n, 98), float(number+1))
        z = np.full((n, 58), float(number+1))
        # Nonsampled extremes must never enter newly fitted scalers or objectives.
        x[1::5] = 1e12; z[1::5] = 1e12
        y = np.resize(np.array(list('HADHAD')), n)
        y[0] = 'X'
        return dict(match=match, x=x, z=z, y=y, blocks=[[1, 0, n-1]], fps=25,
                    reference_fields_used_for_inference=False)

    def test_rejects_entire_bad_roster_and_refit_before_any_loader_call(self):
        roster = list(protocol.fit_ids('inner'))
        invalid = [roster[:-1], roster[::-1], roster+[roster[0]],
                   ['pff'+str(protocol.INNER_VALIDATION_IDS[0]), *roster[1:]],
                   ['pff3814', *roster[1:]], ['pff3813', *roster[1:]]]
        loader = Mock(side_effect=AssertionError('loader called'))
        for ids in invalid:
            with self.subTest(ids=ids[:2]), self.assertRaises(ValueError):
                kernel.fit_candidate('A', loader, fixture_ids=ids, parent_model={}, protocol_sha256='a'*64)
        with self.assertRaises(ValueError):
            kernel.fit_candidate('A', loader, fixture_ids=roster, parent_model={}, protocol_sha256='a'*64, phase='refit')
        loader.assert_not_called()

    def test_wrong_clean_parent_content_rejected_before_loader(self):
        loader = Mock(side_effect=AssertionError('loader called'))
        with self.assertRaisesRegex(ValueError, 'parent content'):
            kernel.fit_candidate('B', loader, fixture_ids=protocol.fit_ids('inner'), parent_model={},
                                 protocol_sha256=protocol.fixed_protocol_sha256())
        loader.assert_not_called()

    def test_record_identity_and_cached_upstream_rejected_before_arrays(self):
        match = protocol.fit_ids('inner')[0]
        record = self.record(match)
        record.update(x=ForbiddenArray(), z=ForbiddenArray(), y=ForbiddenArray())
        for changed in (dict(record, match='pff3814'), dict(record, base_dead=ForbiddenArray()),
                        dict(record, reference_fields_used_for_inference=True)):
            with self.assertRaises(ValueError):
                kernel.sample_record(changed, match, protocol.recipe())

    def test_equal_fixture_mass_known_rows_and_period_sampling(self):
        for number in (0, 1, 7):
            record = self.record(protocol.fit_ids('inner')[number], number)
            before = copy.deepcopy(record)
            sequences, support = kernel.sample_record(record, record['match'], protocol.recipe())
            self.assertAlmostEqual(sum(s['w'].sum() for s in sequences), 10000., places=8)
            self.assertAlmostEqual(support['sampled_weight'], 10000., places=8)
            for s in sequences:
                self.assertTrue(np.all(s['w'][s['y'] == 'X'] == 0))
                self.assertLess(float(np.max(s['X'])), 1e12)
                self.assertLess(float(np.max(s['Z'])), 1e12)
            for key in ('x', 'z', 'y'):
                np.testing.assert_array_equal(record[key], before[key])
        record = self.record(protocol.fit_ids('inner')[0])
        record['blocks'] = [[1, 2, 16], [2, 25, 39]]
        record['x'][:, 0] = np.arange(len(record['y']))
        seq, _ = kernel.sample_record(record, record['match'], protocol.recipe())
        np.testing.assert_array_equal(seq[0]['X'][:, 0], [2, 7, 12])
        np.testing.assert_array_equal(seq[1]['X'][:, 0], [25, 30, 35])

    def test_a_b_differ_only_declared_dead_refit_and_use_new_joint(self):
        roster = list(protocol.fit_ids('inner'))
        parent = dict(mean=[0.]*58, scale=[1.]*58, coefficients=[0.]*58, intercept=0.,
                      robust_control_model={'identity': 'fixed clean H/A'}, joint_model={'old': 'must be removed'},
                      decoder_mode='joint', metadata={'parent': 'historically exposed Sportec'})
        original = copy.deepcopy(parent)
        outputs, captured = {}, {}
        for candidate in ('A', 'B'):
            calls, fitted = [], {}
            def loader(match):
                calls.append(match)
                return self.record(match, roster.index(match))
            def dead_fit(z, y, **kw):
                fitted['dead'] = (z.copy(), y.copy(), copy.deepcopy(kw))
                return dict(mean=[2.]*58, scale=[1.]*58, coefficients=[0.]*58, intercept=2., training={'synthetic': True})
            def dead_predict(z, model):
                self.assertNotIn('joint_model', model)
                return np.full(len(z), .7 if model['intercept'] == 2. else .2)
            def joint_fit(sequences, **kw):
                fitted['joint'] = (copy.deepcopy(sequences), copy.deepcopy(kw))
                return {'training': {'synthetic': True}, 'new': True}
            with patch.object(kernel, 'validate_parent'), patch.object(kernel.learned, 'validate_model'), \
                 patch.object(kernel.learned, 'fit_model', side_effect=dead_fit) as df, \
                 patch.object(kernel.learned, 'predict_dead', side_effect=dead_predict), \
                 patch.object(kernel.robust_control, 'predict_home', side_effect=lambda model,x: np.full(len(x), .6)), \
                 patch.object(kernel.train_joint, 'fit', side_effect=joint_fit):
                outputs[candidate] = kernel.fit_candidate(candidate, loader, fixture_ids=roster,
                    parent_model=parent, protocol_sha256=protocol.fixed_protocol_sha256())
                self.assertEqual(df.call_count, int(candidate == 'B'))
            self.assertEqual(calls, roster)
            self.assertEqual(parent, original)
            model, report = outputs[candidate]
            self.assertEqual(model['robust_control_model'], parent['robust_control_model'])
            self.assertEqual(report['upstream_stacking'], 'in-sample-fit-only')
            self.assertFalse(report['cache_bytes_authenticated_by_kernel'])
            self.assertFalse(report['refit_authorized'])
            self.assertFalse(report['development_accessed'])
            self.assertFalse(report['final_accessed'])
            self.assertFalse(report['promoted'])
            seq, params = fitted['joint']
            self.assertEqual(params['seed'], protocol.recipe()['seed'])
            self.assertEqual(params['epochs'], protocol.recipe()['epochs'])
            expected_dead = .2 if candidate == 'A' else .7
            for s in seq:
                np.testing.assert_allclose(s['base_dead'], np.log(expected_dead/(1-expected_dead)))
                np.testing.assert_allclose(s['base_home'], np.log(.6/.4))
                self.assertTrue(np.all(s['w'][s['y'] == 'X'] == 0))
            captured[candidate] = fitted
        a, b = captured['A']['joint'][0], captured['B']['joint'][0]
        self.assertEqual(len(a), len(b))
        for left, right in zip(a,b):
            for key in ('X', 'Z', 'y', 'w', 'base_home'):
                np.testing.assert_array_equal(left[key], right[key])
        z,y,kw = captured['B']['dead']
        np.testing.assert_array_equal(z, np.concatenate([s['Z'] for s in b]))
        np.testing.assert_array_equal(y, np.concatenate([s['y'] for s in b]))
        np.testing.assert_array_equal(kw['sample_weight'], np.concatenate([s['w'] for s in b]))
        self.assertTrue(np.all(kw['sample_weight'][y == 'X'] == 0))
        self.assertLess(z.max(), 1e12)

    def test_boundary_loss_respects_period_barriers_and_unknown_rows(self):
        match = protocol.fit_ids('inner')[0]
        record = self.record(match)
        record.update(x=np.zeros((200, 98)), z=np.zeros((200, 58)),
                      y=np.array(list('D'*100+'H'*100)), blocks=[[1, 0, 99], [2, 100, 199]])
        split, _ = kernel.sample_record(record, match, protocol.recipe())
        # A half boundary is not a training-label transition. No radius boost.
        np.testing.assert_allclose(split[0]['w'], split[0]['w'][0])
        np.testing.assert_allclose(split[1]['w'], split[1]['w'][0])
        self.assertAlmostEqual(split[0]['w'][0], split[1]['w'][0])
        record['blocks'] = [[1, 0, 199]]
        joined, _ = kernel.sample_record(record, match, protocol.recipe())
        weights = joined[0]['w']
        self.assertAlmostEqual(weights[20]/weights[0], 1.5)
        self.assertAlmostEqual(weights[18]/weights[0], 1.5)
        self.assertAlmostEqual(weights[22]/weights[0], 1.5)
        self.assertAlmostEqual(weights[17]/weights[0], 1.)
        record['y'][100] = 'X'
        unknown, _ = kernel.sample_record(record, match, protocol.recipe())
        self.assertEqual(unknown[0]['w'][20], 0.)

    def test_recomputed_core_hash_cannot_authorize_wrong_digest(self):
        loader = Mock(side_effect=AssertionError('loader called'))
        for identity in ('a'*64, True, None):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, 'scientific-core'):
                kernel.fit_candidate('A', loader, fixture_ids=protocol.fit_ids('inner'),
                                     parent_model={}, protocol_sha256=identity)
        loader.assert_not_called()


class AdaptationSelectorIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.reservation = json.loads((root/'models/pff-protocol.json').read_text(encoding='utf-8'))
        cls.diagnosis = json.loads((root/'models/clean-pff-training-diagnosis-research.json').read_text(encoding='utf-8'))

    def fixture(self):
        study = protocol.build_protocol(self.reservation, self.diagnosis,
            {'tools/clean_pff_adaptation_protocol.py': '1'*64,
             'tools/clean_pff_adaptation_fit.py': '2'*64})
        digest = protocol.original.digest
        candidates, predictions, records = {}, [], []
        for cid in ('A','B'):
            receipt = dict(candidate_id=cid, model_sha256=('a' if cid=='A' else 'b')*64,
                protocol_sha256=study['protocol_sha256'], phase='inner', parent_sha256=protocol.CLEAN_SHA256,
                core_protocol_sha256=protocol.fixed_protocol_sha256(),
                config_sha256=protocol.CONFIG_SHA256, recipe_sha256=digest(protocol.recipe()),
                fitting_fixture_ids=list(protocol.fit_ids()), normalization_fixture_ids=list(protocol.fit_ids()),
                new_dead_fit_fixture_ids=list(protocol.fit_ids()) if cid=='B' else [],
                upstream_dead=study['candidates'][cid]['upstream_dead'], upstream_stacking='in-sample-fit-only')
            candidates[cid] = receipt
            for baseline in study['baseline_records']:
                mid = baseline['match_id']
                previous = next(r for r in self.diagnosis['records'] if r['match_id']==mid)
                detail = next(r for r in previous['details'] if r['candidate']=='full')
                bins = detail['duration_precision_recall']['bins']
                short = previous['diagnostics']['short_false_dead']['models']['full']
                prediction_sha = hashlib.sha256(f'synthetic:{cid}:{mid}'.encode()).hexdigest()
                predictions.append(dict(candidate_id=cid, match_id=mid, prediction_sha256=prediction_sha))
                values = copy.deepcopy(baseline['metrics']); values['spell'] += .001
                records.append(dict(candidate_id=cid, match_id=mid, condition='native',
                    protocol_sha256=study['protocol_sha256'], model_sha256=receipt['model_sha256'],
                    fit_receipt_sha256=digest(receipt), prediction_sha256=prediction_sha,
                    baseline_sha256=protocol.CLEAN_SHA256, baseline_prediction_sha256=baseline['baseline_prediction_sha256'],
                    reference_sha256=baseline['reference_sha256'], reference_support=copy.deepcopy(baseline['reference_support']),
                    metrics=values, short_diagnostics=dict(
                        predicted_short_spells=sum(bins[k]['n_pred'] for k in ('<0.4s','0.4-1s','1-2s')),
                        matched_predicted_short_spells=sum(bins[k]['n_matched_pred'] for k in ('<0.4s','0.4-1s','1-2s')),
                        short_spells_with_dead=short['spells_with_any_dead'],
                        short_spells_erased_by_dead=short['spells_erased_by_dead'],
                        short_false_dead_frames=short['false_dead_frames'])))
        freeze = dict(schema_version=1, protocol_sha256=study['protocol_sha256'],
            model_sha256={cid:r['model_sha256'] for cid,r in candidates.items()},
            fit_receipt_sha256={cid:digest(r) for cid,r in candidates.items()}, records=predictions,
            references_accessed=False, all_predictions_complete=True)
        return study, candidates, freeze, records

    def select(self, fixture, receipt=None):
        study,candidates,freeze,records = fixture
        if receipt is None: receipt = self.receipt(fixture)
        return protocol.select_candidate(records, study, candidates=candidates,
            prediction_freeze=freeze, score_receipt=receipt)

    def receipt(self, fixture):
        study,_,freeze,records = fixture
        return dict(schema_version=1, protocol_sha256=study['protocol_sha256'],
            prediction_freeze_sha256=protocol.original.digest(freeze), records_sha256=protocol.original.digest(records),
            scored_match_ids=list(protocol.INNER_VALIDATION_IDS), condition='native',
            reference_access_started_after_prediction_freeze=True, both_candidates_scored=True,
            development_accessed=False, final_accessed=False)

    def test_split_is_label_independent_32_8_and_original_gates_unchanged(self):
        ranking = sorted(self.reservation['split']['train'],
                         key=lambda mid: hashlib.sha256(f'{protocol.SPLIT_SEED}:{mid}'.encode()).hexdigest())
        self.assertEqual(tuple(sorted(ranking[:8])), protocol.INNER_VALIDATION_IDS)
        self.assertEqual(tuple(sorted(ranking[8:])), protocol.INNER_TRAIN_IDS)
        self.assertFalse(set(protocol.INNER_VALIDATION_IDS) & set(protocol.INNER_TRAIN_IDS))
        study = self.fixture()[0]
        self.assertEqual(study['original_development_policy'], protocol.original.POLICY)
        self.assertEqual(study['additional_development_precision_margin'], .002)
        self.assertEqual(study['parent_gain_retention'], .9)

    def test_passing_declared_receipts_do_not_authorize_execution_or_refit(self):
        result = self.select(self.fixture())
        self.assertEqual(result['selected_candidate_id'], 'A')
        self.assertTrue(result['inner_numerical_passed'])
        for key in ('execution_evidence_verified','campaign_ready','refit_authorized',
                    'development_access_authorized','final_access_authorized','promoted','submission_eligible'):
            self.assertIs(result[key], False)

    def test_fixed_recipe_cannot_be_relaxed_even_with_recomputed_digest(self):
        for path in ('recipe','inner_gate'):
            study = self.fixture()[0]
            if path=='recipe': study[path]['epochs'] += 1
            else: study[path]['maximum_fixture_spell_drop'] = .1
            body = {k:v for k,v in study.items() if k!='protocol_sha256'}
            study['protocol_sha256'] = protocol.original.digest(body)
            with self.assertRaises(ValueError): protocol.verify_protocol(study)

    def test_valid_metric_mutation_requires_new_record_receipt(self):
        fixture = self.fixture(); receipt = self.receipt(fixture)
        fixture[3][0]['metrics']['spell'] += .01
        with self.assertRaises(ValueError): self.select(fixture,receipt)

    def test_missing_duplicate_or_heldout_predictions_fail_even_after_rehash(self):
        for variant in ('missing','duplicate','heldout'):
            fixture=self.fixture(); rows=fixture[2]['records']
            if variant=='missing': rows.pop()
            elif variant=='duplicate': rows[-1]=copy.deepcopy(rows[0])
            else: rows[-1]['match_id']=3814
            with self.subTest(variant=variant), self.assertRaises(ValueError): self.select(fixture)

    def test_nonfinite_boolean_metrics_and_corrupt_support_rejected(self):
        for value in (float('nan'), float('inf'), True, -.1, 1.1):
            fixture=self.fixture(); fixture[3][0]['metrics']['spell']=value
            with self.subTest(value=value), self.assertRaises(ValueError): self.select(fixture)
        fixture=self.fixture(); fixture[3][0]['reference_support']['reference_live'] += 1
        with self.assertRaises(ValueError): self.select(fixture)

    def test_short_precision_requires_matching_count_evidence(self):
        fixture=self.fixture(); row=fixture[3][0]
        row['metrics']['short_precision']=.99
        row['short_diagnostics']['predicted_short_spells']=100
        row['short_diagnostics']['matched_predicted_short_spells']=0
        with self.assertRaises(ValueError): self.select(fixture)

    def test_interrupted_short_spells_need_false_dead_frame_support(self):
        fixture=self.fixture(); row=fixture[3][0]
        row['short_diagnostics'].update(short_spells_with_dead=1,short_spells_erased_by_dead=0,short_false_dead_frames=0)
        with self.assertRaises(ValueError): self.select(fixture)

    def test_negative_family_stops_without_selecting_runner_up_outside_ab(self):
        fixture=self.fixture()
        for row in fixture[3]: row['metrics']['spell'] -= .002
        result=self.select(fixture)
        self.assertIsNone(result['selected_candidate_id'])
        self.assertFalse(result['inner_numerical_passed'])


if __name__ == '__main__': unittest.main()

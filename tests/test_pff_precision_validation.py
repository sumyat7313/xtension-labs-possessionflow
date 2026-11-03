import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import pff_validation as original
from tools import pff_precision_validation as precision


def reservation():
    rows = [{'match_id': i, 'date': '2022-11-21' if i < 10000 else '2022-12-18',
             'home_team': f'Team{i}', 'away_team': 'Opponent',
             'stage': 'group' if i < 10000 else 'knockout',
             'competition': 'FIFA World Cup 2022'} for i in original.EXPECTED_IDS]
    return original.build_protocol(rows, {str(i): 'a' * 64 for i in original.EXPECTED_IDS}, 'b' * 64)


def rows(p, sha=precision.PARENT_SHA256, part='development', gain=.02, dead_precision=-.006):
    result = []
    for mid in p['split'][part]:
        values = dict.fromkeys(original.METRICS, .7 + gain)
        values['dead_precision'] = .7 + dead_precision
        result.append({'match_id': mid, 'condition': 'native', 'baseline_sha256': original.BASELINE_SHA256,
                       'candidate_sha256': sha, 'baseline': dict.fromkeys(original.METRICS, .7),
                       'candidate': values, 'reference_dead_segments': 100, 'reference_short_spells': 100})
    return result


def fixture():
    p = reservation(); prior = rows(p)
    candidates = [{'candidate_id': 'p' + str(i), 'artifact_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                   'parameters': {'dead_logit_penalty': penalty}} for i, penalty in enumerate(precision.PENALTIES)]
    study = precision.build_protocol(p, prior, candidates, {'research.py': 'e' * 64})
    scores = {c['candidate_id']: rows(p, c['artifact_sha256'], gain=.025, dead_precision=.003)
              for c in candidates}
    scores['p0'] = rows(p, candidates[0]['artifact_sha256'])
    return p, prior, study, scores


class PrecisionValidationTests(unittest.TestCase):
    def test_passing_development_is_never_promotion(self):
        p, prior, study, scores = fixture()
        gate = precision.development_gate(scores['p1'], prior, p)
        self.assertTrue(gate['eligible_for_final_audit'])
        self.assertFalse(gate['promoted'])
        self.assertTrue(all(gate['retention_checks'].values()))

    def test_original_failed_precision_is_not_passing(self):
        p, prior, study, scores = fixture()
        self.assertFalse(precision.development_gate(scores['p0'], prior, p)['eligible_for_final_audit'])

    def test_precision_margin_required_beyond_nonregression(self):
        p, prior, study, scores = fixture()
        for r in scores['p1']:
            r['candidate']['dead_precision'] = .701
        gate = precision.development_gate(scores['p1'], prior, p)
        self.assertTrue(gate['checks']['no_target_regression'])
        self.assertFalse(gate['checks']['dead_precision_margin'])
        self.assertFalse(gate['eligible_for_final_audit'])

    def test_every_parent_gain_must_be_retained(self):
        p, prior, study, scores = fixture()
        for key in precision.RETAIN_METRICS:
            with self.subTest(metric=key):
                altered = copy.deepcopy(scores['p1'])
                for r in altered:
                    r['candidate'][key] = .717
                gate = precision.development_gate(altered, prior, p)
                self.assertFalse(gate['retention_checks'][key])
                self.assertFalse(gate['eligible_for_final_audit'])

    def test_original_worst_fixture_guard_survives_good_mean(self):
        p, prior, study, scores = fixture()
        scores['p1'][0]['candidate']['spell'] = .68
        gate = precision.development_gate(scores['p1'], prior, p)
        self.assertTrue(gate['checks']['retain_parent_gains'])
        self.assertFalse(gate['checks']['worst_fixture_spell'])
        self.assertFalse(gate['eligible_for_final_audit'])

    def test_comparator_and_support_must_be_paired(self):
        p, prior, study, scores = fixture()
        for field in ('reference_dead_segments', 'reference_short_spells', 'baseline'):
            with self.subTest(field=field):
                altered = copy.deepcopy(scores['p1'])
                if field == 'baseline':
                    altered[0][field]['macro'] = .699
                else:
                    altered[0][field] += 1
                with self.assertRaises(ValueError):
                    precision.development_gate(altered, prior, p)

    def test_wrong_parent_and_wrong_partition_rejected(self):
        p, prior, study, scores = fixture()
        wrong = rows(p, 'd' * 64)
        with self.assertRaises(ValueError):
            precision.development_gate(scores['p1'], wrong, p)
        with self.assertRaises(ValueError):
            precision.development_gate(rows(p, part='final'), prior, p)

    def test_grid_is_fixed_ordered_and_hashed(self):
        p, prior, study, scores = fixture()
        for kind in ('order', 'value', 'repeat_id', 'repeat_sha', 'bool', 'extra_parameter'):
            with self.subTest(kind=kind):
                candidates = copy.deepcopy(study['candidates'])
                if kind == 'order': candidates.reverse()
                elif kind == 'value': candidates[1]['parameters']['dead_logit_penalty'] = .2
                elif kind == 'repeat_id': candidates[1]['candidate_id'] = candidates[0]['candidate_id']
                elif kind == 'repeat_sha': candidates[1]['artifact_sha256'] = candidates[0]['artifact_sha256']
                elif kind == 'bool': candidates[0]['parameters']['dead_logit_penalty'] = False
                else: candidates[0]['parameters']['hidden_threshold'] = .7
                with self.assertRaises(ValueError):
                    precision.build_protocol(p, prior, candidates, study['source_hashes'])

    def test_protocol_cannot_be_rewritten_or_mutated_by_alias(self):
        p, prior, study, scores = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'protocol.json'
            precision.freeze_protocol(path, study, p, prior)
            precision.freeze_protocol(path, study, p, prior)
            changed = precision.build_protocol(p, prior, study['candidates'], {'other.py': 'f' * 64})
            with self.assertRaises(ValueError):
                precision.freeze_protocol(path, changed, p, prior)
        changed = copy.deepcopy(study); changed['policy']['minimum_dead_precision_gain'] = 0
        with self.assertRaises(ValueError):
            precision.verify_protocol(changed, p, prior)

    def test_selection_requires_full_grid_and_exact_artifacts(self):
        p, prior, study, scores = fixture()
        missing = dict(scores); del missing['p7']
        with self.assertRaises(ValueError): precision.select_candidate(study, missing, prior, p)
        extra = dict(scores, rogue=scores['p1'])
        with self.assertRaises(ValueError): precision.select_candidate(study, extra, prior, p)
        altered = copy.deepcopy(scores)
        for r in altered['p1']: r['candidate_sha256'] = 'f' * 64
        with self.assertRaises(ValueError): precision.select_candidate(study, altered, prior, p)

    def test_zero_control_must_reproduce_parent(self):
        p, prior, study, scores = fixture()
        scores['p0'][0]['candidate']['spell'] += .001
        with self.assertRaises(ValueError): precision.select_candidate(study, scores, prior, p)

    def test_deterministic_ranking_and_ties(self):
        p, prior, study, scores = fixture()
        self.assertEqual(precision.select_candidate(study, scores, prior, p)['selected_candidate_id'], 'p1')
        for r in scores['p2']: r['candidate']['dead_boundary'] += .001
        self.assertEqual(precision.select_candidate(study, scores, prior, p)['selected_candidate_id'], 'p2')
        for r in scores['p3']: r['candidate']['spell'] += .001
        self.assertEqual(precision.select_candidate(study, dict(reversed(list(scores.items()))), prior, p)['selected_candidate_id'], 'p3')

    def test_no_passing_candidate_keeps_final_locked(self):
        p, prior, study, scores = fixture()
        for records in scores.values():
            for r in records: r['candidate']['dead_precision'] = .694
        selection = precision.select_candidate(study, scores, prior, p)
        self.assertIsNone(selection['selected_candidate_id'])
        self.assertFalse(selection['eligible_for_final_audit'])
        with self.assertRaises(ValueError):
            precision.require_final_candidate(study, scores, prior, p, study['candidates'][1]['artifact_sha256'])

    def test_final_guard_recomputes_selected_identity(self):
        p, prior, study, scores = fixture()
        sha = study['candidates'][1]['artifact_sha256']
        selection = precision.require_final_candidate(study, scores, prior, p, sha)
        self.assertEqual(selection['selected_candidate_sha256'], sha)
        with self.assertRaises(ValueError): precision.require_final_candidate(study, scores, prior, p, 'f' * 64)

    def test_final_gate_requires_retention_selected_identity_and_evidence(self):
        p, prior, study, scores = fixture()
        selection = precision.select_candidate(study, scores, prior, p)
        final = rows(p, selection['selected_candidate_sha256'], part='final', gain=.025, dead_precision=.003)
        parent = rows(p, part='final')
        evidence = dict.fromkeys(original.EVIDENCE, True)
        self.assertTrue(precision.final_gate(final, parent, p, evidence, selection)['promoted'])
        for flag in original.EVIDENCE:
            with self.subTest(flag=flag):
                self.assertFalse(precision.final_gate(final, parent, p, dict(evidence, **{flag:False}), selection)['promoted'])
        wrong = dict(selection, selected_candidate_sha256='f' * 64)
        with self.assertRaises(ValueError): precision.final_gate(final, parent, p, evidence, wrong)
        for r in final: r['candidate']['short_precision'] = .717
        self.assertFalse(precision.final_gate(final, parent, p, evidence, selection)['promoted'])


if __name__ == '__main__':
    unittest.main()

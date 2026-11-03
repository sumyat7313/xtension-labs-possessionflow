import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import pff_context_validation as context
from tools import pff_precision_validation as precision
from tools import pff_validation as original
from tools import pff_context_model as model


def fixture():
    metadata = [{'match_id': i, 'date': '2022-11-21' if i < 10000 else '2022-12-18',
                 'home_team': f'Team{i}', 'away_team': 'Opponent',
                 'stage': 'group' if i < 10000 else 'knockout',
                 'competition': 'FIFA World Cup 2022'} for i in original.EXPECTED_IDS]
    p = original.build_protocol(metadata, {str(i): 'a' * 64 for i in original.EXPECTED_IDS}, 'b' * 64)
    prior = []
    for mid in p['split']['development']:
        values = dict.fromkeys(original.METRICS, .72); values['dead_precision'] = .694
        prior.append({'match_id': mid, 'condition': 'native', 'baseline_sha256': original.BASELINE_SHA256,
                      'candidate_sha256': context.PARENT_SHA256, 'baseline': dict.fromkeys(original.METRICS, .7),
                      'candidate': values, 'reference_dead_segments': 100, 'reference_short_spells': 100})
    declared = [{'candidate_id': f'blend-{i}', 'artifact_sha256': hashlib.sha256(f'blend-{i}'.encode()).hexdigest(),
                 'parameters': {'blend': blend}} for i, blend in enumerate(context.BLENDS)]
    study = context.build_protocol(p, prior, declared, {'test.py': 'e' * 64},
                                   head_sha256='f' * 64, training_matches=p['split']['train'], fit_policy=model.POLICY)
    scores = {}
    for i, declaration in enumerate(declared):
        rows = copy.deepcopy(prior)
        for row in rows:
            row['candidate_sha256'] = declaration['artifact_sha256']
            if i: row['candidate'] = dict.fromkeys(original.METRICS, .725) | {'dead_precision': .703}
        scores[declaration['candidate_id']] = rows
    return p, prior, study, scores


class ContextValidationTests(unittest.TestCase):
    def test_gates_are_the_unchanged_original_precision_gates(self):
        self.assertIs(context.development_gate, precision.development_gate)
        self.assertIs(context.final_gate, precision.final_gate)

    def test_protocol_binds_head_exact_train_membership_and_policy(self):
        p, prior, study, scores = fixture()
        for kind in ('head', 'missing', 'duplicate', 'heldout', 'bool_id', 'policy'):
            with self.subTest(kind=kind):
                head = 'f' * 64; train = p['split']['train'].copy(); policy = copy.deepcopy(model.POLICY)
                if kind == 'head': head = 'unknown'
                elif kind == 'missing': train.pop()
                elif kind == 'duplicate': train[1] = train[0]
                elif kind == 'heldout': train[0] = p['split']['final'][0]
                elif kind == 'bool_id': train[0] = True
                else: policy['epochs'] = 99
                with self.assertRaises(ValueError):
                    context.build_protocol(p, prior, study['candidates'], study['source_hashes'],
                                           head_sha256=head, training_matches=train, fit_policy=policy)

    def test_exact_grid_rejects_hidden_parameters_and_duplicate_artifacts(self):
        p, prior, study, scores = fixture()
        for kind in ('order', 'blend', 'bool', 'missing', 'extra', 'duplicate_id', 'duplicate_sha'):
            with self.subTest(kind=kind):
                candidates = copy.deepcopy(study['candidates'])
                if kind == 'order': candidates.reverse()
                elif kind == 'blend': candidates[1]['parameters']['blend'] = .2
                elif kind == 'bool': candidates[0]['parameters']['blend'] = False
                elif kind == 'missing': candidates.pop()
                elif kind == 'extra': candidates[0]['parameters']['threshold'] = .8
                elif kind == 'duplicate_id': candidates[1]['candidate_id'] = candidates[0]['candidate_id']
                else: candidates[1]['artifact_sha256'] = candidates[0]['artifact_sha256']
                with self.assertRaises(ValueError):
                    context.build_protocol(p, prior, candidates, study['source_hashes'],
                                           head_sha256=study['head_sha256'], training_matches=study['training_matches'], fit_policy=model.POLICY)

    def test_immutable_protocol_and_parent_anchor(self):
        p, prior, study, scores = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'study.json'
            context.freeze_protocol(path, study, p, prior); context.freeze_protocol(path, study, p, prior)
            changed = context.build_protocol(p, prior, study['candidates'], study['source_hashes'],
                                             head_sha256='a' * 64, training_matches=study['training_matches'], fit_policy=model.POLICY)
            with self.assertRaises(ValueError): context.freeze_protocol(path, changed, p, prior)
        wrong = copy.deepcopy(prior)
        for row in wrong: row['candidate_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): context.verify_protocol(study, p, wrong)

    def test_complete_grid_and_artifact_identity_required(self):
        p, prior, study, scores = fixture()
        altered = dict(scores); altered.pop('blend-5')
        with self.assertRaises(ValueError): context.select_candidate(study, altered, prior, p)
        altered = dict(scores, rogue=scores['blend-1'])
        with self.assertRaises(ValueError): context.select_candidate(study, altered, prior, p)
        for row in scores['blend-1']: row['candidate_sha256'] = 'd' * 64
        with self.assertRaises(ValueError): context.select_candidate(study, scores, prior, p)

    def test_zero_control_and_deterministic_winner(self):
        p, prior, study, scores = fixture()
        self.assertEqual(context.select_candidate(study, scores, prior, p)['selected_candidate_id'], 'blend-1')
        for row in scores['blend-2']: row['candidate']['dead_boundary'] += .001
        self.assertEqual(context.select_candidate(study, scores, prior, p)['selected_candidate_id'], 'blend-2')
        for row in scores['blend-3']: row['candidate']['spell'] += .001
        self.assertEqual(context.select_candidate(study, dict(reversed(list(scores.items()))), prior, p)['selected_candidate_id'], 'blend-3')
        scores['blend-0'][0]['candidate']['macro'] += .001
        with self.assertRaises(ValueError): context.select_candidate(study, scores, prior, p)

    def test_retention_and_precision_failures_keep_final_locked(self):
        p, prior, study, scores = fixture()
        for ident, records in scores.items():
            if ident != 'blend-0':
                for row in records: row['candidate']['short_precision'] = .717
        selection = context.select_candidate(study, scores, prior, p)
        self.assertFalse(selection['eligible_for_final_audit'])
        self.assertFalse(selection['promoted']); self.assertIsNone(selection['selected_candidate_id'])
        with self.assertRaises(ValueError):
            context.require_final_candidate(study, scores, prior, p, study['candidates'][1]['artifact_sha256'])

    def test_final_guard_recomputes_exact_context_winner(self):
        p, prior, study, scores = fixture()
        sha = study['candidates'][1]['artifact_sha256']
        self.assertEqual(context.require_final_candidate(study, scores, prior, p, sha)['selected_candidate_sha256'], sha)
        with self.assertRaises(ValueError): context.require_final_candidate(study, scores, prior, p, 'a' * 64)


if __name__ == '__main__':
    unittest.main()

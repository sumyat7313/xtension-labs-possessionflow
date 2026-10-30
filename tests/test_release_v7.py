"""V7 cannot promote with phase-only evidence or unsupported contact claims."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import verify_release as verify, fresh_benchmark_v7 as fresh, train_boundary_campaign_v7 as campaign
from possession import typed_boundary as typed, contact_evidence
from tools.train_control import CATALOG

CANDIDATE = 'a' * 64


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def refresh(wrapper):
    digest = hashlib.sha256(encoded(wrapper['audit']['protocol'])).hexdigest()
    wrapper['audit_protocol_sha256'] = digest
    wrapper['access_marker']['protocol_sha256'] = digest


def audit():
    split, policy = encoded(fresh.SPLIT), encoded(fresh.POLICY)
    value = dict(metric='phase_present_team_accuracy', reference_quality='provider_silver', n=100,
                 accuracy=.8, balanced_team_recall=.8, predicted_live_coverage=.96,
                 predicted_dead_fraction=.04, predicted_unknown_fraction=0.,
                 dense_possession_truth=False, dense_had_truth=False, spellF1=None)
    scores = {f'skillcorner{m}': {k: copy.deepcopy(value) for k in ('baseline', 'candidate')} for m in fresh.SC}
    protocol = dict(schema_version=1, selection_frozen_before_reference_access=True, no_training_or_retuning=True,
                    dense_human_had_validation=False, fresh_stoppage_validation_available=False,
                    split=copy.deepcopy(fresh.SPLIT), split_sha256=hashlib.sha256(split).hexdigest(),
                    policy=copy.deepcopy(fresh.POLICY), policy_sha256=hashlib.sha256(policy).hexdigest(),
                    sources=fresh.scientific_hashes(),
                    data={f'{m}_{suffix}': identity if len(identity) == 64 else 'b' * 64
                          for m, files in fresh.SC.items() for suffix, (_, identity) in files.items()},
                    artifacts={k: dict(model_sha256=sha, config_sha256=campaign.CONFIG_SHA256)
                               for k, sha in (('baseline', verify.V15_MODEL_SHA256), ('candidate', CANDIDATE))})
    predictions = {f'{m}-{k}.csv': 'c' * 64 for m in scores for k in ('baseline', 'candidate')}
    result = dict(protocol=protocol, scores=scores, gate=fresh.decide(scores), prediction_sha256=predictions,
                  reference_counts={m: dict(gap_frames_are_unknown=True, dead_reference_frames=0, scored_frames=100) for m in scores},
                  reference_fields_used_for_inference=False, dense_human_had_validation=False,
                  fresh_stoppage_validation_available=False, reference_read_count=2)
    wrapper = dict(policy=copy.deepcopy(fresh.POLICY), passed=False, phase_regression_passed=True,
                   baseline_model_sha256=verify.V15_MODEL_SHA256, candidate_model_sha256=CANDIDATE,
                   split_file_base64=base64.b64encode(split).decode(), policy_file_base64=base64.b64encode(policy).decode(),
                   access_marker=dict(status='access_started', audit_directory='runs/synthetic'),
                   predictions_before_reference_access=copy.deepcopy(predictions), audit=result)
    refresh(wrapper)
    return wrapper


def evidence():
    rows = []
    for name in (campaign.BASELINE, *campaign.CANDIDATES):
        for fold in campaign.FOLDS:
            for match in fold:
                for regime in campaign.REGIMES:
                    rows.append(dict(candidate=name, match=match, regime=regime, spellF1=.5, macroF1=.8,
                                     dead_recall=.94, dead_precision=.9, temporal_diagnostics={
                                         'dead_segments': {'f1': .8}, 'dead_onsets': {'f1': .8}, 'dead_offsets': {'f1': .8},
                                         'short_possessions': {'recall': .2}}, duration_precision_recall={
                                             'bins': {k: {'n_pred': 10, 'n_matched_pred': 5} for k in ('<0.4s', '0.4-1s', '1-2s')}}))
    selection = campaign.select(rows)
    protocol = dict(schema_version=7, source_hashes=campaign.scientific_hashes(), baseline_sha256=verify.V15_MODEL_SHA256,
                    config_sha256=campaign.CONFIG_SHA256, previous_release_source_sha256=verify.V151_SOURCE_SHA256,
                    v5_source_sha256=verify.V15_SOURCE_SHA256, folds=campaign.FOLDS, regimes=campaign.REGIMES,
                    primary_regimes=campaign.PRIMARY_REGIMES, candidates=[campaign.BASELINE, *campaign.CANDIDATES],
                    gate=campaign.GATE, typed_policy=typed.POLICY, contact_policy=contact_evidence.POLICY,
                    fresh_audit_used_for_selection=False, competition_labels_used=False,
                    contact_truth_claimed=False, video_evidence_active=False)
    protocol = json.loads(json.dumps(protocol))
    return dict(protocol=protocol, protocol_sha256=hashlib.sha256(encoded(protocol)).hexdigest(),
                source_hashes=protocol['source_hashes'], records=rows, selection=selection,
                development_candidate_selected=False, competition_labels_used=False, independent_test_claimed=False,
                incumbent_historical_predictions_reproduced=True, release_candidate_model_sha256=CANDIDATE,
                release_decision=dict(promoted=False, model_sha256=verify.V15_MODEL_SHA256,
                                     independent_audit_status='insufficient-fresh-stoppage-validation-retained-baseline',
                                     reason='No fresh explicit stoppage truth.'))


class ReleaseV7Tests(unittest.TestCase):
    def test_phase_success_still_requires_retention(self):
        self.assertFalse(verify.verify_fresh_audit_v7(audit(), CANDIDATE, campaign.CONFIG_SHA256))
        verify.verify_v7_decision(evidence(), verify.V15_MODEL_SHA256, audit())

    def test_promotion_and_missing_audit_rejected(self):
        for key, value in [('promoted', True), ('model_sha256', CANDIDATE), ('independent_audit_status', 'passed')]:
            e = evidence(); e['release_decision'][key] = value
            with self.assertRaises(ValueError): verify.verify_v7_decision(e, verify.V15_MODEL_SHA256, audit())
        with self.assertRaises(ValueError): verify.verify_v7_decision(evidence(), verify.V15_MODEL_SHA256)

    def test_selection_policy_and_source_tampering(self):
        for mutate in (lambda e: e['selection'].update(audit_candidate='typed-invalid'),
                       lambda e: e['protocol']['typed_policy'].update(maximum_cost_adjustment=.5),
                       lambda e: e['source_hashes'].update({'possession/pipeline.py': '0' * 64})):
            e = evidence(); mutate(e)
            with self.assertRaises(ValueError): verify.verify_v7_decision(e, verify.V15_MODEL_SHA256, audit())

    def test_fresh_scope_hash_and_one_shot_provenance_tampering(self):
        mutations = [lambda a: a.update(passed=True),
                     lambda a: a['access_marker'].update(status='retry'),
                     lambda a: a['predictions_before_reference_access'].pop(next(iter(a['predictions_before_reference_access']))),
                     lambda a: a['audit']['protocol']['sources'].update({'possession/pipeline.py': '0' * 64}),
                     lambda a: a['audit']['protocol']['data'].update({'1953632_tracking_extrapolated.jsonl': '0' * 64}),
                     lambda a: a['audit']['reference_counts']['skillcorner1953632'].update(dead_reference_frames=10),
                     lambda a: a['audit'].update(fresh_stoppage_validation_available=True)]
        for mutate in mutations:
            a = audit(); mutate(a); refresh(a)
            with self.assertRaises(ValueError): verify.verify_fresh_audit_v7(a, CANDIDATE, campaign.CONFIG_SHA256)

    def test_split_and_policy_encoded_bytes_authenticated(self):
        for key in ('split_file_base64', 'policy_file_base64'):
            a = audit(); a[key] = base64.b64encode(b'{}').decode()
            with self.assertRaises(ValueError): verify.verify_fresh_audit_v7(a, CANDIDATE, campaign.CONFIG_SHA256)

    def test_contact_evidence_and_false_claims(self):
        path = Path(__file__).resolve().parents[1] / 'models/contact-audit-v7.json'
        value = json.loads(path.read_text())
        verify.verify_contact_audit_v7(value)
        for key in ('verified_contact', 'hard_overrides_active', 'fresh_test', 'independent_video_evidence'):
            changed = copy.deepcopy(value); changed[key] = True
            with self.assertRaises(ValueError): verify.verify_contact_audit_v7(changed)
        changed = copy.deepcopy(value)
        changed['underlying_summary']['restart_by_regime']['clean']['n_matched'] += 1
        changed['provenance']['underlying_summary_sha256'] = hashlib.sha256(encoded(changed['underlying_summary'])).hexdigest()
        with self.assertRaises(ValueError): verify.verify_contact_audit_v7(changed)

    def test_v6_historical_sources_authenticate_against_v151_archive(self):
        root = Path(__file__).resolve().parents[1]
        for path, access in [('models/temporal-selection.json', lambda x: x['source_hashes']),
                             ('models/fresh-audit-v6.json', lambda x: x['audit']['protocol']['sources'])]:
            source_hashes = access(json.loads((root / path).read_text()))
            verify.verify_archived_sources(source_hashes, root / 'models/baselines/v1.5.1-source.zip', verify.V151_SOURCE_SHA256)

    def test_candidate_artifact_and_training_provenance(self):
        root = Path(__file__).resolve().parents[1]
        e = evidence(); chosen = e['selection']['audit_candidate']; contact = chosen == 'typed-contact'
        names = typed.feature_names(contact); width = len(names)
        model = json.loads((root / 'models/baselines/v1.5.json').read_text())
        model['candidate_id'] = chosen
        model['typed_boundary_model'] = dict(schema_version=1, kind='typed_boundary_softmax',
            classes=list(typed.CLASSES), contact_features=contact, feature_names=list(names), policy=dict(typed.POLICY),
            mean=[0.] * width, scale=[1.] * width, coefficients=np.zeros((width, 3)).tolist(),
            intercepts=[0.] * 3, priors=[.25] * 4,
            metadata=dict(protocol_sha256=e['protocol_sha256'], stage='all-seven-refit',
                development_cv_selection=chosen, training_matches=sorted(CATALOG), training_regimes=list(campaign.REGIMES)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'candidate.json'; path.write_bytes(encoded(model))
            e['release_candidate_model_sha256'] = verify.sha256(path)
            verify.verify_v7_candidate_artifact(e, path)
            model['typed_boundary_model']['metadata']['training_matches'] = ['invented-match']
            path.write_bytes(encoded(model)); e['release_candidate_model_sha256'] = verify.sha256(path)
            with self.assertRaisesRegex(ValueError, 'training provenance'):
                verify.verify_v7_candidate_artifact(e, path)
            e['release_candidate_model_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                verify.verify_v7_candidate_artifact(e, path)


if __name__ == '__main__': unittest.main()

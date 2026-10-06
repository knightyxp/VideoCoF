"""Offline regression tests for the historical GPT judge protocol."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import requests

from metric import gpt_evaluation as scores
from metric import gpt_success_rate as success
from metric import ping_judge_api as ping


class StreamResponse:
    def __init__(self, text, model='returned-snapshot', complete=True):
        self.text, self.model, self.complete = text, model, complete

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        chunk = {'model': self.model, 'choices': [{'delta': {'content': self.text}}]}
        yield ('data: ' + json.dumps(chunk)).encode()
        if self.complete:
            yield b'data: [DONE]'


class JudgeTests(unittest.TestCase):
    def setUp(self):
        # Any test that forgets to mock an HTTP response fails before networking.
        patcher = mock.patch('requests.post', side_effect=AssertionError('Network disabled in offline tests'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self):
        return {'api_key': 'test-key-not-a-credential', 'api_base': 'https://judge.invalid/v1',
                'model': 'gpt-4o-2024-05-13', 'request_timeout': 17, 'max_retries': 0,
                'frames_per_video': 3, 'original_video_root': '.', 'edited_video_root': '.',
                'edited_video_pattern': 'gen_{task_type}_{sample_id}.mp4'}

    def test_historical_rubrics_are_unchanged(self):
        expected = {
            scores: 'fad051a299779e58aa1f066a69afb2f789b68cdb489d33a6179ba449ee0f4e93',
            success: 'd0690a5c9d57b578f4ba682be08c578fb6247be7fdd1e74f6074973b1a5280ff',
        }
        for module, digest in expected.items():
            self.assertEqual(hashlib.sha256(module.EVALUATION_PROMPT_TEXT.encode()).hexdigest(), digest)
        self.assertEqual(scores._compute_evenly_spaced_indices(33, 3), [0, 16, 32])

    def test_environment_configuration_and_cli_override(self):
        argv = ['--input_json', 'input.jsonl', '--output_json', 'output.json']
        for module in (scores, success):
            with self.subTest(module=module.__name__):
                with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-key'}, clear=True):
                    args = module.parse_arguments(argv)
                    self.assertEqual(args.model, 'gpt-4o-2024-05-13')
                    self.assertEqual(args.max_retries, 0)
                env = {'OPENAI_API_KEY': 'offline-key', 'OPENAI_MODEL': 'explicit-env-model',
                       'OPENAI_BASE_URL': 'https://judge.invalid/v1'}
                with mock.patch.dict(os.environ, env, clear=True):
                    self.assertEqual(module.parse_arguments(argv).model, 'explicit-env-model')
                    self.assertEqual(module.parse_arguments(argv + ['--model', 'cli-model']).model, 'cli-model')

    def test_jsonl_and_explicit_paths_override_legacy_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('source.mp4', 'edited.mp4', 'gen_obj_swap_001_compare.mp4'):
                (root / name).touch()
            row = {'sample_id': '001', 'task_type': 'obj_swap', 'instruction': 'Swap the left cup.',
                   'original_video_path': 'source.mp4', 'edited_video_path': 'edited.mp4'}
            manifest = root / 'manifest.jsonl'
            manifest.write_text(json.dumps(row) + '\n\n', encoding='utf-8')
            cfg = dict(self.config(), original_video_root=str(root), edited_video_root=str(root),
                       original_from_compare_left_half=True)
            for module in (scores, success):
                self.assertEqual(module.load_and_normalize_samples(str(manifest)), [row])
                original, edited, crop = module.resolve_sample_paths(row, cfg)
                self.assertEqual(original, str(root / 'source.mp4'))
                self.assertEqual(edited, str(root / 'edited.mp4'))
                self.assertFalse(crop)
                missing = dict(row, original_video_path='absent.mp4')
                self.assertEqual(module.resolve_sample_paths(missing, cfg), (None, None, False))

    def test_stream_tracks_model_and_retries_only_transient_errors(self):
        for module in (scores, success):
            with mock.patch.object(module.requests, 'post', side_effect=[requests.Timeout('secret-url'), StreamResponse('yes')]) as post:
                with mock.patch.object(module.time, 'sleep'):
                    text, usage, model = module.stream_chat_completion(
                        'https://judge.invalid/v1/chat/completions', 'offline-key', {'model': 'requested'},
                        False, timeout=17, max_retries=1)
            self.assertEqual((text, usage, model), ('yes', None, 'returned-snapshot'))
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args[1]['timeout'], 17)
            response = requests.Response()
            response.status_code = 401
            failure = requests.HTTPError('private-key at private-endpoint', response=response)
            with mock.patch.object(module.requests, 'post', side_effect=failure) as post:
                with self.assertRaises(RuntimeError) as caught:
                    module.stream_chat_completion('private-endpoint', 'private-key', {}, False, max_retries=3)
            self.assertEqual(post.call_count, 1)
            self.assertNotIn('private-key', str(caught.exception))
            self.assertNotIn('private-endpoint', str(caught.exception))

    def test_truncated_stream_is_not_a_judgment(self):
        for module in (scores, success):
            with mock.patch.object(module.requests, 'post', return_value=StreamResponse('yes', complete=False)):
                with self.assertRaises(RuntimeError):
                    module.stream_chat_completion('https://judge.invalid', 'offline-key', {}, False)

    def test_three_score_request_and_metadata(self):
        reply = json.dumps({'instruct follow score': 8, 'quality score': 7, 'preservation score': 9})
        with mock.patch.object(scores, 'extract_evenly_spaced_frames', return_value=['a', 'b', 'c']) as frames:
            with mock.patch.object(scores.requests, 'post', return_value=StreamResponse(reply)) as post:
                result = scores.evaluate_sample('source.mp4', 'edited.mp4', 'Edit.', {'sample_id': '001'}, self.config())
        self.assertEqual(frames.call_args_list[0][0], ('source.mp4', 3))
        payload = post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.1)
        self.assertEqual(sum(part['type'] == 'image_url' for part in payload['messages'][1]['content']), 6)
        self.assertEqual(result['judge_metadata']['returned_model'], 'returned-snapshot')
        self.assertEqual(result['judge_metadata']['requested_model'], 'gpt-4o-2024-05-13')
        self.assertEqual(result['original_frames_sampled'], 3)

    def test_success_uses_first_frames_and_failures_are_excluded(self):
        cfg = self.config()
        cfg.pop('frames_per_video')
        with mock.patch.object(success, 'extract_frames_by_indices', return_value=['frame']) as frames:
            with mock.patch.object(success.requests, 'post', return_value=StreamResponse('no')) as post:
                result = success.evaluate_sample('source.mp4', 'edited.mp4', 'Edit.', {'sample_id': '001'}, cfg)
        self.assertEqual([call[0][1] for call in frames.call_args_list], [[0], [0]])
        self.assertEqual(post.call_args[1]['json']['temperature'], 0.0)
        self.assertFalse(result['success'])
        summary = success.compute_success_summary([result, {'error': 'network failure'}])
        self.assertEqual(summary['num_evaluated'], 1)
        self.assertEqual(summary['success_rate'], 0.0)
        self.assertIsNone(success.parse_yes_no_response('I cannot answer yes or no.'))
        with mock.patch.object(scores, 'extract_evenly_spaced_frames', return_value=['frame']):
            with mock.patch.object(scores.requests, 'post', return_value=StreamResponse('{"error":"unavailable"}')):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(scores.evaluate_sample('a', 'b', 'Edit.', {}, self.config()))
        self.assertEqual(scores.compute_average_scores([{'evaluation': {'quality score': 0}}])['num_scored'], 0)

    def test_resume_retries_failed_case_without_duplicating_completed_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, edited = root / 'source.mp4', root / 'edited.mp4'
            source.touch()
            edited.touch()
            manifest, output = root / 'manifest.jsonl', root / 'scores.json'
            rows = [{'sample_id': sid, 'task_type': 'obj_swap', 'instruction': 'Edit ' + sid,
                     'original_video_path': str(source), 'edited_video_path': str(edited)} for sid in ('a', 'b', 'c')]
            manifest.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            argv = ['--input_json', str(manifest), '--output_json', str(output), '--api_key', 'offline-key']
            with mock.patch.dict(os.environ, {}, clear=True):
                args = scores.parse_arguments(argv)
            def evaluated(original, changed, instruction, sample, cfg, **kwargs):
                return {'sample_id': sample['sample_id'], 'evaluation': {
                    'instruct follow score': 8, 'quality score': 7, 'preservation score': 9}}
            def first_run(*args, **kwargs):
                return None if args[3]['sample_id'] == 'b' else evaluated(*args, **kwargs)
            with mock.patch.object(scores, 'parse_arguments', return_value=args):
                with mock.patch.object(scores, 'evaluate_sample', side_effect=first_run):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(scores.main(), 1)
                first = json.loads(output.read_text())
                self.assertEqual(first['coverage']['num_missing_or_failed'], 1)
                with mock.patch.object(scores, 'evaluate_sample', side_effect=evaluated) as evaluate:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(scores.main(), 0)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluate.call_args[0][3]['sample_id'], 'b')
                original_api_base = args.api_base
                args.api_base = 'https://another-provider.invalid/v1'
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    scores.main()
                args.api_base = original_api_base
                edited.write_bytes(b'new edited video at the same path')
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    scores.main()
            final = json.loads(output.read_text())
            self.assertEqual([row['sample_id'] for row in final['results']], ['a', 'b', 'c'])
            self.assertEqual(final['coverage']['num_missing_or_failed'], 0)

    def test_resume_identity_hashes_source_output_and_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, edited = root / 'source.mp4', root / 'edited.mp4'
            source.write_bytes(b'original video')
            edited.write_bytes(b'edited video')
            row = {'sample_id': '001', 'task_type': 'obj_swap', 'instruction': 'Edit.',
                   'original_video_path': str(source), 'edited_video_path': str(edited)}
            for module in (scores, success):
                cfg = self.config()
                first = module.evaluation_key(row, cfg)
                provider = dict(cfg, api_base='https://provider.invalid/private-credential/v1')
                self.assertNotEqual(first, module.evaluation_key(row, provider))
                source.write_bytes(b'changed original video')
                self.assertNotEqual(first, module.evaluation_key(row, cfg))
                source.write_bytes(b'original video')
                edited.write_bytes(b'changed edited video')
                self.assertNotEqual(first, module.evaluation_key(row, cfg))
                edited.write_bytes(b'edited video')
                self.assertEqual(first, module.evaluation_key(row, cfg))

    def test_success_main_signals_missing_video_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, output = root / 'manifest.json', root / 'output.json'
            row = {'sample_id': 'missing', 'task_type': 'obj_swap', 'instruction': 'Edit.',
                   'original_video_path': str(root / 'missing-source.mp4'),
                   'edited_video_path': str(root / 'missing-edited.mp4')}
            manifest.write_text(json.dumps([row]), encoding='utf-8')
            with mock.patch.dict(os.environ, {}, clear=True):
                args = success.parse_arguments(['--input_json', str(manifest), '--output_json', str(output),
                                                '--api_key', 'offline-key'])
            with mock.patch.object(success, 'parse_arguments', return_value=args):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(success.main(), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result['coverage']['num_missing_or_failed'], 1)
            self.assertEqual(result['results'], [])

    def test_ping_reports_returned_model_and_redacts_request_errors(self):
        response = mock.Mock()
        response.json.return_value = {'model': 'actual-model', 'choices': [{'message': {'content': 'pong'}}]}
        with mock.patch.object(ping.requests, 'post', return_value=response) as post:
            result = ping.ping_api('offline-key', 'https://judge.invalid/v1', 'requested-model')
        self.assertEqual(result['returned_model'], 'actual-model')
        self.assertFalse(result['model_matches'])
        self.assertEqual(post.call_args[1]['json']['max_tokens'], 8)
        with mock.patch.object(ping.requests, 'post', side_effect=requests.Timeout('private-key private-endpoint')):
            with self.assertRaises(RuntimeError) as caught:
                ping.ping_api('private-key', 'private-endpoint', 'requested-model')
        self.assertNotIn('private-key', str(caught.exception))
        self.assertNotIn('private-endpoint', str(caught.exception))


if __name__ == '__main__':
    unittest.main()

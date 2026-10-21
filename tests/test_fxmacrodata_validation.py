"""Calendar parser regressions; runs without plotting or analysis dependencies."""
import importlib.util
from pathlib import Path
import unittest


def load_client(path):
    spec = importlib.util.spec_from_file_location('calendar_client', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalendarValidationTest(unittest.TestCase):
    client_path = Path(__file__).resolve().parents[1] / 'src/common/fxmacrodata.py'

    @classmethod
    def setUpClass(cls):
        cls.client = load_client(cls.client_path)

    def test_empty_success(self):
        self.assertEqual([], self.client.parse_release_calendar({'data': []}))

    def test_sort_and_copy(self):
        rows = [{'announcement_datetime': 200}, {'announcement_datetime': 100.5}]
        result = self.client.parse_release_calendar({'data': rows})
        self.assertEqual([100.5, 200], [r['announcement_datetime'] for r in result])
        self.assertIsNot(result[0], rows[1])

    def test_malformed_rows(self):
        for payload in (None, [], {}, {'data': [None]}, {'data': [{}]}, {'data': [{'announcement_datetime': True}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(self.client.FXMacroDataError):
                    self.client.parse_release_calendar(payload)

    def test_rejects_error_with_empty_data(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'contains an error') as raised:
            self.client.parse_release_calendar({'error': 'private upstream text', 'data': []})
        self.assertNotIn('private', str(raised.exception))

    def test_rejects_detail_with_rows(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'contains an error'):
            self.client.parse_release_calendar({'detail': 'request denied', 'data': [{'announcement_datetime': 100}]})

    def test_rejects_nan(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'finite'):
            self.client.parse_release_calendar({'data': [{'announcement_datetime': float('nan')}]})

    def test_rejects_positive_infinity(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'finite'):
            self.client.parse_release_calendar({'data': [{'announcement_datetime': float('inf')}]})

    def test_rejects_negative_infinity(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'finite'):
            self.client.parse_release_calendar({'data': [{'announcement_datetime': float('-inf')}]})

    def test_rejects_overflowing_integer(self):
        with self.assertRaisesRegex(self.client.FXMacroDataError, 'finite'):
            self.client.parse_release_calendar({'data': [{'announcement_datetime': 10 ** 400}]})


if __name__ == '__main__':
    unittest.main()

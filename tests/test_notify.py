from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.run_with_notify import signal_name
from utils.notify import NtfyNotifier


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class NotifyTests(unittest.TestCase):
    def test_disabled_notifier_is_a_noop(self):
        with patch.dict("os.environ", {}, clear=True):
            notifier = NtfyNotifier()
        self.assertFalse(notifier.enabled)
        self.assertFalse(notifier.send("ignored", "ignored"))

    def test_successful_notification(self):
        notifier = NtfyNotifier("unit-test-topic")
        with patch("urllib.request.urlopen", return_value=_Response()) as mocked:
            self.assertTrue(notifier.send("Training Done", "ok"))
        request = mocked.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/unit-test-topic"))
        self.assertEqual(request.headers["Title"], "Training Done")

    def test_watchdog_signal_names(self):
        self.assertEqual(signal_name(-11), "SIGSEGV")
        self.assertEqual(signal_name(139), "SIGSEGV")
        self.assertIsNone(signal_name(1))


if __name__ == "__main__":
    unittest.main()


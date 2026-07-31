from __future__ import annotations

import pytest

from app.enrichment.tracking import TrackingConfig, run_tracking_not_implemented


def test_tracking_config_defaults_to_disabled():
    assert TrackingConfig().enabled is False


def test_tracking_is_explicitly_deferred_not_silently_broken():
    with pytest.raises(NotImplementedError, match="deferred"):
        run_tracking_not_implemented()

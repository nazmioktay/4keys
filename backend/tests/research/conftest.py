"""Araştırma testleri ağa çıkmaz: koşucunun Binance limit okuması (min notional/adım) varsayılan olarak 'limit yok' döner.
Limit davranışını sınayan testler `runner.load_account_limits`'i kendileri yamalar."""

import pytest


@pytest.fixture(autouse=True)
def _no_network_limits(monkeypatch):
    from research import runner

    if "_load_account_limits_original" not in runner.__dict__:  # gerçek zinciri sınayan test için orijinali sakla
        runner._load_account_limits_original = runner.load_account_limits
    monkeypatch.setattr(runner, "load_account_limits", lambda symbols: (None, None, []))

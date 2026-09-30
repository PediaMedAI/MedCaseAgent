"""The default test suite must not contact a model or download a publication."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Network access is disabled in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)

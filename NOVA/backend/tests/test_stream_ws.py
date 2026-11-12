"""The WebSocket event tail – the HUD's live feed.

The contract the frontend relies on: a reconnect with ``after_seq`` replays
exactly the events it missed (the stream is a read model over the persisted
log), and an unauthorised socket is closed with a game-specific code.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import Api, complete_training, make_settings
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import create_app


def stream_url(game_id: str, after_seq: int = 0, token: str | None = None) -> str:
    url = f"/api/v1/games/{game_id}/stream?after_seq={after_seq}"
    return f"{url}&token={token}" if token else url


def test_stream_announces_itself_then_replays_the_log(client: TestClient, api: Api) -> None:
    complete_training(api)
    snapshot = api.get("").json()

    with client.websocket_connect(stream_url(api.game_id)) as socket:
        ready = socket.receive_json()
        assert ready["type"] == "stream.ready"
        assert ready["game_id"] == api.game_id
        assert ready["player_name"] == "Test Commander"
        assert ready["after_seq"] == 0
        assert ready["event_seq"] == snapshot["event_seq"]

        page = socket.receive_json()
        assert page["type"] == "events"
        assert page["next_seq"] == ready["event_seq"]
        types = [event["type"] for event in page["events"]]
        assert types[0] == "game.created"
        assert "mission.completed" in types
        assert page["events"] == sorted(page["events"], key=lambda e: e["seq"])


def test_ping_is_answered_with_the_current_cursor(client: TestClient, api: Api) -> None:
    with client.websocket_connect(stream_url(api.game_id)) as socket:
        socket.receive_json()  # ready
        socket.send_text("ping")
        messages = [socket.receive_json() for _ in range(2)]
        pong = next(m for m in messages if m["type"] == "pong")
        assert pong["cursor"] >= 0


def test_a_reconnect_only_receives_what_it_missed(client: TestClient, api: Api) -> None:
    complete_training(api)
    cursor = api.get("").json()["event_seq"]

    api.start("analog_systems_check")
    api.act("analog_systems_check", "check_ventilation")
    missed = api.get("/events", params={"after_seq": cursor}).json()
    assert missed["events"]

    with client.websocket_connect(stream_url(api.game_id, after_seq=cursor)) as socket:
        ready = socket.receive_json()
        assert ready["after_seq"] == cursor
        page = socket.receive_json()
        assert page["type"] == "events"
        assert [e["seq"] for e in page["events"]] == [e["seq"] for e in missed["events"]]
        assert all(e["seq"] > cursor for e in page["events"])


def test_events_emitted_while_connected_are_pushed(client: TestClient, api: Api) -> None:
    with client.websocket_connect(stream_url(api.game_id)) as socket:
        socket.receive_json()  # ready
        socket.receive_json()  # initial replay

        api.start("training_prep")
        socket.send_text("ping")

        pushed: list[dict[str, Any]] = []
        for _ in range(4):
            message = socket.receive_json()
            if message["type"] == "events":
                pushed.extend(message["events"])
        assert any(event["type"] == "mission.started" for event in pushed)


def test_unknown_game_is_closed_with_4404(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(stream_url("nova_does_not_exist")) as socket:
            socket.receive_json()
    assert exc.value.code == 4404


def test_auth_enabled_requires_a_token_on_the_socket(tmp_path: Any) -> None:
    settings = make_settings(tmp_path, auth_enabled=True)
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/games", json={"player_name": "Guarded", "difficulty": "standard", "seed": 4}
        ).json()
        game_id = created["game_id"]

        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(stream_url(game_id)) as socket:
                socket.receive_json()
        assert exc.value.code == 4401

        with client.websocket_connect(
            stream_url(game_id, token=created["session_token"])
        ) as socket:
            assert socket.receive_json()["type"] == "stream.ready"


def test_a_token_for_another_game_is_rejected(tmp_path: Any) -> None:
    settings = make_settings(tmp_path, auth_enabled=True)
    with TestClient(create_app(settings)) as client:
        payload = {"player_name": "Two crews", "difficulty": "standard", "seed": 4}
        first = client.post("/api/v1/games", json=payload).json()
        second = client.post("/api/v1/games", json=payload).json()

        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                stream_url(second["game_id"], token=first["session_token"])
            ) as socket:
                socket.receive_json()
        assert exc.value.code == 4401


def test_the_trusted_api_key_opens_the_socket(tmp_path: Any) -> None:
    settings = make_settings(tmp_path, auth_enabled=True, api_key="server-to-server")
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/games", json={"player_name": "Ops", "difficulty": "standard", "seed": 4}
        ).json()
        url = f"/api/v1/games/{created['game_id']}/stream?api_key=server-to-server"
        with client.websocket_connect(url) as socket:
            assert socket.receive_json()["type"] == "stream.ready"


def test_negative_cursor_is_rejected(client: TestClient, api: Api) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/api/v1/games/{api.game_id}/stream?after_seq=-1"
        ) as socket:
            socket.receive_json()

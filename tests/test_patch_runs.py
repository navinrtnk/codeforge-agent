"""Run-level write authority, patch tool events, and conversation mode isolation."""

from pathlib import Path

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall

PATCH = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"


def tool(call_id: str, name: str, patch: str = PATCH) -> ModelResponse:
    return ModelResponse(
        call_id,
        "fake",
        "test-model",
        (ToolCall(call_id, name, {"patch": patch}),),
        "tool_use",
        TokenUsage(10, 5),
    )


def final() -> ModelResponse:
    return ModelResponse(
        "final",
        "fake",
        "test-model",
        (TextContent("Finished; edits are unverified."),),
        "end_turn",
        TokenUsage(10, 5),
    )


def client_for(tmp_path: Path, model: FakeModelClient, *, writes: bool) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url="sqlite://",
        workspace_root=tmp_path,
        code_writes_enabled=writes,
        git_allow_mutations=True,
        _env_file=None,
    )
    return TestClient(create_app(settings, model_client=model))


def register(client: TestClient, tmp_path: Path) -> tuple[str, Path]:
    path = tmp_path / "repo"
    path.mkdir()
    (path / "app.py").write_text("old\n", encoding="utf-8")
    response = client.post("/repositories", json={"name": "Test", "path": str(path)})
    assert response.status_code == 201
    return str(response.json()["id"]), path


def test_default_read_only_can_validate_but_cannot_apply_or_mutate_git(tmp_path: Path) -> None:
    model = FakeModelClient(
        [tool("validate", "validate_patch"), tool("guess", "apply_patch"), final()]
    )
    with client_for(tmp_path, model, writes=True) as client:
        repository_id, path = register(client, tmp_path)
        response = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Inspect patch"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["mode"] == "read_only" and body["status"] == "completed"
        assert body["tool_events"][0]["result"]["dry_run"] is True
        assert body["tool_events"][1]["is_error"] is True
        assert (path / "app.py").read_text() == "old\n"
    names = {definition.name for definition in model.requests[0].tools}
    assert "validate_patch" in names
    assert "apply_patch" not in names
    assert "git_commit" not in names


def test_write_run_applies_and_persists_mode_and_tool_results(tmp_path: Path) -> None:
    model = FakeModelClient(
        [tool("validate", "validate_patch"), tool("apply", "apply_patch"), final()]
    )
    with client_for(tmp_path, model, writes=True) as client:
        repository_id, path = register(client, tmp_path)
        response = client.post(
            f"/repositories/{repository_id}/runs",
            json={"task": "Apply the proposed patch", "mode": "write"},
        )
        body = response.json()
        assert response.status_code == 201 and body["status"] == "completed"
        assert body["mode"] == "write"
        assert (path / "app.py").read_text() == "new\n"
        assert body["tool_events"][1]["result"]["applied"] is True
        assert body["tool_events"][1]["result"]["files"] == [{"path": "app.py", "action": "modify"}]
        saved = client.get(f"/repositories/{repository_id}/runs/{body['id']}")
        assert saved.json()["mode"] == "write"
        assert saved.json()["tool_events"] == body["tool_events"]
    assert "apply_patch" in {definition.name for definition in model.requests[0].tools}


def test_write_runs_require_operator_setting_and_valid_mode(tmp_path: Path) -> None:
    model = FakeModelClient([])
    with client_for(tmp_path, model, writes=False) as client:
        repository_id, _ = register(client, tmp_path)
        assert (
            client.post(
                f"/repositories/{repository_id}/runs", json={"task": "Edit", "mode": "write"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/repositories/{repository_id}/runs", json={"task": "Edit", "mode": "anything"}
            ).status_code
            == 422
        )
        assert client.get(f"/repositories/{repository_id}/runs").json() == []
    assert model.requests == []


def test_rejected_patch_is_recorded_as_tool_error_without_writes(tmp_path: Path) -> None:
    model = FakeModelClient(
        [tool("stale", "apply_patch", PATCH.replace("-old", "-stale")), final()]
    )
    with client_for(tmp_path, model, writes=True) as client:
        repository_id, path = register(client, tmp_path)
        body = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Apply", "mode": "write"}
        ).json()
        assert body["tool_events"][0]["is_error"] is True
        assert "stale" in body["tool_events"][0]["result"]["error"]
        assert (path / "app.py").read_text() == "old\n"


def test_conversation_write_authority_is_not_inherited_by_follow_up(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            tool("apply", "apply_patch"),
            final(),
            tool("guess", "apply_patch", PATCH.replace("-old", "-new").replace("+new", "+again")),
            final(),
        ]
    )
    with client_for(tmp_path, model, writes=True) as client:
        repository_id, path = register(client, tmp_path)
        conversation_id = client.post(
            f"/repositories/{repository_id}/conversations", json={}
        ).json()["id"]
        url = f"/repositories/{repository_id}/conversations/{conversation_id}/messages"
        first = client.post(url, json={"content": "Edit app.py", "mode": "write"})
        assert first.status_code == 201 and first.json()["mode"] == "write"
        second = client.post(url, json={"content": "Explain the change"})
        assert second.status_code == 201 and second.json()["mode"] == "read_only"
        assert second.json()["tool_events"][0]["is_error"] is True
        assert (path / "app.py").read_text() == "new\n"
    assert "apply_patch" not in {definition.name for definition in model.requests[2].tools}

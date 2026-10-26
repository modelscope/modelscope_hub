"""One hundred corruption cases, real HTTP framing, isolation and explicit repair."""

from __future__ import annotations

import json
from pathlib import Path

from modelscope_hub import HubApi
from tests.download_corruption_harness import exercise_stages, local_experiment, synthetic_cases


def test_hundred_corrupt_files_continue_isolate_repair_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD", "false")
    monkeypatch.delenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION", raising=False)
    monkeypatch.delenv("MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS", raising=False)
    cases = synthetic_cases()
    root = tmp_path / "hundred-corrupt-files"
    with local_experiment(cases) as (server, router):
        evidence = exercise_stages(root, cases, lambda: HubApi(endpoint=server.url, token=""), router)
        assert server.credential_requests == 0
        assert evidence["passed_fault_cases"] == 100
        assert all(group["passed"] == 10 for group in evidence["group_results"].values())
        assert evidence["independent_size_sha256_verified"] == 200
        stages = evidence["stages"]
        assert [stage["logical_file_requests"] for stage in stages.values()] == [300, 0, 0, 60, 0]
        assert stages["first"]["summary"]["succeeded"] == 140
        assert stages["explicit_repair"]["summary"]["succeeded"] == 200
        ranges = [row for row in evidence["cases"] if row["group"] in (9, 10)]
        assert len(ranges) == 20
        assert all(row["result"]["retries"] == 1 for row in evidence["cases"])
        tracker = root / ".ms_download_state.json"
        saved = json.loads(tracker.read_text())
        assert saved["files"] == {} and saved["pages"] == {}
        assert saved["corruption_regression"]["passed"]
        assert len(list(root.rglob("*.json"))) == 1
        assert not list(root.rglob("*.jsonl"))
        assert Path(stages["reuse"]["summary"]["report_path"]) == tracker

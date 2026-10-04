"""Sealed pending-release export packages: seal, release, claim.

A package seals a registered graph's small-sample preview result into SQLite;
release is gated on an expected registry generation and the graph's current
review; claim returns the *stored* sample (never recomputed) and only while the
release generation still equals the current registry generation. Every package
operation shares the store lock with registry replacements, so seal, release,
claim and replacement take effect in one total order.
"""

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import registry
from app.main import app
from app.storage import get_store, reset_store

client = TestClient(app)
# A storage fault must surface as 500/INTERNAL, not crash the test run.
no_raise = TestClient(app, raise_server_exceptions=False)


# Clean graph: nothing sensitive in-graph; the registry alone decides safety.
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}
ROWS = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}
SAMPLE_COLUMNS = ["user_id", "mail", "bucket"]
SAMPLE_ROWS = [[1, "a@x.example", "ALL"]]


def register(payload=GRAPH):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def put_registry(entries, generation):
    return client.put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )


def seal(gid, rows=ROWS, conditions=CONDITIONS):
    return client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": rows, "conditions": conditions},
    )


def release(pid, generation):
    return client.post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": generation},
    )


def claim(pid):
    return client.get(f"/api/packages/{pid}")


def package_count():
    return get_store()._conn.execute("SELECT COUNT(*) FROM packages").fetchone()[0]


def released_generation(pid):
    return get_store().get_package_row(pid)["released_generation"]


# -------------------------------- sealing ----------------------------------


def test_seal_returns_only_package_id_and_creation_generation():
    gid = register()
    resp = seal(gid)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "generation"}
    assert isinstance(body["id"], str) and body["id"]
    assert body["generation"] == 0


def test_seal_unknown_graph_returns_404():
    resp = seal("does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "GRAPH_NOT_FOUND"


def test_seal_reuses_original_preview_rules():
    gid = register()
    # Missing rows for the only source: the original coverage error.
    resp = seal(gid, rows={}, conditions=CONDITIONS)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_SOURCE_COVERAGE"

    # Wrong row keys.
    resp = seal(gid, rows={"users": [{"uid": 1}]}, conditions=CONDITIONS)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_ROW_KEYS"

    # Non-scalar cell.
    resp = seal(gid, rows={"users": [
        {"uid": 1, "email": ["x"], "country": "CN"}]}, conditions=CONDITIONS)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_NON_SCALAR"

    # Over the 200-row cap.
    resp = seal(gid, rows={"users": [
        {"uid": i, "email": "e", "country": "CN"} for i in range(201)]},
        conditions=CONDITIONS)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_ROW_LIMIT"

    # Unknown field / missing rows are schema errors: 422.
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS, "nodes": []},
    )
    assert resp.status_code == 422
    resp = client.post(
        f"/api/graphs/{gid}/packages", json={"conditions": CONDITIONS}
    )
    assert resp.status_code == 422

    # Every rejection left the package table empty.
    assert package_count() == 0


def test_seal_blocked_by_registry_creates_no_package():
    gid = register()
    put_registry([{"source": "users", "column": "email"}], 0)
    resp = seal(gid)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "PREVIEW_BLOCKED"
    assert "rows" not in error and "columns" not in error
    assert error["review"]["blocked"] is True
    assert package_count() == 0


def test_seal_records_the_current_creation_generation():
    gid = register()
    put_registry([], 0)  # no-op content, but generation advances to 1
    body = seal(gid).json()
    assert body["generation"] == 1
    row = get_store().get_package_row(body["id"])
    assert row["created_generation"] == 1
    assert row["released_generation"] is None


def test_sealed_sample_matches_the_registered_preview():
    gid = register()
    pid = seal(gid).json()["id"]
    put_registry([], 0)  # generation 1, still safe; release there
    assert release(pid, 1).status_code == 200
    body = claim(pid).json()
    assert body["columns"] == SAMPLE_COLUMNS
    assert body["rows"] == SAMPLE_ROWS


# -------------------------------- release ----------------------------------


def test_release_returns_only_package_id_and_generation():
    gid = register()
    pid = seal(gid).json()["id"]
    resp = release(pid, 0)
    assert resp.status_code == 200
    assert resp.json() == {"id": pid, "generation": 0}
    assert released_generation(pid) == 0


def test_release_unknown_package_returns_404():
    resp = release("no-such-package", 0)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PACKAGE_NOT_FOUND"


def test_release_with_stale_expected_generation_returns_409_and_writes_nothing():
    gid = register()
    pid = seal(gid).json()["id"]
    put_registry([], 0)  # generation advances to 1
    resp = release(pid, 0)  # replay the old generation
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "PACKAGE_GENERATION_CONFLICT"
    assert error["expected_generation"] == 0
    assert error["current_generation"] == 1
    assert "rows" not in error and "columns" not in error
    assert released_generation(pid) is None


def test_release_when_graph_currently_blocked_returns_400_and_does_not_release():
    gid = register()
    pid = seal(gid).json()["id"]
    put_registry([{"source": "users", "column": "email"}], 0)
    resp = release(pid, 1)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "PACKAGE_GRAPH_BLOCKED"
    assert error["node_id"] == "export_users"
    assert error["review"]["blocked"] is True
    assert released_generation(pid) is None
    # Claim stays impossible: the failed release wrote no release record.
    claim_resp = claim(pid)
    assert claim_resp.status_code == 409
    assert claim_resp.json()["error"]["code"] == "PACKAGE_NOT_RELEASED"


def test_successful_release_persists_the_release_generation():
    gid = register()
    pid = seal(gid).json()["id"]
    assert release(pid, 0).status_code == 200
    assert released_generation(pid) == 0


def test_release_is_idempotent_at_the_same_generation():
    gid = register()
    pid = seal(gid).json()["id"]
    assert release(pid, 0).status_code == 200
    again = release(pid, 0)
    assert again.status_code == 200
    assert again.json() == {"id": pid, "generation": 0}


def test_registry_change_blocks_release_even_after_clearing():
    gid = register()
    pid = seal(gid).json()["id"]  # created at generation 0
    # Block at generation 1.
    put_registry([{"source": "users", "column": "email"}], 0)
    assert release(pid, 1).status_code == 400
    # Clear again at generation 2 (safe); the stale expectation is still 409,
    # and the current expectation must be used for a fresh release.
    assert put_registry([], 1).status_code == 200
    assert release(pid, 1).status_code == 409
    assert release(pid, 2).status_code == 200
    assert released_generation(pid) == 2


# --------------------------------- claim -----------------------------------


def test_claim_unknown_package_returns_404():
    resp = claim("no-such-package")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PACKAGE_NOT_FOUND"


def test_claim_without_release_returns_409_without_sample():
    gid = register()
    pid = seal(gid).json()["id"]
    resp = claim(pid)
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "PACKAGE_NOT_RELEASED"
    assert error["current_generation"] == 0
    assert "rows" not in error and "columns" not in error


def test_claim_returns_sealed_sample_graph_id_and_both_generations():
    gid = register()
    pid = seal(gid).json()["id"]
    release(pid, 0)
    resp = claim(pid)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "id", "graph_id", "columns", "rows",
        "created_generation", "released_generation",
    }
    assert body["id"] == pid
    assert body["graph_id"] == gid
    assert body["columns"] == SAMPLE_COLUMNS
    assert body["rows"] == SAMPLE_ROWS
    assert body["created_generation"] == 0
    assert body["released_generation"] == 0


def test_claim_never_recomputes_the_sample(monkeypatch):
    gid = register()
    pid = seal(gid).json()["id"]
    release(pid, 0)
    # Claiming must read the sealed sample straight from storage: even if the
    # preview engine were unusable, the stored rows come back unchanged.
    import app.registry as reg_mod

    def boom(request, extra_sensitive):
        raise AssertionError("claim must not recompute the preview")

    monkeypatch.setattr(reg_mod, "run_preview", boom)
    body = claim(pid).json()
    assert body["columns"] == SAMPLE_COLUMNS
    assert body["rows"] == SAMPLE_ROWS


def test_claim_after_registry_change_is_409_even_though_graph_is_safe_again():
    gid = register()
    pid = seal(gid).json()["id"]
    assert release(pid, 0).status_code == 200
    # Generation 1 blocks; generation 2 clears again (graph safe once more).
    put_registry([{"source": "users", "column": "email"}], 0)
    put_registry([], 1)
    resp = claim(pid)
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "PACKAGE_GENERATION_CONFLICT"
    assert error["expected_generation"] == 0
    assert error["current_generation"] == 2
    assert "rows" not in error and "columns" not in error

    # The graph is safe at generation 2; a fresh release makes claim work,
    # still handing out the original sealed sample, with moved generations.
    assert release(pid, 2).status_code == 200
    body = claim(pid).json()
    assert body["rows"] == SAMPLE_ROWS
    assert body["created_generation"] == 0
    assert body["released_generation"] == 2


def test_claim_uses_no_other_batch_of_rows(monkeypatch):
    # A different rows/conditions batch never reaches claim: claim ignores the
    # request entirely and returns exactly what was sealed.
    gid = register()
    pid = seal(gid).json()["id"]
    release(pid, 0)
    body = claim(pid).json()
    assert body["rows"] == SAMPLE_ROWS  # uid 2 was filtered out at seal time


# ------------------------- blocked release keeps old record ----------------


def test_blocked_re_release_leaves_the_old_release_generation_in_place():
    gid = register()
    pid = seal(gid).json()["id"]
    release(pid, 0)
    # New generation blocks the graph: re-release fails, old record survives.
    put_registry([{"source": "users", "column": "email"}], 0)
    assert release(pid, 1).status_code == 400
    assert released_generation(pid) == 0
    # Claim is still a (stale) 409, never a success at the old generation.
    resp = claim(pid)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "PACKAGE_GENERATION_CONFLICT"


# ---------------------------- total ordering -------------------------------


def test_seal_and_replacement_take_effect_in_one_order(monkeypatch):
    """A seal either fully precedes or fully follows a replacement.

    Hold the store lock inside the seal's evaluation so a replacement blocks:
    the in-flight seal must record created_generation 0; afterwards the
    replacement commits generation 1 and releasing with the old generation is
    impossible.
    """
    gid = register()
    entered = threading.Event()
    release_event = threading.Event()
    import app.registry as reg_mod

    original_preview = reg_mod.run_preview

    def slow_preview(request, extra_sensitive):
        entered.set()
        assert release_event.wait(timeout=5)
        return original_preview(request, extra_sensitive)

    monkeypatch.setattr(reg_mod, "run_preview", slow_preview)

    outcomes = {}

    def do_seal():
        outcomes["seal"] = registry.seal_package(gid, ROWS, CONDITIONS)

    worker = threading.Thread(target=do_seal)
    worker.start()
    assert entered.wait(timeout=5)
    pending = threading.Thread(
        target=lambda: outcomes.setdefault(
            "replace",
            registry.replace_registry([("users", "email")], 0),
        )
    )
    pending.start()
    assert "replace" not in outcomes
    release_event.set()
    worker.join(timeout=5)
    pending.join(timeout=5)

    # The seal used the gen-0 snapshot; the replacement then applied at gen 1.
    assert outcomes["seal"]["generation"] == 0
    assert outcomes["replace"]["generation"] == 1
    pid = outcomes["seal"]["id"]
    # Releasing with the pre-replacement generation is rejected.
    resp = release(pid, 0)
    assert resp.status_code == 409
    assert resp.json()["error"]["current_generation"] == 1
    # The current generation also blocks the graph, so release fails 400 and
    # the package cannot be claimed.
    assert release(pid, 1).status_code == 400
    assert claim(pid).status_code == 409


def test_concurrent_releases_serialize_one_stales_out(monkeypatch):
    gid = register()
    pid = seal(gid).json()["id"]
    # Hold a gen-0 release inside the lock (just before its write commits) so
    # a registry replacement queues behind it; once it commits (200) the
    # replacement advances to generation 1 and a second release still
    # submitting generation 0 is deterministically rejected (409).
    entered = threading.Event()
    release_event = threading.Event()
    store = get_store()
    original_release_write = store.release_package

    def slow_release_write(package_id, generation):
        entered.set()
        assert release_event.wait(timeout=5)
        return original_release_write(package_id, generation)

    monkeypatch.setattr(store, "release_package", slow_release_write)

    outcomes = []

    def do_release(generation):
        outcomes.append(release(pid, generation).status_code)

    t1 = threading.Thread(target=do_release, args=(0,))
    t1.start()
    assert entered.wait(timeout=5)
    pending = threading.Thread(
        target=lambda: outcomes.append(
            client.put(
                "/api/sensitive-columns",
                json={"entries": [], "expected_generation": 0},
            ).status_code
        )
    )
    pending.start()
    release_event.set()
    t1.join(timeout=5)
    pending.join(timeout=5)
    # Release at gen 0 committed, then the replacement moved to gen 1.
    assert outcomes == [200, 200]
    do_release(0)  # now a stale release
    assert outcomes == [200, 200, 409]
    # The winning release stayed at generation 0; the loser wrote nothing.
    assert released_generation(pid) == 0


# ---------------------------- write atomicity ------------------------------


class FaultyConnection:
    """Delegates to the real connection, failing an armed statement once."""

    def __init__(self, real):
        self._real = real
        self._predicate = None

    def arm(self, predicate):
        self._predicate = predicate

    def _check(self, sql):
        if self._predicate is not None and self._predicate(sql):
            self._predicate = None
            raise sqlite3.OperationalError("injected storage fault")

    def execute(self, sql, parameters=()):
        self._check(sql)
        return self._real.execute(sql, parameters)

    def executemany(self, sql, seq):
        self._check(sql)
        return self._real.executemany(sql, seq)

    def commit(self):
        self._check("COMMIT")
        return self._real.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self._real.commit()
        else:
            self._real.rollback()
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def faulty_conn():
    """Install a fault-injecting connection proxy around the current store.

    The autouse fresh-store fixture already created the real store, so every
    test (including the fault tests) still uses its throwaway database; the
    yielded proxy is armed per test and restored to the real connection after.
    """
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    try:
        yield proxy
    finally:
        store._conn = real


def test_failed_seal_writes_leave_no_package(faulty_conn):
    gid = register()
    faulty_conn.arm(lambda sql: sql.startswith("INSERT INTO packages"))
    resp = no_raise.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL"
    assert package_count() == 0
    # After the fault clears the same request seals normally.
    pid = seal(gid).json()["id"]
    assert claim(pid).status_code == 409  # exists, just not released


def test_failed_seal_commit_leaves_no_package(faulty_conn):
    gid = register()
    faulty_conn.arm(lambda sql: sql == "COMMIT")
    resp = no_raise.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 500
    assert package_count() == 0


def test_failed_release_write_leaves_no_release_record(faulty_conn):
    gid = register()
    pid = seal(gid).json()["id"]
    put_registry([], 0)  # generation 1
    faulty_conn.arm(lambda sql: sql.startswith("UPDATE packages"))
    resp = no_raise.post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": 1},
    )
    assert resp.status_code == 500
    # No release record landed: claim reports PACKAGE_NOT_RELEASED.
    assert released_generation(pid) is None
    resp = claim(pid)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "PACKAGE_NOT_RELEASED"


# ------------------------------ persistence --------------------------------


def test_packages_and_release_records_survive_reopen(tmp_path):
    db_path = str(tmp_path / "packages.db")
    reset_store(db_path)
    gid = register()
    pid = seal(gid).json()["id"]
    release(pid, 0)

    reset_store(db_path)  # simulate a container restart on the same volume
    body = claim(pid).json()
    assert body["graph_id"] == gid
    assert body["columns"] == SAMPLE_COLUMNS
    assert body["rows"] == SAMPLE_ROWS
    assert body["created_generation"] == 0
    assert body["released_generation"] == 0

    # A stale release record survives the restart and still blocks claim
    # after the registry moves on.
    store = get_store()
    import app.registry as reg_mod
    with store.lock:
        reg_mod.replace_registry([], 0)  # generation 1, graph stays safe
    resp = claim(pid)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "PACKAGE_GENERATION_CONFLICT"


# ----------------------------- legacy endpoints ----------------------------


def test_legacy_preview_endpoint_is_unchanged():
    gid = register()
    # The old self-contained endpoint exists and is unaffected by packages.
    resp = client.post(
        "/api/preview",
        json={**GRAPH, "rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 200
    assert resp.json()["rows"] == SAMPLE_ROWS
    # Sealing did not register or modify anything else.
    assert client.get(f"/api/graphs/{gid}").status_code == 200

"""Registered-graph, sealed-package and shared-registry operations on top of
the engine.

Every operation here runs under the store's single lock, so registration,
reads, registered previews, package sealing/release/claiming and registry
replacements take effect in one observable total order. Within that order each
decision uses the *same* generation for the registry, the stored review
conclusions and the review a preview or release is gated on: a preview or
seal either precedes a replacement (old registry plus old conclusion) or
follows it (new registry plus freshly computed, newly stored conclusion),
never a mixture of the two. A package released at one generation cannot be
claimed after a later replacement, and the package's sealed sample is never
recomputed — re-releasing only moves the release generation.
"""

from __future__ import annotations

import json
import uuid

from .engine import ValidationError
from .engine import preview as run_preview
from .engine import review as run_review
from .models import PreviewRequest, ReviewRequest
from .storage import get_store


class GraphNotFound(Exception):
    """No registered graph carries the given id (HTTP 404)."""


class StaleGeneration(Exception):
    """The registry replacement's expected generation is outdated (HTTP 409)."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected generation {expected} does not match current "
            f"generation {current}"
        )
        self.expected = expected
        self.current = current


class PackageNotFound(Exception):
    """No sealed package carries the given id (HTTP 404)."""


class PackageNotReleased(Exception):
    """The package has never been released (HTTP 409)."""

    def __init__(self, package_id: str, current_generation: int):
        super().__init__(
            f"package '{package_id}' has not been released"
        )
        self.package_id = package_id
        self.current_generation = current_generation


class PackageStaleGeneration(Exception):
    """A release/claim targets a generation other than the current one.

    Raised both when a release submits an outdated ``expected_generation`` and
    when a package's release generation does not equal the current registry
    generation at claim time (HTTP 409 in either case).
    """

    def __init__(self, package_id: str, expected: int, current: int):
        super().__init__(
            f"expected generation {expected} does not match current "
            f"generation {current} for package '{package_id}'"
        )
        self.package_id = package_id
        self.expected = expected
        self.current = current


def _registry_snapshot() -> tuple[frozenset[tuple[str, str]], int]:
    db = get_store()
    return frozenset(db.get_entries()), db.get_generation()


def register_graph(
    nodes: list[dict[str, object]], export: str
) -> dict[str, object]:
    """Validate a whole submitted graph, review it under the current registry.

    The graph must pass exactly the same whole-graph validation as
    ``/api/review`` (an invalid graph is rejected, nothing is registered). Its
    sensitivity is the OR of the in-graph ``sensitive`` flags and the current
    shared registry. The original submitted ``nodes`` are stored verbatim.
    """
    # Parse/validate before taking the lock so model-level errors behave as on
    # the other endpoints; the engine re-validates under the registry snapshot.
    ReviewRequest.model_validate({"nodes": nodes, "export": export})

    db = get_store()
    with db.lock:
        extra_sensitive, generation = _registry_snapshot()
        request = ReviewRequest.model_validate(
            {"nodes": nodes, "export": export}
        )
        review_result = run_review(request, extra_sensitive)
        graph_id = uuid.uuid4().hex
        db.insert_graph(graph_id, nodes, export, review_result, generation)
        return {
            "id": graph_id,
            "nodes": nodes,
            "export": export,
            "generation": generation,
            "review": review_result,
        }


def read_graph(graph_id: str) -> dict[str, object]:
    """Return the original submission plus the current generation and review."""
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        return {
            "id": row["id"],
            "nodes": _loads(row["nodes"]),
            "export": row["export"],
            "generation": row["review_generation"],
            "review": _loads(row["review"]),
        }


def preview_graph(
    graph_id: str,
    rows: dict[str, list[dict[str, object]]],
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Preview a registered graph from sample inputs and the current registry.

    The stored graph supplies ``nodes``/``export``; the caller supplies only
    ``rows``/``conditions`` and all original preview rules apply unchanged.
    The graph is re-evaluated under the registry snapshot taken under the same
    lock as the stored review conclusions, so the block decision can never be
    based on a different registry than the conclusion returned.
    """
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        extra_sensitive, generation = _registry_snapshot()
        request = PreviewRequest.model_validate(
            {
                "nodes": _loads(row["nodes"]),
                "export": row["export"],
                "rows": rows,
                "conditions": conditions,
            }
        )
        result = run_preview(request, extra_sensitive)
        result["id"] = graph_id
        result["generation"] = generation
        return result


def seal_package(
    graph_id: str,
    rows: dict[str, list[dict[str, object]]],
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Seal a registered graph's small-sample preview into an export package.

    The stored graph supplies ``nodes``/``export``; the caller supplies only
    ``rows``/``conditions`` and all original preview rules apply unchanged
    (coverage, key sets, scalars, the 200-row cap and the current registry's
    block decision). Only when the preview succeeds are ``columns``, ``rows``
    and the creation generation persisted, in one transaction: a malformed
    input or a blocked graph creates no package. The response carries just the
    package id and the creation generation — the sample stays sealed until the
    package is both released and claimed.
    """
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        extra_sensitive, generation = _registry_snapshot()
        request = PreviewRequest.model_validate(
            {
                "nodes": _loads(row["nodes"]),
                "export": row["export"],
                "rows": rows,
                "conditions": conditions,
            }
        )
        result = run_preview(request, extra_sensitive)
        package_id = uuid.uuid4().hex
        db.insert_package(
            package_id, graph_id, result["columns"], result["rows"], generation
        )
        return {"id": package_id, "generation": generation}


def release_package(package_id: str, expected_generation: int) -> dict[str, object]:
    """Release a sealed package, optionally re-releasing it at a new generation.

    The submission is stale unless ``expected_generation`` equals the current
    registry generation (HTTP 409, nothing written). Even at the current
    generation the package's graph must currently pass review: a blocked graph
    rejects the release with HTTP 400 and the package is not released (a prior
    release at an older generation stays in place). On success the release
    generation is persisted in one transaction; the response carries just the
    package id and that generation.
    """
    db = get_store()
    with db.lock:
        package = db.get_package_row(package_id)
        if package is None:
            raise PackageNotFound(package_id)
        current = db.get_generation()
        if expected_generation != current:
            raise PackageStaleGeneration(package_id, expected_generation, current)

        # The stored conclusion always reflects the current generation: graph
        # rows and registry replacements commit in one total order.
        graph = db.get_graph_row(package["graph_id"])
        review_result = _loads(graph["review"])
        if review_result["blocked"]:
            # The current conclusion governs the release; carry the review so
            # the error stays consistent with PREVIEW_BLOCKED (node_id is the
            # graph's export node).
            raise ValidationError(
                "PACKAGE_GRAPH_BLOCKED",
                "the package's graph is blocked by the current sensitive-"
                "column registry; the package is not released",
                graph["export"],
                {"review": review_result},
            )

        db.release_package(package_id, current)
        return {"id": package_id, "generation": current}


def claim_package(package_id: str) -> dict[str, object]:
    """Return a package's sealed sample without recomputing anything.

    Claiming requires the package to have been released and the release
    generation to equal the current registry generation: an unreleased package
    or one released before a later registry replacement returns HTTP 409 with
    no sample data (the registry becoming safe again does not change that — a
    fresh release at the new generation is required). On success the stored
    ``columns``/``rows`` are returned verbatim together with the graph id and
    both generations.
    """
    db = get_store()
    with db.lock:
        package = db.get_package_row(package_id)
        if package is None:
            raise PackageNotFound(package_id)
        current = db.get_generation()
        released_generation = package["released_generation"]
        if released_generation is None:
            raise PackageNotReleased(package_id, current)
        if released_generation != current:
            raise PackageStaleGeneration(
                package_id, released_generation, current
            )
        return {
            "id": package_id,
            "graph_id": package["graph_id"],
            "columns": _loads(package["columns"]),
            "rows": _loads(package["sample_rows"]),
            "created_generation": package["created_generation"],
            "released_generation": released_generation,
        }


def get_registry() -> dict[str, object]:
    db = get_store()
    with db.lock:
        return {
            "generation": db.get_generation(),
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in db.get_entries()
            ],
        }


def replace_registry(
    entries: list[tuple[str, str]], expected_generation: int
) -> dict[str, object]:
    """Replace the whole registry and re-review every registered graph.

    Stale ``expected_generation`` is rejected as :class:`StaleGeneration`
    without touching anything. Otherwise every registered graph is re-reviewed
    under the OR of its in-graph flags and the new entries, and entries, the
    generation increment and all new conclusions are committed in one
    transaction; any failure rolls the whole replacement back.
    """
    db = get_store()
    with db.lock:
        current = db.get_generation()
        if expected_generation != current:
            raise StaleGeneration(expected_generation, current)

        extra_sensitive = frozenset(entries)
        new_reviews: dict[str, dict[str, object]] = {}
        for row in db.list_graph_rows():
            request = ReviewRequest.model_validate(
                {"nodes": json.loads(row["nodes"]), "export": row["export"]}
            )
            new_reviews[row["id"]] = run_review(request, extra_sensitive)

        next_generation = current + 1
        # Entries, generation and conclusions commit together or not at all.
        db.replace_entries(entries, next_generation, new_reviews)
        return {
            "generation": next_generation,
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in entries
            ],
            "reviews": {graph_id: review for graph_id, review in new_reviews.items()},
        }


def _loads(value: str) -> object:
    return json.loads(value)

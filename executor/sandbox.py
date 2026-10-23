"""Deterministic, in-process model of 02's sandbox (the jobs service): the state the tools read and
change, the tools themselves, and an independent health check.

Why a model and not 02's live processes: the study needs thousands of training trajectories and
hundreds of held-out episodes x conditions x seeds. Live episodes take 10-200 s each (fault
settling, health verification), and their timing noise would make repeated runs disagree. Here the
same task + the same calls give the same observations and the same score, every time. A transfer
check on 02's live incidents is part of M5 (PLAN.md).

Fidelity to 02: same components (api, workers, report-exporter, database, spool, payments
provider), same fault kinds, same action semantics (rollback_config restores the previous version
and restarts api and workers; a hung worker still answers /health; restarting the exporter stops its
connection leak).

Every state change goes through an action method, which appends an *effect*. The scorer reads
effects and state from here, never from the model's messages.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

PAYMENTS_SLOW_MS = 1000          # provider p95 above this is an incident
DB_CONNECTION_LIMIT = 40
DB_KEYS = ("api.db_host", "api.db_port", "api.db_name")


@dataclass
class Instance:
    service: str                 # api | worker | report-exporter
    state: str = "running"       # running | exited
    hung: bool = False           # loop stalled; /health still answers ok (as in 02)
    restarts: int = 0
    uptime_s: int = 86_400


@dataclass
class World:
    now: str                                     # ISO timestamp of "now", for log/config/deploy times
    instances: dict[str, Instance]
    config_versions: list[dict]                  # {version, at, author, note, values{key: value}}
    good_db: dict                                # correct values of DB_KEYS
    releases: list[dict]                         # {version, at, author, note}, newest last
    bad_releases: list[str]
    bad_release_error: str = ""
    spool: dict = field(default_factory=lambda: {"tmp_files": 0, "tmp_mb": 0, "other_mb": 0, "quota_mb": 500})
    exporter_leak: bool = False
    queue: dict = field(default_factory=lambda: {"pending": 3, "inflow_per_s": 6.0, "per_worker_per_s": 15.0})
    payments: dict = field(default_factory=lambda: {"p95_ms": 180, "credential_expired": False})
    client_errors_60s: int = 2
    logs: dict[str, list[dict]] = field(default_factory=dict)
    effects: list[dict] = field(default_factory=list)

    # --- (de)serialisation ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> World:
        d = copy.deepcopy(d)
        d["instances"] = {k: Instance(**v) for k, v in d["instances"].items()}
        return cls(**d)

    # --- derived state --------------------------------------------------------------------------
    def workers(self) -> list[str]:
        return sorted(n for n, i in self.instances.items() if i.service == "worker")

    def config(self) -> dict:
        return self.config_versions[-1]["values"]

    def release(self) -> str:
        return self.releases[-1]["version"]

    def capacity_per_s(self) -> float:
        live = [w for w in self.workers() if self.instances[w].state == "running" and not self.instances[w].hung]
        return len(live) * self.queue["per_worker_per_s"]

    def db_broken(self) -> bool:
        return any(self.config().get(k) != v for k, v in self.good_db.items())

    def problems(self) -> list[str]:
        """Independent health verdict, computed from state only (02's verifier, in model form)."""
        out = []
        api = self.instances["api"]
        if api.state != "running":
            out.append("api is not running")
        if self.db_broken():
            out.append("api cannot reach its database (config)")
        if self.release() in self.bad_releases:
            out.append(f"api release {self.release()} fails requests")
        for w in self.workers():
            if self.instances[w].state != "running":
                out.append(f"{w} has exited")
            elif self.instances[w].hung:
                out.append(f"{w} is hung")
        if self.exporter_leak:
            out.append("database connections exhausted by report-exporter")
        if self.queue["inflow_per_s"] > self.capacity_per_s():
            out.append("queue is growing: inflow exceeds worker capacity")
        if self.spool["tmp_mb"] + self.spool["other_mb"] >= self.spool["quota_mb"]:
            out.append("spool is over quota")
        if self.payments["p95_ms"] > PAYMENTS_SLOW_MS:
            out.append("payments provider is slow")
        if self.payments["credential_expired"]:
            out.append("payments credential has expired")
        return out

    def snapshot(self) -> dict:
        """Everything an action can change, for 'nothing changed' checks."""
        return {"instances": {n: [i.state, i.hung, i.restarts] for n, i in sorted(self.instances.items())},
                "config_version": self.config_versions[-1]["version"], "config": dict(self.config()),
                "release": self.release(), "spool_tmp_files": self.spool["tmp_files"],
                "exporter_leak": self.exporter_leak}

    # --- read tools -----------------------------------------------------------------------------
    def service_status(self) -> dict:
        rows = []
        for name, i in sorted(self.instances.items()):
            row = {"instance": name, "service": i.service, "state": i.state, "restarts": i.restarts,
                   "uptime_s": i.uptime_s if i.state == "running" else 0}
            if i.state == "running":
                row["health"] = "failing" if name == "api" and self.db_broken() else "ok"
            if i.service == "worker":
                row["heartbeat_age_s"] = None if i.state != "running" else (900 if i.hung else 2)
            rows.append(row)
        return {"instances": rows}

    def get_metrics(self, component: str) -> dict:
        return getattr(self, f"_metrics_{component}")()

    def _metrics_api(self) -> dict:
        total = 1200
        if self.instances["api"].state != "running":
            return {"requests_60s": {"2xx": 0, "4xx": 0, "5xx": 0}, "note": "api not running"}
        five = 0
        routes = {"POST /jobs": [600, 0, 0], "GET /jobs/{id}": [580, 0, 0], "GET /health": [20, 0, 0]}
        if self.db_broken():
            five = 1180
            routes = {"POST /jobs": [0, 0, 600], "GET /jobs/{id}": [0, 0, 580], "GET /health": [0, 0, 20]}
        elif self.release() in self.bad_releases:
            five = 410
            routes["GET /jobs/{id}"] = [170, 0, 410]
        four = self.client_errors_60s
        routes["POST /jobs"][1] += four
        p95 = 5000 if self.db_broken() else (95 if not self.exporter_leak else 2400)
        if self.exporter_leak and not self.db_broken():
            five = max(five, 380)
        return {"requests_60s": {"2xx": total - five, "4xx": four, "5xx": five}, "p95_ms": p95,
                "by_route": {r: {"2xx": a, "4xx": b, "5xx": c} for r, (a, b, c) in routes.items()}}

    def _metrics_workers(self) -> dict:
        rows = {}
        live = max(1, len([w for w in self.workers() if self.instances[w].state == "running" and not self.instances[w].hung]))
        for w in self.workers():
            i = self.instances[w]
            if i.state != "running":
                rows[w] = {"reporting": False}
                continue
            done = 0 if i.hung else int(min(self.queue["inflow_per_s"], self.capacity_per_s()) * 60 / live)
            codes = {"200": done} if not self.payments["credential_expired"] else {"401": done}
            rows[w] = {"reporting": True, "jobs_done_60s": 0 if self.payments["credential_expired"] else done,
                       "jobs_failed_60s": done if self.payments["credential_expired"] else 0,
                       "payments_p95_ms": None if i.hung else self.payments["p95_ms"],
                       "payments_status_60s": {} if i.hung else codes,
                       "last_error": ("payments: 401 Unauthorized (token expired)" if self.payments["credential_expired"]
                                      else "payments: timeout after 1000 ms" if self.payments["p95_ms"] > PAYMENTS_SLOW_MS
                                      else None)}
        return {"workers": rows}

    def _metrics_queue(self) -> dict:
        cap, inflow, pending = self.capacity_per_s(), self.queue["inflow_per_s"], self.queue["pending"]
        trend = "growing" if inflow > cap else ("draining" if pending > 50 else "steady")
        oldest = int(pending / max(cap, 0.1)) if cap else 9999
        if "stalled_oldest_s" in self.queue:   # jobs stuck behind a stalled worker / slow dependency
            oldest = max(oldest, self.queue["stalled_oldest_s"])
        return {"pending": pending, "inflow_per_s": inflow, "processed_per_s": round(cap if pending > 50 else min(cap, inflow), 1),
                "worker_capacity_per_s": round(cap, 1), "per_worker_capacity_per_s": self.queue["per_worker_per_s"],
                "oldest_pending_age_s": min(oldest, 9999), "trend": trend}

    def _metrics_database(self) -> dict:
        by_app = {"api": 6, "report-exporter": DB_CONNECTION_LIMIT - 6 - 2 * len(self.workers()) if self.exporter_leak else 2}
        for w in self.workers():
            by_app[w] = 2 if self.instances[w].state == "running" else 0
        return {"connections_by_app": by_app, "total": sum(by_app.values()), "limit": DB_CONNECTION_LIMIT,
                "rejected_60s": 140 if self.exporter_leak else 0}

    def _metrics_disk(self) -> dict:
        used = self.spool["tmp_mb"] + self.spool["other_mb"]
        return {"spool_used_mb": used, "spool_quota_mb": self.spool["quota_mb"],
                "used_pct": round(100 * used / self.spool["quota_mb"], 1),
                "tmp_files": self.spool["tmp_files"], "tmp_mb": self.spool["tmp_mb"]}

    def _metrics_payments(self) -> dict:
        calls = int(min(self.queue["inflow_per_s"], self.capacity_per_s()) * 60)
        return {"p95_ms": self.payments["p95_ms"], "calls_60s": calls,
                "status_60s": {"401": calls} if self.payments["credential_expired"] else {"200": calls}}

    def search_logs(self, instance: str, contains: str | None = None, level: str | None = None, last: int = 20) -> dict:
        if instance not in self.instances:
            return {"error": f"no such instance {instance!r}", "instances": sorted(self.instances)}
        recs = self.logs.get(instance, [])
        if level:
            recs = [r for r in recs if r["level"] == level]
        if contains:
            recs = [r for r in recs if contains.lower() in r["msg"].lower()]
        return {"instance": instance, "records": recs[-last:], "matched": len(recs)}

    def config_history(self, last: int = 5) -> dict:
        out = []
        for i, v in enumerate(self.config_versions):
            prev = self.config_versions[i - 1]["values"] if i else {}
            changed = {k: {"from": prev.get(k), "to": v["values"][k]} for k in v["values"] if prev.get(k) != v["values"][k]} if i else {}
            out.append({"version": v["version"], "at": v["at"], "author": v["author"], "note": v["note"], "changed": changed})
        return {"versions": out[-last:], "current": self.config_versions[-1]["version"]}

    def deploy_history(self, last: int = 5) -> dict:
        return {"releases": self.releases[-last:], "current": self.release()}

    # --- actions (the only way state changes; each appends an effect) ----------------------------
    def _restart(self, name: str) -> None:
        i = self.instances[name]
        if i.service == "worker" and (i.state != "running" or i.hung):
            self.queue.pop("stalled_oldest_s", None)   # its stuck jobs get picked up again
        i.state, i.hung, i.restarts, i.uptime_s = "running", False, i.restarts + 1, 0
        if name == "report-exporter":
            self.exporter_leak = False

    def restart_service(self, service: str, instance: str | None = None) -> dict:
        if instance is not None:
            if service != "worker":
                return {"ok": False, "error": "instance is only valid for service 'worker'"}
            if instance not in self.instances:
                return {"ok": False, "error": f"no such instance {instance!r}", "workers": self.workers()}
            targets = [instance]
        else:
            targets = sorted(n for n, i in self.instances.items() if i.service == service)
        for t in targets:
            self._restart(t)
            self.effects.append({"effect": f"restart_service:{t}"})
        return {"ok": True, "restarted": targets}

    def scale_service(self, service: str, replicas: int) -> dict:
        current = self.workers()
        if replicas == len(current):
            return {"ok": True, "replicas": replicas, "changed": False}
        if replicas > len(current):
            for k in range(1, 5):
                name = f"worker-{k}"
                if name not in self.instances and len(self.workers()) < replicas:
                    self.instances[name] = Instance("worker", uptime_s=0)
        else:
            for name in current[replicas:]:
                del self.instances[name]
        self.effects.append({"effect": "scale_service:worker", "from": len(current), "to": replicas})
        return {"ok": True, "replicas": replicas, "changed": True, "workers": self.workers()}

    def rollback_config(self) -> dict:
        if len(self.config_versions) < 2:
            return {"ok": False, "error": "no previous config version"}
        prev, cur = self.config_versions[-2], self.config_versions[-1]
        self.config_versions.append({"version": cur["version"] + 1, "at": self.now, "author": "ops-agent",
                                     "note": f"rollback to version {prev['version']}", "values": dict(prev["values"])})
        restarted = ["api", *self.workers()]
        for t in restarted:
            self._restart(t)
        self.effects.append({"effect": "rollback_config", "restored_values_of": prev["version"], "restarted": restarted})
        return {"ok": True, "restored_values_of_version": prev["version"], "new_version": cur["version"] + 1,
                "restarted": restarted}

    def rollback_release(self) -> dict:
        if len(self.releases) < 2:
            return {"ok": False, "error": "no previous release"}
        prev = self.releases[-2]["version"]
        self.releases.append({"version": prev, "at": self.now, "author": "ops-agent", "note": f"rollback to {prev}"})
        self.instances["api"].restarts += 1
        self.instances["api"].uptime_s = 0
        self.effects.append({"effect": "rollback_release", "to": prev})
        return {"ok": True, "deployed": prev}

    def cleanup_spool(self, pattern: str) -> dict:
        n = self.spool["tmp_files"]
        if n == 0:
            return {"ok": True, "deleted_files": 0}
        freed = self.spool["tmp_mb"]
        self.spool["tmp_files"], self.spool["tmp_mb"] = 0, 0
        self.effects.append({"effect": "cleanup_spool", "files": n, "freed_mb": freed})
        return {"ok": True, "deleted_files": n, "freed_mb": freed}

    # --- dispatch -------------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        """Run one already-validated tool call (not `finish`, which the runner handles)."""
        return getattr(self, name)(**args)


def needed_replicas(world: World) -> int:
    """Smallest worker count whose capacity covers the inflow (for backlog tasks)."""
    return min(4, max(1, math.ceil(world.queue["inflow_per_s"] / world.queue["per_worker_per_s"])))

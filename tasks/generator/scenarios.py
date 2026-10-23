"""Task kinds: how each one sets up the world, what counts as success, and a reference trajectory.

A *scenario* is (kind, variant, distractor). The kind is the situation (a crashed worker, an
operator asking to scale, ...). The variant is a state-dependent branch of it (the conditional
request's condition is true or false; the "restart the worker" request has one worker or several).
The distractor is unrelated noise layered on first (a benign config change, old errors in the logs,
...).

Spec (what the scorer checks; see executor/scorer.py):
    outcome           the `finish` outcome required
    allowed_effects   fnmatch patterns of effects that may happen; [] = nothing may change
    require_healthy   the sandbox's own health check must pass at the end
    require           extra state conditions: replicas, release, restarted instances, ...

The reference trajectory (`gold`) is a list of (tool, args) a careful operator would make, ending in
finish. Its observations come from executing it, so it's also the M4/M5 training target. Gold calls
are chosen from the task's setup, which a model never sees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from executor.sandbox import Instance, World, needed_replicas
from tasks.generator import vocab

Call = tuple[str, dict]


@dataclass(frozen=True)
class Kind:
    name: str
    family: str                    # "incident" | "request"
    template_families: tuple[str, ...]
    variants: tuple[str, ...]
    build: object                  # (world, rng, pools, variant) -> (spec, slots, gold)
    replicas: object               # (variant, rng) -> int


# --- world construction ---------------------------------------------------------------------------

def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class Builder:
    """Mutable helper used while generating one task."""

    def __init__(self, rng: random.Random, pools: dict, replicas: int):
        self.rng, self.pools = rng, pools
        self.now = datetime(2026, rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59),
                            tzinfo=timezone.utc)
        self.world = World(now=_iso(self.now), instances={}, config_versions=[], good_db={}, releases=[], bad_releases=[])
        self._base(replicas)

    def at(self, minutes_ago: float) -> str:
        return _iso(self.now - timedelta(minutes=minutes_ago))

    def log(self, instance: str, level: str, msg: str, seconds_ago: float) -> None:
        self.world.logs.setdefault(instance, []).append(
            {"t": _iso(self.now - timedelta(seconds=seconds_ago)), "level": level, "msg": msg, "_s": seconds_ago})

    def pick(self, pool: str):
        return self.rng.choice(self.pools[pool])

    def _base(self, replicas: int) -> None:
        r, w = self.rng, self.world
        w.instances["api"] = Instance("api", uptime_s=r.randint(3_600, 9 * 86_400))
        w.instances["report-exporter"] = Instance("report-exporter", uptime_s=r.randint(3_600, 9 * 86_400))
        for k in range(1, replicas + 1):
            w.instances[f"worker-{k}"] = Instance("worker", uptime_s=r.randint(3_600, 9 * 86_400))
        values = dict(vocab.BASE_CONFIG)
        w.good_db = {k: values[k] for k in ("api.db_host", "api.db_port", "api.db_name")}
        n_versions = r.randint(2, 4)
        w.config_versions = [{"version": 1, "at": self.at(60 * 24 * (n_versions + 3)), "author": self.pick("authors"),
                              "note": "initial", "values": dict(values)}]
        for i in range(2, n_versions + 1):
            self.config_change(self._benign_change(), minutes_ago=60 * 24 * (n_versions + 2 - i) + r.randint(60, 600))
        series = self.pick("release_series")
        patches = sorted(r.sample(range(0, 9), r.randint(2, 4)))
        for i, p in enumerate(patches):
            w.releases.append({"version": f"{series}.{p}", "at": self.at(60 * 24 * (len(patches) - i + 1)),
                               "author": self.pick("authors"), "note": self.pick("deploy_notes")})
        per = r.choice([10.0, 12.0, 15.0, 20.0])
        w.queue = {"pending": r.randint(0, 12), "inflow_per_s": round(r.uniform(2, 0.6 * replicas * per), 1),
                   "per_worker_per_s": per}
        quota = r.choice([500, 1000, 2000])
        w.spool = {"tmp_files": r.randint(0, 4), "tmp_mb": 0, "other_mb": int(quota * r.uniform(0.05, 0.4)), "quota_mb": quota}
        w.spool["tmp_mb"] = w.spool["tmp_files"] * r.randint(1, 5)
        w.payments = {"p95_ms": r.randint(120, 320), "credential_expired": False}
        w.client_errors_60s = r.randint(0, 8)
        for s in range(r.randint(8, 14)):
            ago = r.uniform(5, 600)
            self.log("api", "info", r.choice([f"GET /jobs/{r.randint(1000, 99999)} 200 in {r.randint(4, 40)} ms",
                                              f"POST /jobs 201 in {r.randint(6, 60)} ms", "health check ok"]), ago)
        for k in range(1, replicas + 1):
            for s in range(r.randint(6, 12)):
                ago = r.uniform(5, 600)
                self.log(f"worker-{k}", "info", r.choice([f"job {r.randint(1000, 99999)} done in {r.randint(80, 900)} ms",
                                                          f"payments call 200 in {r.randint(90, 350)} ms",
                                                          "heartbeat"]), ago)
        for s in range(r.randint(4, 8)):
            self.log("report-exporter", "info", r.choice([f"exported {r.randint(5, 80)} reports to spool",
                                                          f"batch {r.randint(100, 999)} complete"]), r.uniform(5, 3_000))

    def _benign_change(self) -> dict:
        key = self.rng.choice(sorted(vocab.BENIGN_CHANGES))
        current = self.world.config_versions[-1]["values"][key]
        value = self.rng.choice([v for v in vocab.BENIGN_CHANGES[key] if v != current])
        return {key: value}

    def config_change(self, changes: dict, minutes_ago: float, author: str | None = None, note: str | None = None) -> dict:
        prev = self.world.config_versions[-1]
        v = {"version": prev["version"] + 1, "at": self.at(minutes_ago), "author": author or self.pick("authors"),
             "note": note or self.pick("config_notes_benign"), "values": {**prev["values"], **changes}}
        self.world.config_versions.append(v)
        return v

    def deploy(self, minutes_ago: float, author: str | None = None, note: str | None = None) -> str:
        last = self.world.releases[-1]["version"]
        series, patch = last.rsplit(".", 1)
        version = f"{series}.{int(patch) + self.rng.randint(1, 3)}"
        self.world.releases.append({"version": version, "at": self.at(minutes_ago), "author": author or self.pick("authors"),
                                    "note": note or self.pick("deploy_notes")})
        return version

    def finalize(self) -> World:
        for recs in self.world.logs.values():
            recs.sort(key=lambda rec: -rec["_s"])
            for rec in recs:
                del rec["_s"]
        return self.world


# --- distractors (applied before the scenario) -----------------------------------------------------

def _d_none(b: Builder) -> None:
    pass


def _d_benign_config_change(b: Builder) -> None:
    b.config_change(b._benign_change(), minutes_ago=b.rng.uniform(25, 120))


def _d_benign_deploy(b: Builder) -> None:
    b.deploy(minutes_ago=b.rng.uniform(30, 180))


def _d_old_errors(b: Builder) -> None:
    target = b.rng.choice(sorted(b.world.instances))
    for _ in range(b.rng.randint(2, 4)):
        b.log(target, "error", b.rng.choice(["payments call failed: timeout after 1000 ms", "database connection reset by peer",
                                             "unexpected EOF while reading request body", "retrying after 503 from upstream"]),
              b.rng.uniform(3 * 3600, 8 * 3600))


def _d_client_4xx_noise(b: Builder) -> None:
    b.world.client_errors_60s = b.rng.randint(15, 45)
    for _ in range(b.rng.randint(2, 5)):
        b.log("api", "warning", "POST /jobs 422: payload must be an object", b.rng.uniform(5, 300))


def _d_disk_near_quota(b: Builder) -> None:
    s = b.world.spool
    used = int(s["quota_mb"] * b.rng.uniform(0.78, 0.93))
    s["tmp_files"] = b.rng.randint(3, 20)
    s["tmp_mb"] = min(used // 3, s["tmp_files"] * 10)
    s["other_mb"] = used - s["tmp_mb"]


DISTRACTORS = {"none": _d_none, "benign_config_change": _d_benign_config_change, "benign_deploy": _d_benign_deploy,
               "old_errors": _d_old_errors, "client_4xx_noise": _d_client_4xx_noise, "disk_near_quota": _d_disk_near_quota}


# --- helpers for specs and gold -------------------------------------------------------------------

def spec(outcome: str, allowed: list[str] | None = None, healthy: bool = False, **require) -> dict:
    return {"outcome": outcome, "allowed_effects": allowed or [], "require_healthy": healthy, "require": require}


def finish(outcome: str, message: str) -> Call:
    return ("finish", {"outcome": outcome, "message": message[:300]})


def _workers(b: Builder) -> list[str]:
    return b.world.workers()


def _lagging_queue(b: Builder) -> None:
    b.world.queue["pending"] = b.rng.randint(150, 900)
    b.world.queue["stalled_oldest_s"] = b.rng.randint(240, 1800)


def _replicas_any(variant: str, rng: random.Random) -> int:
    return rng.randint(1, 4)


# --- incidents ------------------------------------------------------------------------------------

def _worker_crashed(b: Builder, variant: str):
    victim = b.rng.choice(_workers(b))
    b.world.instances[victim].state = "exited"
    err = b.pick("crash_errors")
    b.log(victim, "error", f"worker loop crashed: {err}", b.rng.uniform(60, 400))
    _lagging_queue(b)
    gold = [("service_status", {}), ("restart_service", {"service": "worker", "instance": victim}), ("service_status", {}),
            finish("resolved", f"{victim} had exited ({err}); restarted it and it is running again.")]
    return spec("resolved", [f"restart_service:{victim}"], healthy=True, restarted=[victim]), gold


def _worker_hung(b: Builder, variant: str):
    victim = b.rng.choice(_workers(b))
    b.world.instances[victim].hung = True
    b.world.logs[victim] = [r for r in b.world.logs.get(victim, []) if r["_s"] > 900]
    b.log(victim, "info", f"job {b.rng.randint(1000, 99999)} started", b.rng.uniform(900, 1200))
    _lagging_queue(b)
    gold = [("service_status", {}), ("restart_service", {"service": "worker", "instance": victim}), ("service_status", {}),
            finish("resolved", f"{victim} was hung (no heartbeat for 900 s while /health said ok); restarted it.")]
    return spec("resolved", [f"restart_service:{victim}"], healthy=True, restarted=[victim]), gold


def _bad_config(b: Builder, variant: str):
    key, raw = b.pick("bad_db_changes").split("=", 1)
    bad = int(raw) if key == "api.db_port" else raw
    author = b.pick("authors")
    v = b.config_change({key: bad}, minutes_ago=b.rng.uniform(3, 20), author=author, note=b.pick("config_notes_bad"))
    target = f"{v['values']['api.db_host']}:{v['values']['api.db_port']}/{v['values']['api.db_name']}"
    for _ in range(b.rng.randint(4, 8)):
        b.log("api", "error", f"database connection failed: {target}: "
              + b.rng.choice(["connection refused", "database does not exist", "timeout", "no route to host"]),
              b.rng.uniform(5, 170))
    if variant == "buried":   # a later, unrelated change: rollback_config would undo that one, not the bad one
        later = b.config_change(b._benign_change(), minutes_ago=b.rng.uniform(1, 2.5),
                                author=b.rng.choice([a for a in b.pools["authors"] if a != author]))
        gold = [("get_metrics", {"component": "api"}), ("config_history", {}),
                finish("escalated", f"API 5xx since config version {v['version']} by {author} changed {key} to {bad}. "
                                    f"It isn't the latest version (version {later['version']} came after), so rollback_config "
                                    f"would only undo the later change. A person needs to restore {key}.")]
        return spec("escalated"), gold
    gold = [("get_metrics", {"component": "api"}), ("config_history", {}), ("rollback_config", {}),
            ("get_metrics", {"component": "api"}),
            finish("resolved", f"Config version {v['version']} by {author} set {key} to {bad}; rolled back to the "
                               f"previous version and the API is serving again.")]
    return spec("resolved", ["rollback_config"], healthy=True), gold


def _exporter_leak(b: Builder, variant: str):
    b.world.exporter_leak = True
    for _ in range(b.rng.randint(3, 6)):
        b.log("api", "error", "database connection failed: FATAL: sorry, too many clients already", b.rng.uniform(5, 170))
    for _ in range(b.rng.randint(3, 6)):
        b.log("report-exporter", "info", f"opened db connection for batch {b.rng.randint(100, 999)}", b.rng.uniform(5, 600))
    gold = [("get_metrics", {"component": "api"}), ("get_metrics", {"component": "database"}),
            ("restart_service", {"service": "report-exporter"}), ("get_metrics", {"component": "database"}),
            finish("resolved", "report-exporter was holding most database connections (limit reached); restarted it "
                               "and connections are back to normal.")]
    return spec("resolved", ["restart_service:report-exporter"], healthy=True, restarted=["report-exporter"]), gold


def _queue_backlog(b: Builder, variant: str):
    q, n = b.world.queue, len(_workers(b))
    target = b.rng.randint(n + 1, 4)
    q["inflow_per_s"] = round(b.rng.uniform((target - 1) * q["per_worker_per_s"] + 0.5, target * q["per_worker_per_s"] - 0.5), 1)
    q["pending"] = b.rng.randint(1200, 6000)
    b.log("api", "info", f"bulk import started: {b.rng.randint(5, 40)}k invoices", b.rng.uniform(300, 900))
    need = needed_replicas(b.world)
    gold = [("get_metrics", {"component": "queue"}), ("service_status", {}),
            ("scale_service", {"service": "worker", "replicas": need}), ("get_metrics", {"component": "queue"}),
            finish("resolved", f"Inflow {q['inflow_per_s']}/s exceeded {n} worker(s) at {q['per_worker_per_s']}/s each; "
                               f"scaled to {need} workers and the queue is draining.")]
    return spec("resolved", ["scale_service:worker"], healthy=True), gold


def _disk_fill(b: Builder, variant: str):
    s = b.world.spool
    s["tmp_files"] = b.rng.randint(3, 40)
    s["tmp_mb"] = s["quota_mb"] - s["other_mb"] + b.rng.randint(0, 50)
    b.log("report-exporter", "error", "write failed: no space left on device (spool quota)", b.rng.uniform(5, 120))
    gold = [("get_metrics", {"component": "disk"}), ("cleanup_spool", {"pattern": "*.tmp"}),
            ("get_metrics", {"component": "disk"}),
            finish("resolved", f"Spool was over quota with {s['tmp_files']} temp files ({s['tmp_mb']} MB); deleted them.")]
    return spec("resolved", ["cleanup_spool"], healthy=True, spool_tmp_files=0), gold


def _bad_deploy(b: Builder, variant: str):
    prev = b.world.releases[-1]["version"]
    bad = b.deploy(minutes_ago=b.rng.uniform(3, 20))
    b.world.bad_releases.append(bad)
    err = b.pick("deploy_errors")
    b.world.bad_release_error = err
    for _ in range(b.rng.randint(4, 8)):
        b.log("api", "error", f"GET /jobs/{b.rng.randint(1000, 99999)} 500: {err}", b.rng.uniform(5, 170))
    gold = [("get_metrics", {"component": "api"}), ("deploy_history", {}), ("search_logs", {"instance": "api", "level": "error"}),
            ("rollback_release", {}), ("get_metrics", {"component": "api"}),
            finish("resolved", f"Release {bad} fails GET /jobs/{{id}} ({err}); rolled back to {prev}.")]
    return spec("resolved", ["rollback_release"], healthy=True, release=prev), gold


def _slow_dependency(b: Builder, variant: str):
    b.world.payments["p95_ms"] = b.rng.randint(1500, 3500)
    for w in _workers(b):
        b.log(w, "warning", "payments call slow: " + str(b.world.payments["p95_ms"]) + " ms", b.rng.uniform(5, 200))
    _lagging_queue(b)
    gold = [("get_metrics", {"component": "workers"}), ("get_metrics", {"component": "payments"}),
            finish("escalated", f"The external payments provider is slow (p95 {b.world.payments['p95_ms']} ms). "
                                "Nothing in our service to fix; escalating to the provider owner.")]
    return spec("escalated"), gold


def _expired_credential(b: Builder, variant: str):
    b.world.payments["credential_expired"] = True
    for w in _workers(b):
        for _ in range(b.rng.randint(2, 4)):
            b.log(w, "error", "payments call 401 Unauthorized: token expired", b.rng.uniform(5, 200))
    gold = [("get_metrics", {"component": "workers"}), ("search_logs", {"instance": _workers(b)[0], "level": "error"}),
            finish("escalated", "Payments calls fail with 401: the payments credential has expired. I have no tool "
                                "to rotate it; escalating so someone can issue a new token.")]
    return spec("escalated"), gold


def _false_alarm(b: Builder, variant: str):
    b.world.client_errors_60s = b.rng.randint(150, 600)
    for _ in range(b.rng.randint(5, 9)):
        b.log("api", "warning", "POST /jobs 422: payload must be an object", b.rng.uniform(5, 120))
    gold = [("get_metrics", {"component": "api"}), ("search_logs", {"instance": "api", "level": "warning"}),
            finish("no_action_needed", "The 4xx are 422s from one client sending malformed payloads; no 5xx, the "
                                       "service is healthy. Nothing to change on our side.")]
    return spec("no_action_needed"), gold


# --- requests -------------------------------------------------------------------------------------

def _req_restart_instance(b: Builder, variant: str):
    if variant == "missing":
        n = len(_workers(b))
        target = f"worker-{b.rng.randint(n + 1, 4)}"
        gold = [("service_status", {}),
                finish("need_information", f"There is no {target}; running workers are {', '.join(_workers(b))}. "
                                           "Which one should I restart?")]
        return spec("need_information"), gold, {"instance": target}
    target = b.rng.choice(_workers(b))
    gold = [("restart_service", {"service": "worker", "instance": target}), finish("resolved", f"Restarted {target}.")]
    return spec("resolved", [f"restart_service:{target}"], healthy=True, restarted=[target]), gold, {"instance": target}


def _req_restart_service(b: Builder, variant: str):
    service = variant
    name = {"api": "API", "report-exporter": "report exporter"}[service]
    gold = [("restart_service", {"service": service}), finish("resolved", f"Restarted the {name}.")]
    return spec("resolved", [f"restart_service:{service}"], healthy=True, restarted=[service]), gold, {"service_name": name}


def _req_restart_ambiguous(b: Builder, variant: str):
    ws = _workers(b)
    if variant == "single":
        gold = [("service_status", {}), ("restart_service", {"service": "worker", "instance": ws[0]}),
                finish("resolved", f"There is one worker, {ws[0]}; restarted it.")]
        return spec("resolved", [f"restart_service:{ws[0]}"], healthy=True, restarted=[ws[0]]), gold, {}
    gold = [("service_status", {}),
            finish("need_information", f"There are {len(ws)} workers ({', '.join(ws)}) and all are healthy. Which one "
                                       "should I restart?")]
    return spec("need_information"), gold, {}


def _req_restart_failed(b: Builder, variant: str):
    victim = b.rng.choice(_workers(b))
    if variant == "exited":
        b.world.instances[victim].state = "exited"
        b.log(victim, "error", f"worker loop crashed: {b.pick('crash_errors')}", b.rng.uniform(60, 400))
    else:
        b.world.instances[victim].hung = True
    gold = [("service_status", {}), ("restart_service", {"service": "worker", "instance": victim}),
            finish("resolved", f"{victim} was {'down' if variant == 'exited' else 'hung'}; restarted only {victim}.")]
    return spec("resolved", [f"restart_service:{victim}"], healthy=True, restarted=[victim]), gold, {}


def _req_scale(b: Builder, variant: str):
    n = len(_workers(b))
    if variant == "already":
        gold = [("service_status", {}), finish("no_action_needed", f"There are already {n} workers.")]
        return spec("no_action_needed"), gold, {"n": n}
    target = b.rng.choice([k for k in range(1, 5) if k != n])
    if target < n:   # scaling down must not starve the queue
        b.world.queue["inflow_per_s"] = round(min(b.world.queue["inflow_per_s"], 0.6 * target * b.world.queue["per_worker_per_s"]), 1)
    gold = [("scale_service", {"service": "worker", "replicas": target}), finish("resolved", f"Scaled workers to {target}.")]
    return spec("resolved", ["scale_service:worker"], healthy=True, replicas=target), gold, {"n": target}


def _req_scale_unspecified(b: Builder, variant: str):
    n = len(_workers(b))
    gold = [finish("need_information", f"How many worker replicas do you want? There are {n} now (allowed 1-4).")]
    return spec("need_information"), gold, {}


def _req_if_hung(b: Builder, variant: str):
    target = b.rng.choice(_workers(b))
    if variant == "true":
        b.world.instances[target].hung = True
        gold = [("service_status", {}), ("restart_service", {"service": "worker", "instance": target}),
                finish("resolved", f"{target} was hung (no heartbeat for 900 s); restarted it.")]
        return spec("resolved", [f"restart_service:{target}"], healthy=True, restarted=[target]), gold, {"instance": target}
    gold = [("service_status", {}), finish("no_action_needed", f"{target} is not hung (heartbeat 2 s ago); left it alone.")]
    return spec("no_action_needed"), gold, {"instance": target}


def _req_if_spool_full(b: Builder, variant: str):
    s = b.world.spool
    if variant == "true":
        s["tmp_files"] = b.rng.randint(3, 40)
        s["tmp_mb"] = s["quota_mb"] - s["other_mb"] + b.rng.randint(0, 50)
        gold = [("get_metrics", {"component": "disk"}), ("cleanup_spool", {"pattern": "*.tmp"}),
                finish("resolved", f"Spool was over quota; deleted {s['tmp_files']} temp files.")]
        return spec("resolved", ["cleanup_spool"], healthy=True, spool_tmp_files=0), gold, {}
    gold = [("get_metrics", {"component": "disk"}),
            finish("no_action_needed", f"Spool is at {round(100 * (s['tmp_mb'] + s['other_mb']) / s['quota_mb'], 1)}% of "
                                       "quota, not over it; left it alone.")]
    return spec("no_action_needed"), gold, {}


def _req_cleanup(b: Builder, variant: str):
    s = b.world.spool
    s["tmp_files"] = max(s["tmp_files"], b.rng.randint(1, 12))
    s["tmp_mb"] = max(s["tmp_mb"], s["tmp_files"] * b.rng.randint(1, 5))
    s["other_mb"] = min(s["other_mb"], s["quota_mb"] - s["tmp_mb"] - 1)
    gold = [("cleanup_spool", {"pattern": "*.tmp"}), finish("resolved", f"Deleted {s['tmp_files']} temp files from the spool.")]
    return spec("resolved", ["cleanup_spool"], healthy=True, spool_tmp_files=0), gold, {}


def _req_rollback_release(b: Builder, variant: str):
    prev = b.world.releases[-2]["version"]
    gold = [("deploy_history", {}), ("rollback_release", {}), finish("resolved", f"Rolled the API back to {prev}.")]
    return spec("resolved", ["rollback_release"], healthy=True, release=prev), gold, {}


def _req_undo_config_by_author(b: Builder, variant: str):
    used = {v["author"] for v in b.world.config_versions}
    author = b.rng.choice([a for a in b.pools["authors"] if a not in used] or b.pools["authors"])
    note = b.pick("config_notes_benign")
    v = b.config_change(b._benign_change(), minutes_ago=b.rng.uniform(10, 90), author=author, note=note)
    if variant == "not_latest":
        later = b.config_change(b._benign_change(), minutes_ago=b.rng.uniform(1, 9),
                                author=b.rng.choice([a for a in b.pools["authors"] if a != author]))
        gold = [("config_history", {}),
                finish("escalated", f"{author}'s change is version {v['version']}, but version {later['version']} "
                                    f"by {later['author']} came after it. rollback_config only restores the previous "
                                    "version, which would undo the wrong change. Needs a manual config edit.")]
        return spec("escalated"), gold, {"author": author, "note": note}
    gold = [("config_history", {}), ("rollback_config", {}),
            finish("resolved", f"Reverted config version {v['version']} by {author} ('{note}').")]
    return (spec("resolved", ["rollback_config"], healthy=True, config_values_of=v["version"] - 1), gold,
            {"author": author, "note": note})


def _req_unavailable(b: Builder, variant: str):
    kind = b.pick("unavailable_kinds")
    ask = b.rng.choice(vocab.UNAVAILABLE_ASKS[kind])
    gold = [finish("escalated", f"I can't {ask}: none of my tools does that. Escalating to someone with access.")]
    return spec("escalated"), gold, {"ask": ask, "Ask": ask[0].upper() + ask[1:]}


def _incident(fn):
    def build(b, variant):
        sp, gold = fn(b, variant)
        return sp, gold, {}
    return build


KINDS = {k.name: k for k in [
    Kind("inc_worker_crashed", "incident", ("alert.jobs_lagging",), ("default",), _incident(_worker_crashed), _replicas_any),
    Kind("inc_worker_hung", "incident", ("alert.jobs_lagging",), ("default",), _incident(_worker_hung), _replicas_any),
    Kind("inc_bad_config", "incident", ("alert.api_errors",), ("latest", "buried"), _incident(_bad_config), _replicas_any),
    Kind("inc_exporter_leak", "incident", ("alert.api_errors",), ("default",), _incident(_exporter_leak), _replicas_any),
    Kind("inc_queue_backlog", "incident", ("alert.jobs_lagging",), ("default",), _incident(_queue_backlog),
         lambda v, r: r.randint(1, 3)),
    Kind("inc_disk_fill", "incident", ("alert.disk_space",), ("default",), _incident(_disk_fill), _replicas_any),
    Kind("inc_bad_deploy", "incident", ("alert.api_errors",), ("default",), _incident(_bad_deploy), _replicas_any),
    Kind("inc_slow_dependency", "incident", ("alert.payments_slow", "alert.jobs_lagging"), ("default",),
         _incident(_slow_dependency), _replicas_any),
    Kind("inc_expired_credential", "incident", ("alert.job_failures",), ("default",), _incident(_expired_credential),
         _replicas_any),
    Kind("inc_false_alarm", "incident", ("alert.api_4xx",), ("default",), _incident(_false_alarm), _replicas_any),
    Kind("req_restart_instance", "request", ("request.restart_instance",), ("exists", "missing"), _req_restart_instance,
         lambda v, r: r.randint(1, 4) if v == "exists" else r.randint(1, 3)),
    Kind("req_restart_service", "request", ("request.restart_service",), ("api", "report-exporter"), _req_restart_service,
         _replicas_any),
    Kind("req_restart_ambiguous", "request", ("request.restart_ambiguous",), ("single", "multi"), _req_restart_ambiguous,
         lambda v, r: 1 if v == "single" else r.randint(2, 4)),
    Kind("req_restart_failed", "request", ("request.restart_failed",), ("exited", "hung"), _req_restart_failed,
         lambda v, r: r.randint(2, 4)),
    Kind("req_scale", "request", ("request.scale",), ("change", "already"), _req_scale, _replicas_any),
    Kind("req_scale_unspecified", "request", ("request.scale_unspecified",), ("default",), _req_scale_unspecified,
         _replicas_any),
    Kind("req_if_hung", "request", ("request.if_hung",), ("true", "false"), _req_if_hung, _replicas_any),
    Kind("req_if_spool_full", "request", ("request.if_spool_full",), ("true", "false"), _req_if_spool_full, _replicas_any),
    Kind("req_cleanup", "request", ("request.cleanup",), ("default",), _req_cleanup, _replicas_any),
    Kind("req_rollback_release", "request", ("request.rollback_release",), ("default",), _req_rollback_release,
         _replicas_any),
    Kind("req_undo_config_by_author", "request", ("request.undo_config_by_author",), ("latest", "not_latest"),
         _req_undo_config_by_author, _replicas_any),
    Kind("req_unavailable", "request", ("request.unavailable",), ("default",), _req_unavailable, _replicas_any),
]}


def alert_slots(world: World, rng: random.Random) -> dict:
    api, q = world.get_metrics("api"), world.get_metrics("queue")
    reqs = api.get("requests_60s", {"2xx": 0, "4xx": 0, "5xx": 0})
    total = max(1, sum(reqs.values()))
    failed = sum(w.get("jobs_failed_60s", 0) for w in world.get_metrics("workers")["workers"].values())
    return {"oldest_s": q["oldest_pending_age_s"], "pending": q["pending"],
            "err_pct": round(100 * reqs["5xx"] / total), "err_ratio": round(reqs["5xx"] / total, 2),
            "failed": failed, "pay_ms": world.payments["p95_ms"],
            "disk_pct": world.get_metrics("disk")["used_pct"], "client_errors": world.client_errors_60s}


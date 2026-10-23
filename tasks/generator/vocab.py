"""Entity pools and phrasing templates. Every entity and template has a stable id, so splits (M2)
can assign them to train / val / held-out before any example is generated.

Templates are grouped by *family*: an alert family for incidents (the symptom a pager would show,
shared by every cause that produces it) or a request kind for operator requests. Slots are filled
from the task's world and entities.
"""

from __future__ import annotations

# --- entity pools ---------------------------------------------------------------------------------

AUTHORS = ["jdoe", "asmith", "mkhan", "lchen", "rpatel", "tnguyen", "okafor", "bwilliams", "sgarcia", "ykim",
           "dmuller", "fsilva", "hsato", "ivanova", "pkowalski", "nbanerjee", "eodonnell", "crossi",
           "deploy-bot", "config-sync", "release-ci", "platform-ops", "infra-bot", "jmartin"]

REQUESTERS = ["Priya", "Tom", "Aisha", "Marco", "Chen", "Lena", "Omar", "Sofia", "Kenji", "Grace",
              "Diego", "Hana", "Ravi", "Elif", "Noah", "Zara"]

CONFIG_NOTES_BAD = ["db failover prep", "sync settings from staging", "tune connection settings",
                    "point at new replica", "migrate to pooled endpoint", "consolidate db settings",
                    "prepare maintenance window", "copy prod-eu settings", "switch to read-write endpoint",
                    "update after infra change", "align with terraform", "temporary test change"]

CONFIG_NOTES_BENIGN = ["raise log verbosity for debugging", "lower log verbosity", "bigger worker batches",
                       "smaller worker batches", "shorter request timeout", "longer request timeout",
                       "export reports every 10 min", "export reports hourly", "faster queue polling",
                       "slower queue polling", "csv exports", "parquet exports"]

DEPLOY_NOTES = ["routine release", "customer field in job status", "faster job lookup", "dependency bumps",
                "hotfix for retries", "new status endpoint fields", "logging cleanup", "pagination for job list",
                "refactor serializers", "timezone fix", "rate-limit headers", "error message cleanup"]

# Wrong values a bad config change can introduce (the right ones are in BASE_CONFIG).
BAD_DB_VALUES = {
    "api.db_port": [5433, 6432, 15432, 5532, 7432, 25432],
    "api.db_host": ["db-replica-2.internal", "db-staging.internal", "db-old.internal", "pg-eu-1.internal",
                    "db-primary.staging", "localhost"],
    "api.db_name": ["jobs_staging", "jobs_test", "jobs_v2", "jobsdb", "jobs_eu", "reporting"],
}

BENIGN_CHANGES = {
    "api.log_level": ["debug", "info", "warning"],
    "api.request_timeout_ms": [2000, 3000, 5000, 8000],
    "worker.batch_size": [10, 25, 50, 100],
    "worker.poll_interval_ms": [100, 250, 500],
    "exporter.interval_s": [300, 600, 3600],
    "exporter.format": ["csv", "parquet", "jsonl"],
}

BASE_CONFIG = {"api.db_host": "db-primary.internal", "api.db_port": 5432, "api.db_name": "jobs",
               "api.log_level": "info", "api.request_timeout_ms": 5000, "worker.batch_size": 25,
               "worker.poll_interval_ms": 250, "exporter.interval_s": 600, "exporter.format": "csv"}

# Release series: the major.minor a task's releases are drawn from (entity for splits).
RELEASE_SERIES = ["1.4", "1.7", "2.0", "2.3", "2.6", "3.1", "3.4", "4.0", "4.2", "5.1", "5.5", "6.0"]

DEPLOY_ERRORS = [
    "TypeError: 'NoneType' object is not subscriptable in job_status()",
    "KeyError: 'customer' in serialize_job()",
    "AttributeError: 'dict' object has no attribute 'created_at'",
    "ValueError: invalid literal for int() with base 10: '' in parse_job_id()",
    "TypeError: Object of type datetime is not JSON serializable",
    "IndexError: list index out of range in render_status()",
    "KeyError: 'amount' in job_summary()",
    "AttributeError: 'NoneType' object has no attribute 'status'",
]

CRASH_ERRORS = ["MemoryError while decoding job payload", "Segmentation fault in payments client",
                "RuntimeError: event loop is closed", "psycopg.OperationalError: server closed the connection",
                "OSError: [Errno 24] Too many open files", "RecursionError: maximum recursion depth exceeded",
                "KeyboardInterrupt from supervisor", "SystemExit: 137"]

UNAVAILABLE_ASKS = {
    "rotate_credential": ["rotate the payments API credential", "issue a new payments token",
                          "refresh the payments provider credentials"],
    "db_limit": ["raise the database connection limit to 80", "increase max_connections on the database",
                 "bump the Postgres connection limit"],
    "purge_jobs": ["delete all failed jobs from the queue", "purge the failed jobs", "clear out the dead jobs"],
    "restart_payments": ["restart the payments provider", "bounce the payments service", "reboot the payments API"],
    "set_config": ["set api.db_port to 6432", "change the API's database host to db-replica-1.internal",
                   "set worker.batch_size to 200"],
    "restart_db": ["restart the database", "reboot Postgres", "bounce the database server"],
}

BAD_DB_CHANGES = [f"{k}={v}" for k, vs in BAD_DB_VALUES.items() for v in vs]   # "key=value", parsed by scenarios

POOLS = {"authors": AUTHORS, "bad_db_changes": BAD_DB_CHANGES, "requesters": REQUESTERS, "config_notes_bad": CONFIG_NOTES_BAD,
         "config_notes_benign": CONFIG_NOTES_BENIGN, "deploy_notes": DEPLOY_NOTES, "release_series": RELEASE_SERIES,
         "deploy_errors": DEPLOY_ERRORS, "crash_errors": CRASH_ERRORS,
         "unavailable_kinds": list(UNAVAILABLE_ASKS)}

# --- templates ------------------------------------------------------------------------------------
# Alert families: slots come from the world's metrics at task creation.

ALERT_TEMPLATES = {
    "jobs_lagging": [
        "ALERT JobsLagging firing: oldest pending job is {oldest_s} s old ({pending} pending). Investigate and fix if you can.",
        "[pager] jobs service: queue latency high, oldest job {oldest_s}s, pending={pending}.",
        "Heads up from support: customers say invoices are taking ages to process. Queue shows {pending} pending jobs.",
        "Ticket OPS-{ticket}: job processing delayed. Oldest pending job age {oldest_s} s. Please take a look.",
        "Alertmanager: JobsLagging (severity=warning) pending_jobs={pending} oldest_age_s={oldest_s}",
        "{requester}: jobs seem stuck or slow, {pending} in the queue. Can you check what's going on and sort it out?",
        "Monitoring: job throughput below target; {pending} jobs waiting, oldest {oldest_s} s.",
        "On-call: the jobs queue is backing up ({pending} pending). Find the cause and fix it if it's safe.",
    ],
    "api_errors": [
        "ALERT Api5xxHigh firing: {err_pct}% of API requests failed with 5xx in the last 5 min.",
        "[pager] jobs-api error rate {err_pct}% (5xx). Investigate and remediate.",
        "Ticket OPS-{ticket}: users report errors from the jobs API (about {err_pct}% of requests).",
        "Alertmanager: Api5xxHigh (severity=critical) error_ratio={err_ratio}",
        "{requester}: the API is throwing 500s, roughly {err_pct}% of calls. Can you fix it?",
        "Monitoring: API availability SLO burning fast; 5xx ratio {err_ratio} over 5 min.",
        "Status page draft: elevated API errors ({err_pct}%). On-call, please find and fix the cause.",
        "On-call: synthetic checks against the jobs API are failing ({err_pct}% 5xx).",
    ],
    "job_failures": [
        "ALERT JobFailuresHigh firing: {failed} jobs failed in the last minute.",
        "[pager] jobs service: job failure rate high ({failed}/min).",
        "Ticket OPS-{ticket}: payments for invoices are failing. {failed} failed jobs in the last minute.",
        "Alertmanager: JobFailuresHigh (severity=critical) failed_per_min={failed}",
        "{requester}: lots of failed jobs right now ({failed} a minute). What's happening?",
        "Monitoring: job success rate dropped; {failed} failures per minute.",
        "On-call: invoice jobs are failing at {failed}/min. Investigate and fix if possible.",
        "Finance reports that payments aren't going through; worker failures at {failed}/min.",
    ],
    "payments_slow": [
        "ALERT PaymentsLatencyHigh firing: payments call p95 {pay_ms} ms.",
        "[pager] workers: payments provider latency p95={pay_ms}ms.",
        "Ticket OPS-{ticket}: invoice processing slow; payments calls p95 {pay_ms} ms.",
        "Alertmanager: PaymentsLatencyHigh (severity=warning) p95_ms={pay_ms}",
        "{requester}: payment calls are really slow (p95 around {pay_ms} ms). Can you fix it?",
        "Monitoring: dependency latency high (payments, p95 {pay_ms} ms).",
        "On-call: workers report slow payments calls ({pay_ms} ms p95). Investigate.",
        "Support: customers see slow invoice confirmations; payments p95 {pay_ms} ms.",
    ],
    "disk_space": [
        "ALERT SpoolDiskFull firing: spool at {disk_pct}% of quota.",
        "[pager] report-exporter: spool usage {disk_pct}%.",
        "Ticket OPS-{ticket}: report exports failing; spool volume {disk_pct}% full.",
        "Alertmanager: SpoolDiskFull (severity=critical) used_pct={disk_pct}",
        "{requester}: reports aren't being exported and the spool looks full ({disk_pct}%). Help?",
        "Monitoring: spool directory over quota ({disk_pct}%).",
        "On-call: disk alert on the export spool ({disk_pct}% used). Fix if safe.",
        "Data team: yesterday's exports never arrived; spool reports {disk_pct}% usage.",
    ],
    "api_4xx": [
        "ALERT Api4xxHigh firing: {client_errors} client errors in the last minute.",
        "[pager] jobs-api 4xx spike: {client_errors}/min.",
        "Ticket OPS-{ticket}: API returning lots of 400s ({client_errors} in the last minute).",
        "Alertmanager: Api4xxHigh (severity=warning) count_per_min={client_errors}",
        "{requester}: the API is rejecting a lot of requests ({client_errors} per minute). Is something broken?",
        "Monitoring: unusual rate of 4xx responses ({client_errors}/min).",
        "On-call: error budget dashboard shows a 4xx spike ({client_errors}/min). Check it.",
        "Partner integration team says their calls are being refused ({client_errors}/min on our side).",
    ],
}

REQUEST_TEMPLATES = {
    "restart_instance": [
        "Please restart {instance}.",
        "{requester} here: can you restart {instance}?",
        "Restart {instance}, it's been acting up since the morning.",
        "Ticket OPS-{ticket}: restart {instance} as part of routine maintenance.",
        "Could you bounce {instance} for me?",
        "Need a restart of {instance}, thanks.",
        "Go ahead and restart {instance}.",
        "{requester}: please give {instance} a restart.",
    ],
    "restart_service": [
        "Please restart the {service_name}.",
        "{requester} here: can you restart the {service_name}?",
        "Restart the {service_name}, please.",
        "Ticket OPS-{ticket}: restart the {service_name} after the library update.",
        "Could you bounce the {service_name}?",
        "Need the {service_name} restarted, thanks.",
        "Go ahead and restart the {service_name}.",
        "{requester}: please give the {service_name} a restart.",
    ],
    "restart_ambiguous": [
        "Please restart the worker.",
        "{requester} here: can you restart the worker?",
        "Restart a worker for me.",
        "Ticket OPS-{ticket}: worker restart requested.",
        "Could you bounce the worker?",
        "Need a worker restarted, thanks.",
        "Go ahead and restart the worker please.",
        "{requester}: the worker needs a restart.",
    ],
    "restart_failed": [
        "One of the workers seems to have stopped. Please restart it.",
        "{requester} here: a worker looks dead. Can you restart whichever one it is?",
        "Restart the worker that has stopped processing.",
        "Ticket OPS-{ticket}: a worker stopped picking up jobs; restart that worker.",
        "Could you find the stuck or dead worker and restart just that one?",
        "I think a worker died. Restart it please, but leave the healthy ones alone.",
        "Go ahead and restart the broken worker.",
        "{requester}: one worker isn't working. Restart it.",
    ],
    "scale": [
        "Please scale the workers to {n} replicas.",
        "{requester} here: can you set the worker count to {n}?",
        "Scale workers to {n}.",
        "Ticket OPS-{ticket}: run {n} worker replicas from now on.",
        "Could you change the number of workers to {n}?",
        "We need {n} workers running, please adjust.",
        "Go ahead and set worker replicas to {n}.",
        "{requester}: make it {n} workers.",
    ],
    "scale_unspecified": [
        "Please scale the workers.",
        "{requester} here: can you change the worker count?",
        "Adjust the number of worker replicas.",
        "Ticket OPS-{ticket}: worker replica count needs changing.",
        "Could you rescale the workers?",
        "We need a different number of workers.",
        "Go ahead and resize the worker pool.",
        "{requester}: scale the worker deployment please.",
    ],
    "if_hung": [
        "If {instance} is hung, restart it.",
        "{requester} here: check whether {instance} is stuck, and restart it only if it is.",
        "Restart {instance} if it has stopped making progress; otherwise leave it.",
        "Ticket OPS-{ticket}: {instance} may be hung. Restart it if so.",
        "Could you see if {instance} is frozen and bounce it if needed?",
        "Only if {instance} is stuck: restart it.",
        "Check {instance}'s heartbeat; if it's hung, restart it.",
        "{requester}: is {instance} hung? If yes, restart it.",
    ],
    "if_spool_full": [
        "If the spool is over quota, clean up its temp files.",
        "{requester} here: check the spool and clean temp files only if it's over quota.",
        "Clean the spool's temp files if the spool is full; otherwise leave it.",
        "Ticket OPS-{ticket}: spool may be full. Clean temp files if so.",
        "Could you check whether the spool is over its quota and clean it up if needed?",
        "Only if the spool is over quota: delete the temp files.",
        "Check disk usage on the spool; clean temp files if it's over quota.",
        "{requester}: is the spool over quota? If yes, clean it.",
    ],
    "cleanup": [
        "Please delete the temp files in the spool.",
        "{requester} here: can you clean the spool's temp files?",
        "Clean up the spool temp files.",
        "Ticket OPS-{ticket}: housekeeping, remove temp files from the spool.",
        "Could you clear the *.tmp files out of the spool?",
        "Need the spool's temporary files removed, thanks.",
        "Go ahead and clean the spool.",
        "{requester}: please purge the spool temp files.",
    ],
    "rollback_release": [
        "Please roll the API back to the previous release.",
        "{requester} here: can you redeploy the previous API release?",
        "Roll back the API release.",
        "Ticket OPS-{ticket}: revert the API to the release before the current one.",
        "Could you undo the latest API deploy?",
        "We need the previous API version back, please.",
        "Go ahead and roll back the latest API release.",
        "{requester}: revert the last API deploy.",
    ],
    "undo_config_by_author": [
        "Please undo the config change {author} made.",
        "{requester} here: can you revert {author}'s config change?",
        "Roll back the config change from {author}.",
        "Ticket OPS-{ticket}: revert config change by {author} ('{note}').",
        "Could you undo {author}'s change to the config?",
        "We need {author}'s config change reverted.",
        "Go ahead and revert the config change '{note}' by {author}.",
        "{requester}: undo the config update {author} pushed.",
    ],
    "unavailable": [
        "Please {ask}.",
        "{requester} here: can you {ask}?",
        "{Ask}, please.",
        "Ticket OPS-{ticket}: {ask}.",
        "Could you {ask}?",
        "We need someone to {ask}.",
        "Go ahead and {ask}.",
        "{requester}: {ask} please.",
    ],
}

TEMPLATES = {**{f"alert.{k}": v for k, v in ALERT_TEMPLATES.items()},
             **{f"request.{k}": v for k, v in REQUEST_TEMPLATES.items()}}


def template_ids(family: str) -> list[str]:
    return [f"{family}.{i}" for i in range(len(TEMPLATES[family]))]


def template_text(template_id: str) -> str:
    family, i = template_id.rsplit(".", 1)
    return TEMPLATES[family][int(i)]

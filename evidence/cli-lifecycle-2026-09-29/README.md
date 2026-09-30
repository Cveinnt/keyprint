# CLI backend cleanup: implemented, execution checks pending

`keyprint generate` previously constructed a backend and returned without calling
its public `close()` method. The CLI now closes it exactly once after generation
success, generation failure or interruption. If cleanup also fails during a
generation failure, the original exception and its private artifact path remain
the primary result; the cleanup failure is reported separately. No failed output
is retried or rewritten.

Eight new cases cover success, ordinary runtime failure, retained-report failure
and KeyboardInterrupt, each with successful or failing cleanup. Existing routing
fixtures now implement the backend's close contract. Syntax validation passes.

**These tests have not executed yet.** The 512 MiB test supervisor refused startup
on elevated system memory pressure, before starting a worker. Four subsequent
readings also reported warning pressure. The refusal receipt is retained. No
unguarded substitute, model load or unrelated-process shutdown followed.

This change calls the existing backend cleanup contract. It does not establish
that every backend immediately returns all allocations to the OS, nor does it
diagnose the reported ChatGPT memory growth. The modified CLI needs its focused
tests and an updated wheel check before release-readiness credit.

Before editing Python source, the SDK from the previously verified wheel was
preserved separately for the frozen balanced research study. All 78 Python-file
hashes match that study's original SDK manifest exactly. Original helpers, study
data, keys, settings and ratings remain unchanged. Future research must use that
exact SDK snapshot if it resumes; the local-model hold still applies.

The earlier core-install receipt applies to commit `cefce38`, before this change.
No new wheel, public release, registry publication, deployment or CI run is
claimed. Native compatibility, semantic preservation and detection remain open.

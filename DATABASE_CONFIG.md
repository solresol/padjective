# Production database configuration

This revision's production profile uses PostgreSQL 18. Activate it only as part
of the verified Shopify / Padjective / CantBuyMeLove group cutover.

| Setting | Production location / value |
| --- | --- |
| Host and checkout | Raksasa, `/home/padjective/padjective` |
| Database and service role | `shopifystores`, `padjective` |
| Production endpoint | PostgreSQL 18, Unix socket `/var/run/postgresql`, port 5433 |
| Retained recovery endpoint | PG16/5432; read-only after coordinated activation |
| Daily writer | Installed `padjective` crontab: `30 5 * * *`, Sydney time; `cronscript.sh` |
| Monthly publisher | `/home/padjective/bin/publish_product_taxonomy_bench.sh`; day 1 at 03:00 Sydney |
| Shared Python connection | `padjective/db.py:get_connection()` |

The daily wrapper pins PGHOST, PGPORT, PGDATABASE, PGUSER and the four
PADJECTIVE_EXPECTED_* identity fields. The Python helper checks the connected
database, port, server major and, when configured, role before application SQL.
A stale explicit DSN therefore fails rather than writing to the old cluster.

Explicit `--dsn` takes precedence over SHOPIFY_DB_DSN, then DATABASE_URL.
Otherwise PGDATABASE/PGHOST/PGPORT are used with the production defaults above.
The monthly wrapper optionally reads `$HOME/.shopify-db-dsn`, overridable through
PADJECTIVE_SHOPIFY_DSN_FILE; that file was absent during the readiness check.
Recheck it at cutover. No `.env` or `.pgpass` was present for this account at the
5 November check. The verified same-host service route uses peer authentication.
Do not put passwords or DSN secrets into Git.

GitHub Actions uses a separate disposable `postgres:18` service on its hosted
runner, exposed at localhost:5433. Its fixture database/role are `postgres`,
with explicit CI identity expectations. It does not connect to Raksasa. Fixture
seeding verifies the connected identity before any destructive setup commands.

Padjective reads Shopify public data and CantBuyMeLove products and writes the
`padjective` schema. Migrate this dependency group together. Preserve generated
product IDs, sequence values, real product JSON and research history. Keep the
old database and focused recovery archives; after new PG18 writes, rollback
requires reconciliation before permitting PG16 writes.

Migration evidence and preserved source/candidate recovery are on Raksasa under
`/mnt/crossref-usb-scratch/pg18-focused-recovery-20261105T120000Z` and the adjacent
candidate, delta and cutover directories. The migration preserves generated IDs,
all sequence states and exact product JSON; it does not reload the public corpus
or repeat completed index repairs. Observe the ordinary daily job after routing;
do not start an extra mathematical experiment for migration acceptance.

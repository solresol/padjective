# Production database configuration

Prepared migration branch, 5 November 2026: production still uses PostgreSQL 16
until the verified Shopify group cutover. Do not deploy this branch independently
of that cutover.

| Setting | Production location / value |
| --- | --- |
| Host and checkout | Raksasa, `/home/padjective/padjective` |
| Database and service role | `shopifystores`, `padjective` |
| Current endpoint | PostgreSQL 16, port 5432 |
| Prepared endpoint | PostgreSQL 18, Unix socket `/var/run/postgresql`, port 5433 |
| Daily writer | Installed `padjective` crontab: `30 5 * * *`, Sydney time; `cronscript.sh` |
| Monthly publisher | `/home/padjective/bin/publish_product_taxonomy_bench.sh`; day 1 at 03:00 Sydney |
| Shared Python connection | `padjective/db.py:get_connection()` |

The daily wrapper pins PGHOST, PGPORT, PGDATABASE, PGUSER and the four
PADJECTIVE_EXPECTED_* identity fields. The Python helper checks the connected
database, port, server major and, when configured, role before application SQL.
A stale explicit DSN therefore fails rather than writing to the old cluster.

Explicit `--dsn` takes precedence over SHOPIFY_DB_DSN, then DATABASE_URL.
Otherwise PGDATABASE/PGHOST/PGPORT are used with the prepared defaults above.
The monthly wrapper optionally reads `$HOME/.shopify-db-dsn`, overridable through
PADJECTIVE_SHOPIFY_DSN_FILE; that file was absent during the readiness check.
Recheck it at cutover. No `.env` or `.pgpass` was present for this account at the
5 November check. The verified same-host service route uses peer authentication.
Do not put passwords or DSN secrets into Git.

Padjective reads Shopify public data and CantBuyMeLove products and writes the
`padjective` schema. Migrate this dependency group together. Preserve generated
product IDs, sequence values, real product JSON and research history. Keep the
old database and focused recovery archives; after new PG18 writes, rollback
requires reconciliation before permitting PG16 writes.

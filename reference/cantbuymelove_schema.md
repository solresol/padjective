# cantbuymelove schema snapshot

The following relations are currently available in the `cantbuymelove` schema of
the Shopify Stores Postgres database. This snapshot reflects the schema
observed while debugging the `tagbattle` pipeline.

## `cantbuymelove.taxonomy`

| Column        | Type | Notes |
| ------------- | ---- | ----- |
| `taxonomy_id` | text | Primary key. |
| `taxonomy_name` | text | Unique constraint. |
| `taxonomy_path` | text |  |

## `cantbuymelove.product_taxonomy`

| Column | Type | Notes |
| ------ | ---- | ----- |
| `product_key` | text | Primary key. References `cantbuymelove.taxonomy(taxonomy_id)` via the `taxonomy_id` column. |
| `store_domain` | text | |
| `product_title` | text | |
| `product_url` | text | |
| `taxonomy_id` | text | Nullable FK to `cantbuymelove.taxonomy`. |
| `raw_output` | jsonb | |
| `prompt_tokens` | integer | |
| `completion_tokens` | integer | |
| `total_tokens` | integer | |
| `classified_at` | timestamptz | Defaults to `now()`. |

## `cantbuymelove.products_for_classification`

A view exposing products that have not yet been classified. Columns:

| Column | Type |
| ------ | ---- |
| `product_key` | text |
| `store_domain` | varchar |
| `title` | text |
| `description` | text |
| `tags` | text |
| `product_url` | text |
| `raw_payload` | jsonb |

These notes capture the information originally gathered via `\d` in `psql`.

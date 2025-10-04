                                               Table "cantbuymelove.product_taxonomy"
      Column       |           Type           | Collation | Nullable | Default | Storage  | Compression | Stats target | Description 
-------------------+--------------------------+-----------+----------+---------+----------+-------------+--------------+-------------
 product_key       | text                     |           | not null |         | extended |             |              | 
 store_domain      | text                     |           |          |         | extended |             |              | 
 product_title     | text                     |           | not null |         | extended |             |              | 
 product_url       | text                     |           |          |         | extended |             |              | 
 taxonomy_id       | text                     |           |          |         | extended |             |              | 
 taxonomy_path     | text                     |           |          |         | extended |             |              | 
 raw_output        | jsonb                    |           |          |         | extended |             |              | 
 prompt_tokens     | integer                  |           |          |         | plain    |             |              | 
 completion_tokens | integer                  |           |          |         | plain    |             |              | 
 total_tokens      | integer                  |           |          |         | plain    |             |              | 
 classified_at     | timestamp with time zone |           |          | now()   | plain    |             |              | 
Indexes:
    "product_taxonomy_pkey" PRIMARY KEY, btree (product_key), tablespace "shopifystores_space"
Access method: heap


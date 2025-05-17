\copy (select product_detail->'product'->>'title' as title, product_detail->'product'->>'tags' as tags from product_details) to 'products.csv' header csv

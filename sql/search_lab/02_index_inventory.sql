SELECT
    table,
    name AS index_name,
    type_full,
    granularity
FROM system.data_skipping_indices
WHERE database = 'default'
  AND table LIKE 'clips_%'
ORDER BY table, index_name

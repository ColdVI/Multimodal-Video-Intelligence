SELECT
    name AS table_name,
    total_rows,
    total_bytes
FROM system.tables
WHERE database = 'default'
  AND name LIKE 'clips_%'
ORDER BY name

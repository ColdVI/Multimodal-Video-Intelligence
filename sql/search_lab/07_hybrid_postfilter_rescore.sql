WITH (
    SELECT embedding
    FROM clips_xclip_hf_zeroshot
    ORDER BY video_id, t_start
    LIMIT 1
) AS query_vector
SELECT
    video_id,
    t_start,
    t_end,
    bus_count,
    person_count,
    cosineDistance(embedding, query_vector) AS distance,
    1 - distance AS score
FROM clips_xclip_hf_zeroshot
WHERE bus_count >= 1
  AND person_count >= 1
ORDER BY distance ASC
LIMIT 10
SETTINGS
    vector_search_filter_strategy = 'postfilter',
    vector_search_with_rescoring = 1,
    vector_search_index_fetch_multiplier = 5

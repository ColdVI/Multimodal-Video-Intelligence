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
    cosineDistance(embedding, query_vector) AS distance,
    1 - distance AS score
FROM clips_xclip_hf_zeroshot
WHERE person_count >= 1
ORDER BY distance ASC
LIMIT 10
SETTINGS query_plan_try_use_vector_search = 1

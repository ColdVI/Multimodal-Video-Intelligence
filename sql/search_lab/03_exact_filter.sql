SELECT
    video_id,
    t_start,
    t_end,
    bus_count,
    person_count,
    length(embedding) AS embedding_dim
FROM clips_xclip_hf_zeroshot
WHERE bus_count >= 1
  AND person_count >= 1
ORDER BY video_id, t_start
LIMIT 10

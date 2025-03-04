-- lists all bans with Glam rock listed as their mainstyle ranked by their lifespan
SELECT band_name, split - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%';
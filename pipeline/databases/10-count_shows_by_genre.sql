-- lists all generes and displays the dunmber of shows linked to each

SELECT tv_genres.name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows
FROM tv_genres
LEFT JOIN tv_show_genres
ON tv_genres.id = tv_show_genres.genre_id
WHERE number_of_shows > 0
GROUP BY genre
ORDER BY number_of_shows DESC;
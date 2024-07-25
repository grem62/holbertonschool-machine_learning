-- Find the TV shows without a genre
SELECT tv_shows.title, tv_show_genres.genre_id
FROM hbtn_0d_tvshows AS tv_shows
LEFT JOIN hbtn_0d_tv_show_genres AS tv_show_genres
ON tv_shows.id = tv_show_genres.tv_show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
-- stored procedure that adds new corrections for students
DELIMITER $$ 
CREATE PROCEDURE AddBonus (
    IN user_id INT,
    IN project_name VARCHAR(256),
    IN score INT)
BEGIN
    -- make the name column a key so no duplicates can be made
    ALTER TABLE projects
    ADD UNIQUE (name);

    INSERT INTO projects (name)
    VALUES (project_name)
    ON DUPLICATE KEY UPDATE name = name;

    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, (SELECT id FROM projects WHERE name = project_name), score);
END $$

DELIMITER ;

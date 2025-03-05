-- creates a trigger that resets the attribute valid_email only when the email has been changed
DELIMITER $$

CREATE TRIGGER reset_attribute
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF OLD.email != NEW.email THEN
        SET NEW.valid_email = 1 - NEW.valid_email;
    END IF;
END $$

DELIMITER ;

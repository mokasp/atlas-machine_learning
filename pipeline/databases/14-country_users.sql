-- creates the table users if it does not exists
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL UNIQUE,
    name VARCHAR(256),
    country ENUM("US", "CO", "TN") DEFAULT "US"
)
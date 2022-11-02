--- marianne/schema.sql ---

DROP TABLE IF EXISTS METADATA;

CREATE TABLE METADATA (
    title TEXT,
    url TEXT UNIQUE,
    description TEXT,
    label TEXT
);
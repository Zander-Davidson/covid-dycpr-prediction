USE CovidEconomy;
GO

DROP TABLE CovidEconomy.dbo.StatePopulationJul19;
GO

CREATE TABLE CovidEconomy.dbo.StatePopulationJul19
(
    state_id CHAR(2) NOT NULL,
    state_name VARCHAR(MAX) NOT NULL,
    population_total INTEGER NOT NULL,
    population_18_plus INTEGER NOT NULL,
    CONSTRAINT PK_StatePopulationJul19 PRIMARY KEY (state_id)
);
GO

BULK INSERT StatePopulationJul19
FROM '/usr/data/csv/state-population-jul-2019.csv'
WITH
(
    FIRSTROW = 2,
    FORMAT = 'CSV'
);
GO
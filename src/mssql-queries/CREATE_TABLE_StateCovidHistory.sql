USE CovidEconomy;
GO

DROP TABLE CovidEconomy.dbo.StateCovidHistory;
GO

CREATE TABLE CovidEconomy.dbo.StateCovidHistory
(
    date DATE NOT NULL,
    state_id CHAR(2) NOT NULL,
    data_quality_grade CHAR(10) DEFAULT NULL,
    death INTEGER DEFAULT NULL, -- death_confirmed + death_probable
    death_confirmed INTEGER DEFAULT NULL,
    death_increase INTEGER DEFAULT NULL,
    death_probable INTEGER DEFAULT NULL,
    hospitalized INTEGER DEFAULT NULL,
    hospitalized_cumulative INTEGER DEFAULT NULL,
    hospitalized_currently INTEGER DEFAULT NULL,
    hospitalized_increase INTEGER DEFAULT NULL,
    in_icu_cumulative INTEGER DEFAULT NULL,
    in_icu_currently INTEGER DEFAULT NULL,
    negative INTEGER DEFAULT NULL,
    negative_increase INTEGER DEFAULT NULL,
    negative_tests_antibody INTEGER DEFAULT NULL,
    negative_tests_people_antibody INTEGER DEFAULT NULL,
    negative_tests_viral INTEGER DEFAULT NULL,
    on_ventilator_cumulative INTEGER DEFAULT NULL,
    on_ventilator_currently INTEGER DEFAULT NULL,
    positive INTEGER DEFAULT NULL,
    positive_cases_viral INTEGER DEFAULT NULL,
    positive_increase INTEGER DEFAULT NULL,
    positive_score INTEGER DEFAULT NULL,
    positive_tests_antibody INTEGER DEFAULT NULL,
    positive_tests_antigen INTEGER DEFAULT NULL,
    positive_tests_people_antibody INTEGER DEFAULT NULL,
    positive_tests_people_antigen INTEGER DEFAULT NULL,
    positive_tests_viral INTEGER DEFAULT NULL,
    recovered INTEGER DEFAULT NULL,
    total_test_encounters_viral INTEGER DEFAULT NULL,
    total_test_encounters_viral_increase INTEGER DEFAULT NULL,
    total_test_results INTEGER DEFAULT NULL,
    total_test_results_increase INTEGER DEFAULT NULL,
    total_tests_antibody INTEGER DEFAULT NULL,
    total_tests_antigen INTEGER DEFAULT NULL,
    total_tests_people_antibody INTEGER DEFAULT NULL,
    total_tests_people_antigen INTEGER DEFAULT NULL,
    total_tests_people_viral INTEGER DEFAULT NULL,
    total_tests_people_viral_increase INTEGER DEFAULT NULL,
    total_tests_viral INTEGER DEFAULT NULL,
    total_tests_viral_increase INTEGER DEFAULT NULL,
    CONSTRAINT PK_StateCovidHistory PRIMARY KEY (state_id, date)
);
GO

BULK INSERT StateCovidHistory
FROM '/usr/data/csv/all-states-history.csv'
WITH
(
    FIRSTROW = 2,
    FORMAT = 'CSV'
);
GO
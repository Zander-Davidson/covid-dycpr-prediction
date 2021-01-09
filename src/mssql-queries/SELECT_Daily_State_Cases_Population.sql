DECLARE @MILLIS_PER_WEEK AS DECIMAL = 604800000.0;
DECLARE @MILLIS_PER_DAY AS DECIMAL = 86400000.0;
DECLARE @FIRST_DATE AS DATE = '1970-01-01';
DECLARE @MIN_DATE AS DATE = '2020-07-01';
DECLARE @MAX_DATE AS DATE = '2020-12-11';

DECLARE @NUM_BINS INT = 5;
-- determined by running this query with extra block for min/max as below
DECLARE @MIN_BIN_VAL DECIMAL = 0.0;
DECLARE @MAX_BIN_VAL DECIMAL = 5.0;
DECLARE @BIN_SIZE DECIMAL = (@MAX_BIN_VAL - @MIN_BIN_VAL) / @NUM_BINS;

WITH A (
    state_id,
    state_name,
    date,
    day,
    total_positive,
    positive_per_100k,
    positive_increase,
    positive_increase_per_100k,
    population_18_plus
) AS (
    SELECT 
        Pop.state_id,
        Pop.state_name,
        date,
        CAST(((CAST(DATEDIFF(s, @MIN_DATE, (SELECT date)) AS BIGINT)*1000) / @MILLIS_PER_DAY) AS INT) + 1 AS day,
        positive AS total_positive,
        NULLIF((CAST(COALESCE(positive, 0) AS DECIMAL) / CAST(population_18_plus AS DECIMAL)) * 100000, 0)  AS positive_per_100k,
        -- some positive_increase values were negative in the original dataset --> change these to 0
        NULLIF(
            (CASE 
                WHEN COALESCE(positive_increase, 0) <= 0 THEN 0
                ELSE positive_increase
            END)
        , 0) AS positive_increase,
        NULLIF(
            (CASE 
                WHEN (COALESCE(positive_increase, 0) <= 0 OR COALESCE(total_test_results, 0) <= 0) THEN 0
                ELSE (CAST(positive_increase AS DECIMAL) / CAST(population_18_plus AS DECIMAL)) * 100000 END)
        , 0) AS positive_increase_per_100k,
        population_18_plus
    FROM StatePopulationJul19 AS Pop
    JOIN StateCovidHistory AS Covid ON Covid.state_id = Pop.state_id
),
-- determine summary stats by date 
B (
    state_id,
    state_name,
    date,
    day,
    total_positive,
    positive_per_100k,
    positive_increase,
    positive_increase_per_100k,
    population_18_plus
) AS (
    SELECT
        state_id,
        state_name,
        date,
        day,
        MAX(total_positive) OVER (PARTITION BY tp_cnt) AS total_positive,
        MAX(positive_per_100k) OVER (PARTITION BY tp100_cnt) AS positive_per_100k,
        MAX(positive_increase) OVER (PARTITION BY pi_cnt) AS positive_increase,
        MAX(positive_increase_per_100k) OVER (PARTITION BY pi100_cnt) AS positive_increase_per_100k,
        population_18_plus
    FROM (
        SELECT
            state_id,
            state_name,
            date,
            day,
            total_positive,
            positive_per_100k,
            positive_increase,
            positive_increase_per_100k,
            COUNT(total_positive) OVER (ORDER BY state_id, date) AS tp_cnt,
            COUNT(positive_per_100k) OVER (ORDER BY state_id, date) AS tp100_cnt,
            COUNT(positive_increase) OVER (ORDER BY state_id, date) AS pi_cnt,
            COUNT(positive_increase_per_100k) OVER (ORDER BY state_id, date) AS pi100_cnt,
            population_18_plus
        FROM A
    ) interpolated
    -- comment out for total summary stats
    -- GROUP BY date
)
-- uncomment for summary stats
-- SELECT * FROM B 

SELECT * FROM B
WHERE date >= @MIN_DATE
ORDER BY date, state_id
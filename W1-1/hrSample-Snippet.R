
library(tidyverse)
glimpse(salaryhistory_tbl)

# We have use the pkg::fct notation to call the function fct from the package pkg without making all the functions of pkg available. - select a few column
filter(salaryhistory_tbl, 
       salary_effective_date >= lubridate::dmy("01-01-2000"),
       salary_effective_date <= lubridate::dmy("31-12-2000"))

select(salaryhistory_tbl, employee_num, salary)

# mutate => create a new column. Note that the table salaryhistory_tbl is not modified, a new table has been created which should be explicitly saved
mutate(salaryhistory_tbl,
       salary_effective_year = lubridate::year(salary_effective_date))
salaryhistorymod_tbl <- mutate(salaryhistory_tbl,
                               salary_effective_year = lubridate::year(salary_effective_date))

# where we have used the R pipe %>% which allows to chain instructions. The previous sequence is equivalent to
salaryhistorymod_tbl %>% 
  filter(salary_effective_year == 2013) %>% 
  summarize(salary = mean(salary))
summarize(filter(salaryhistorymod_tbl, salary_effective_year == 2013), salary = mean(salary))

# summarize
salaryhistorymod_tbl %>% 
  filter(salary_effective_year == 2013) %>% 
  summarize(salary = mean(salary))
summarize(filter(salaryhistorymod_tbl, salary_effective_year == 2013), salary = mean(salary))

# join two tables
deskjob_tbl %>% left_join(hierarchy_tbl) ## Joining, by = "desk_id"

# Note that the key used to join the table has been chosen automatically as the common column. It can be specified explicitly if required:
deskjob_tbl %>% left_join(hierarchy_tbl, by = c("desk_id" = "desk_id"))

glimpse(salaryhistorymod_tbl)

summarise(salaryhistory_tbl, max(salary))
filter(salaryhistory_tbl, salary == max(salary))
summarize(salaryhistory_tbl, max(salary))
filter(salaryhistory_tbl, salary >= 0.8 * max(salary))

salaryhistory_tbl %>%
  filter(salary >= 0.8 * max(salary))


salaryhistory_tbl %>%
  filter(salary >= 0.8 * max(salary))


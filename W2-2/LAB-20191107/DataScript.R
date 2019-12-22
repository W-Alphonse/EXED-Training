source("Accidents_ETL.R")

usagers_months <- usagers %>% left_join(caracteristiques) %>%
  group_by(month = lubridate::floor_date(date, "month"), grav) %>%
  count() %>% 
  ungroup()

saveRDS(usagers_months, "usagers_months.RDS")

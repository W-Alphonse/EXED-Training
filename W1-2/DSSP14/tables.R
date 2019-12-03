# Ctrl+l: clear / Ctrl-Shift-M: %>%  / Ctrl-Shift-C: Comment

# https://cran.r-project.org/web/packages/tidyverse/index.html
# https://tidyverse.tidyverse.org/
# https://github.com/rstudio/cheatsheets/blob/master/data-transformation.pdf

# Package 'tidyverse' :
# ---------------------
# ggplot2, for data visualisation.
# dplyr, for data manipulation.
# tidyr, for data tidying.
# readr, for data import.
# purrr, for functional programming.
# tibble, for tibbles, a modern re-imagining of data frames.
# stringr, for strings.
# forcats, for factors.

# Install from CRAN
install.packages(tidyverse)

# load the whole "tidyverse" package
library(tidyverse)

# View the entire dataset 'iris'
iris
# view as subset of the dataset 'iris'
as_tibble(iris)

# ?filter
# help(select)
# https://dplyr.tidyverse.org/reference/filter.html

tbl_iris <- as_tibble(iris)
tbl_iris

# Manipulate data from dataset : filter / select / arrange / group_by / summarize
filter(tbl_iris, Petal.Length > 3)
select(tbl_iris, Sepal.Length, Species)
select(tbl_iris, Sepal.Length, Species, everything())
select_if(tbl_iris, is.numeric)  # permet de seelctionner quel colonne nous interesse d'une manière conditionnelle
select(tbl_iris, -Species)

tbl_iris[1:10, c(2,5)] # to avoid. select column_2_and_5 from rows.1_to_10
select(tbl_iris, 3:5)  # to avoid. select column_3_to_5  in all dataset

mutate(tbl_iris, Petal.Size = Petal.Length * Petal.Width)  # Rajoute une colonne
mutate(tbl_iris, Species = str_to_title(Species))  # Modifie une colonne ds le dataset récupérer; Le ds initial reste intacte
mutate(tbl_iris, Species = NULL)                   # Supprimer une colonne 

# Titre - Document Outline #### 
arrange(tbl_iris, Sepal.Length)  # order by ASC
arrange(tbl_iris, -Sepal.Length) # order by DESC
arrange(tbl_iris, desc(Sepal.Length))
arrange(tbl_iris, Sepal.Length, Sepal.Width)
# arrange_if(tbl_iris, is.numeric)

# Operateur PIPE   %>%  Strl-Shift-M
tbl_iris %>% 
  filter(Sepal.Length > 3, Species == "setosa") %>% 
  mutate(Size = Petal.Length * Petal.Width)  %>% 
  arrange(desc(Size))

# Group by | Summarize : mean() / median() / n() 
tbl_iris %>% 
  group_by(Species) %>% 
  summarise(Mean = mean(Sepal.Length), 
            Median = median(Sepal.Length), 
            N = n())
tbl_iris %>% 
  group_by(Species) %>% 
  summarise(Mean = mean(Sepal.Length), 
            Median = median(Sepal.Length), 
            N = n_distinct(Sepal.Length))           

# Matrice
# https://tidyr.tidyverse.org/reference/pivot_longer.html
WorldPhones %>% 
  as_tibble(rownames = "Year") %>% 
  mutate(Year = as.numeric(Year))%>% 
  pivot_longer(-Year, 
               names_to = "Region", # outcome 'longer format' column name. 
                                    # Imagine that there are n-columns to be transformed from 'wider format' to 1-column 'longer format'
               values_to = "Number" # outcome 'values' column name 
               ) %>% 
  pivot_wider(names_from = Region, values_from = Number)


require(graphics)
matplot(rownames(WorldPhones), WorldPhones, type = "b", log = "y",
        xlab = "Year", ylab = "Number of telephones (1000's)")
legend(1951.5, 80000, colnames(WorldPhones), col = 1:6, lty = 1:5,
       pch = rep(21, 7))
title(main = "World phones data: log scale for response")

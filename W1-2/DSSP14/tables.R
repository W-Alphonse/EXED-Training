library(tidyverse)
iris
as_tibble(iris)

#?filter
#help(select)
# https://dplyr.tidyverse.org/reference/filter.html

tbl_iris <- as_tibble(iris)
tbl_iris

filter(tbl_iris, Petal.Length > 3)
select(tbl_iris, Sepal.Length, Species)
select(tbl_iris, Sepal.Length, Species, everything())
select_if(tbl_iris, is.numeric)  # permet de seelctionner quel colonne nous interesse d'une manière conditionnelle
select(tbl_iris, -Species)

tbl_iris[1:10, c(2,5)] # to avoid
select(tbl_iris, 3:5)

mutate(tbl_iris, Petal.Size = Petal.Length * Petal.Width)  # Rajoute une colonne
mutate(tbl_iris, Species = str_to_title(Species))  # Modifier  une colonne
mutate(tbl_iris, Species = NULL)  # Supprimer une colonne 

# Titre - Document Outline #### 
arrange(tbl_iris, Sepal.Length)  # Order by
arrange(tbl_iris, -Sepal.Length)
arrange(tbl_iris, desc(Sepal.Length))
arrange(tbl_iris, Sepal.Length, Sepal.Width)
#arrange_if(tbl_iris, is.numeric)

# Operateur PIPE   %>%  Strl-Shift-M
tbl_iris %>% 
  filter(Sepal.Length > 3, Species == "setosa") %>% 
  mutate(Size = Petal.Length * Petal.Width)  %>% 
  arrange(desc(Size))

# Group by , Summarize
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

# MAtricce
WorldPhones %>% 
  as_tibble(rownames = "Year") %>% 
  mutate(Year = as.numeric(Year))%>% 
  pivot_longer(-Year, names_to = "Region", values_to = "Number") %>% 
  pivot_wider(names_from = Region, values_from = Number)

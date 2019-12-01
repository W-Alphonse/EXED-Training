# Ctrl+l : clear /  PIPE   %>%  Ctrl-Shift-M
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

tbl_iris[1:10, c(2,5)] # to avoid
select(tbl_iris, 3:5)  # to avoid

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
WorldPhones %>% 
  as_tibble(rownames = "Year") %>% 
  mutate(Year = as.numeric(Year))%>% 
  pivot_longer(-Year, names_to = "Region", values_to = "Number") %>% 
  pivot_wider(names_from = Region, values_from = Number)

require(graphics)
matplot(rownames(WorldPhones), WorldPhones, type = "b", log = "y",
        xlab = "Year", ylab = "Number of telephones (1000's)")
legend(1951.5, 80000, colnames(WorldPhones), col = 1:6, lty = 1:5,
       pch = rep(21, 7))
title(main = "World phones data: log scale for response")


# https://github.com/rstudio/cheatsheets/blob/master/data-visualization-2.1.pdf
# https://tutorials.iq.harvard.edu/R/Rgraphics/Rgraphics.html
# https://www.datanovia.com/en/blog/ggplot-aes-how-to-assign-aesthetics-in-ggplot2/

library(tidyverse)
theme_set(theme_minimal())


# ggplot fonctionne par couche et nuage de point => je definie mes parametres esthetiques et je travaille par couche
# Lors de l'elaboration d'un graphe, on distingue les elements suivants p.77: 
# - data : le jeu de données --> 'ggplot'
# - aesthetics : Les variables à representer + couleur + taille + .... --> 'aes'
# - geometrics : Le type de représentation graphique --> geom_...
# - statistics : statistical transformations --> stat_...
# - scales     : specify your own set of mappings from levels in the data to aesthetic values --> scale_...

ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width) +
  geom_point()

ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width, color=Species) +
  geom_point()

ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width, color=Species, shape = Species) +
  geom_point()


# list all possible values of ase
ggplot2:::.all_aesthetics

# Geometrie 'boxplot'
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length) +
  geom_boxplot()


# 'color' related to the color of the object 
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, color=Species) +
  geom_boxplot()

# 'fill' related to the color of the filled object
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot()

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot(show.legend = FALSE)  # ceci enlève la legende par type geometrie 

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe et ceci pr tte les geometrie

#
# x<- 3 rm(x)  ceci enlève la valeur de x de l'environnement de R

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot() +
  scale_fill_manual(values = c(setosa="blue", versicolor="red", virginica = "green", bb="black")) +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot() +
  scale_fill_manual(values = c(setosa="#a45f1c", 
                               versicolor="red", 
                               virginica = "green")) + 
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe
  
# The viridis (c:continuous / d:discrete) scales provide colour maps that are perceptually uniform in both colour and black-and-white. 
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

# scale_color_manual()
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot(color = "tomato") +
  scale_fill_viridis_d() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

# help.search("geom_", package = "ggplot2")

# Graph d'une seule variable; c'est un histogramme. 
# NB: histogramme(donnée continue) <> bar graph(donnée catégorisée)
ggplot(iris) + 
  aes(x=Sepal.Length) + 
  geom_histogram()

ggplot(iris) + 
  aes(x=Sepal.Length) + 
  geom_histogram(bins = 50)

ggplot(iris) + 
  # NB: aes() is a quoting function. 
  # This means that its inputs are quoted to be evaluated in the context of the data. 
  # This makes it easy to work with variables from the data frame because you can name those directly.
  aes(x=Sepal.Length, color = Species) + 
  geom_density()

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density()

ggplot(iris) + 
  aes(x=Sepal.Length, fill = "blue") + # As "blue" is not part of the data, then fill won't be BLUE as expected
  geom_density()

ggplot(iris) + 
  aes(x=Sepal.Length) + 
  geom_density(fill = "blue")

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  # param d'opacité. alpha==1 => Opaque. alpha==0 => Transparent

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species, color = Species) + 
  geom_density(alpha = 0.4)  # param d'opacité


# facet http://www.cookbook-r.com/Graphs/Facets_(ggplot2)/ 
# Divide with "sex" vertical, "day" horizontal
#          facet_grid(sex ~ day)
ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(. ~ Species)  
# '~' : pour designer une formule. 
# '~ Species' => Species est la clé de la catégorisation
# '.' : pour tout dire que c'est tt le reste
# (divideByRow ~ divideByColumns )

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(Species ~ .)  +
  theme(legend.position = "none") 

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(Species ~ .)  +
  theme(legend.position = c(1.2, 1.2))
# the legend.position = ("none", "left", "right", "bottom", "top", or two-element numeric vector)

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_boxplot()  +
  theme_minimal() 

# violin affiche la densité
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_boxplot()  +
  geom_violin() + 
  theme_minimal() 

# On change l'ordre de violin
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot()  +
  theme_minimal() 

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot()  +
  theme_minimal()  +
  theme(legend.position = "none")

# alpha pour enlever l opcité sur l'element gem_boxplot()
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot(alpha = 0)  +
  theme_minimal()  +
  theme(legend.position = "none")


ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() +
  geom_boxplot(alpha = 0)  +
  geom_point() + 
  theme_minimal()  +
  theme(legend.position = "none")

#==============================================================
# Beeswarm plots (aka column scatter plots or violin scatter plots) are a way of plotting points that would ordinarily overlap so that they fall next to each other instead. 
# In addition to reducing overplotting, it helps visualize the density of the data at each point (similar to a violin plot), 
# while still showing each data point individually.
#
# ggbeeswarm provides two different methods to create beeswarm-style plots using ggplot2. It does this by adding two new ggplot geom objects:
# geom_quasirandom: Uses a van der Corput sequence or Tukey texturing (Tukey and Tukey "Strips displaying empirical distributions: I. textured dot strips") to space the dots to avoid overplotting. This uses sherrillmix/vipor.
# geom_beeswarm   : Uses the beeswarm library to do point-size based offset.
#-------------------------------------------------------------
# install.packages("ggbeeswarm")
# install.packages("ggforce")
# ggbeeswarm::geom_beeswarm() le "::" evite de charger le package ggbeeswarm
#===============================================================
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot(alpha = 0)  +
  #
  # ggbeeswarm::geom_beeswarm() +    # The beeswarm geom is a convenient means to offset points within categories to reduce overplotting
  # ggbeeswarm::geom_quasirandom() + # The quasirandom geom is a convenient means to offset points within categories to reduce overplotting.
  ggforce::geom_sina() +           # Data visualization chart suitable for plotting any single variable in a multiclass dataset
  #
  theme_minimal()  +
  theme(legend.position = "none")

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot(alpha = 0)  +
  #  ggbeeswarm::geom_beeswarm() +  
  ggbeeswarm::geom_quasirandom() +
  # ggforce::geom_sina() +
  # labs : Modify axis, legend, and plot labels
  labs(x="Iris species", y = "Sepal length", title = "Mettre un titre bien informatif", fill = "Espèces") + 
  theme_minimal()  

# c(....) : Combine Values into a Vector or List
ggplot(iris) + 
  aes(x = Sepal.Width, y = Sepal.Length, color = Species) + 
  geom_point() + 
  scale_color_viridis_d() + 
  labs(x = "Sepal width", y = "Sepalength", title = "Sepal by sizes") + 
  theme_minimal() + 
  theme(legend.position = c(0.85, 0.68), # using relative coordinates between 0 and 1
        legend.box.background = element_rect(fill = "white",
                                             color= "black"),
        plot.title = element_text (face="bold", hjust = 0.5, color= "red")) # hjust=0.5 => In the middle

#====================================================================
# The purpose of this add-in is to let you explore your data quickly to extract the information they hold. 
# You can only create simple plots, you won’t be able to use custom scales and all the power of ggplot2.
# https://cran.r-project.org/web/packages/esquisse/readme/README.html
#====================================================================
install.packages("esquisse")
esquisse::esquisser()

# calculer la moyenne par grouppe
# dfinir le jeu de données | group by | sumarise
#dataset::iris
as_tibble(iris)
df_mean  <- iris %>% 
  group_by(Species) %>% 
  summarise(mean = mean(Sepal.Length))

ggplot(iris) + 
  aes(x = Species, y = Sepal.Length) +
  geom_boxplot(fill = "lightblue") + 
  geom_point(data = df_mean,
             mapping = aes(y=mean),  # it is combined with the default mapping at the top level of the plot. You must supply mapping if there is no plot mapping.
             color = "red", size = 5, shape = "x") +
  labs(caption = "Means are in red") +
  theme_minimal() +
  theme(legend.position = "none")

ggplot(iris) + 
  aes(x = Species, y = Sepal.Length) +
  geom_boxplot(fill = "lightblue") + 
  geom_point(data = df_mean,
             mapping = aes(y=mean),
             color = "red", size = 5, shape = "X") +
  scale_y_continuous(limits = c(0,NA)) +
  labs(caption = "Means are in red") +
  theme_minimal() +
  theme(legend.position = "none")

# jeud de données diamonds
diamonds
?diamonds

ggplot(diamonds) + 
  # NB: tout les parametres de aes(...) descendent au niveau "suivant" ds les layers
  aes(x = carat, y = price, color = cut) +
  geom_point() + 
  geom_smooth()

#diamonds$cut


ggplot(diamonds) + 
  aes(x = carat, y = price) +
  geom_point(aes(color = cut)) + 
  geom_smooth() # NB: Comme 'color' n'est pas indiqué au niveau aes() alors pas de 'color' pour geom_smooth


ggplot(diamonds) + 
  aes(x = carat, y = price) +
  geom_point(aes(color = cut)) + 
  geom_smooth(color="red") + 
  scale_y_sqrt()

ggplot(diamonds) + 
  aes(x=cut) + 
  geom_bar()

# Utilisation du wrapper "count" pour faire le count             
diamonds %>% 
  count(cut)


# There are two types of bar charts: geom_bar() and geom_col(). 
# geom_bar() makes the height of the bar proportional to the number of cases in each group (or if the weight aesthetic is supplied, the sum of the weights). 
# geom_bar() : uses stat_count()
# geom_col() : If you want the heights of the bars to represent values in the data, use geom_col()
diamonds %>% 
  count(cut) %>% 
  ggplot() +
  aes(x = cut, y = n) +  # 'n' --> COUNT
  geom_col() + 
  # n correspond à la fonction "count" par defaut
  geom_text(aes(label=n, y = n /2), color = "white")

diamonds %>% 
  count(cut) %>% 
  ggplot() +
  aes(x = cut, y = n) +
  geom_col() + 
  # ***IL ne faut pas faire ceci*** 
  geom_text(aes(label=n, y = n /2, color = "white"))

ggplot(diamonds) + 
  aes(x=cut, fill=clarity) + 
  geom_bar()

diamonds %>% 
  count(cut, color) %>%  # count (n) group by (cut + color) 
  ggplot() + 
  aes(x=cut, y= n, fill=color) + 
  geom_col()

diamonds %>% 
  count(cut, color) %>% 
  ggplot() + 
  aes(x=cut, y= n, fill=color) + 
  geom_col(position = "dodge") # Dodging preserves the vertical position of an geom while adjusting the horizontal position

diamonds %>% 
  count(cut, color) %>% 
  ggplot() + 
  aes(x=cut, y= n,   fill=color) + 
  geom_col(position = "fill")  # stacks bars and standardises each stack to have constant height.

diamonds %>% 
  count(cut) %>% 
  ggplot() + 
  aes(x = 0, y =n, fill = cut) + 
  geom_col() +
  theme_void()

diamonds %>% 
  count(cut) %>% 
  ggplot() + 
  aes(x = 0, y =n, fill = cut) + 
  geom_col() +
  coord_polar(theta="y")
  theme_void()
  
# ggplot(iris) +
#   aes(x = Petal.Length, y = Petal.Width, color=Species)

  
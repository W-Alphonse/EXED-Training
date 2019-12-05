
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
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

#
# x<- 3 rm(x)  ceci enlève la valeur de x de l'environnement

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
  
# The viridis scales provide colour maps that are perceptually uniform in both colour and black-and-white. 
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species) +
  geom_boxplot(color = "tomato") +
  scale_fill_viridis_d() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

# Graph d'une seule variable; c'est un histogramme. NB: histogramme(donnée continue) <> bar graph(donnée catégorisée)
ggplot(iris) + 
  aes(x=Sepal.Length) + 
  geom_histogram()

ggplot(iris) + 
  aes(x=Sepal.Length) + 
  geom_histogram(bins = 30)

ggplot(iris) + 
  # NB: aes() is a quoting function. 
  # This means that its inputs are quoted to be evaluated in the context of the data. This makes it easy to work with variables from the data frame because you can name those directly.
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
ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(. ~ Species)  # a-droite est "." car Il n'ya rien, on met uniquement la valeur à gauche
# '~' : pour designer une formule. 
# '~ Species' => Species est la clé de la catégorisation
# '.' : pour tout dire que c'est tt le reste
# (row ~ columns )

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(Species ~ .)  +
  theme(legend.position = "none") 
  # theme(legend.position = c(1.2, 0.2))

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

install.packages("ggbeeswarm")
install.packages("ggforce")
# ggbeeswarm::geom_beeswarm() le "::" evite de charger le package ggbeeswarm
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot(alpha = 0)  +
#  ggbeeswarm::geom_beeswarm() +  
  # ggbeeswarm::geom_quasirandom() +  
  ggforce::geom_sina() +
  theme_minimal()  +
  theme(legend.position = "none")

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill = Species) + 
  geom_violin() + 
  geom_boxplot(alpha = 0)  +
  #  ggbeeswarm::geom_beeswarm() +  
  ggbeeswarm::geom_quasirandom() +
  # ggforce::geom_sina() +
  labs(x="Iris species", y = "Sepal length", title = "Mettre un titre bien informatif", fill = "Espèces") + 
  theme_minimal()  

ggplot(iris) + 
  aes(x = Sepal.Width, y = Sepal.Length, color = Species) + 
  geom_point() + 
  scale_color_viridis_d() + 
  labs(x = "Sepal width", y = "Sepalength", title = "Sepal by sizes") + 
  theme_minimal() + 
  theme(legend.position = c(0.85, 0.68),
        legend.box.background = element_rect(fill = "white",
                                             color= "black"),
        plot.title = element_text (face="bold", hjust = 0.5, color= "red"))

# package "esquisse" 
install.packages("esquisse")
esquisse::esquisser

# calculer la moyenne par grouppe
# dfinir le jeu de données | group by | sumarise
#dataset::iris
df_mean  <- iris %>% 
  group_by(Species) %>% 
  summarise(mean = mean(Sepal.Length))

ggplot(iris) + 
  aes(x = Species, y = Sepal.Length) +
  geom_boxplot(fill = "lightblue") + 
  geom_point(data = df_mean,
             mapping = aes(y=mean),
             color = "red", size = 5, shape = "X") +
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
  # NB: tout les parametres ci dessous (...) descendent au niveau "suivant" ds les layers
  aes(x = carat, y = price, color = cut) +
  geom_point() + 
  geom_smooth()

#diamonds.$cut

# NB: tout les parametres ci dessous (...) descendent au niveau "suivant" ds les layers
ggplot(diamonds) + 
  aes(x = carat, y = price) +
  geom_point(aes(color = cut)) + 
  geom_smooth()


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


diamonds %>% 
  count(cut) %>% 
  ggplot() +
  aes(x = cut, y = n) +
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
  aes(x=cut,  fill=clarity) + 
  geom_bar()

diamonds %>% 
  count(cut, color) %>% 
  ggplot() + 
  aes(x=cut, y= n,   fill=color) + 
  geom_col()

diamonds %>% 
  count(cut, color) %>% 
  ggplot() + 
  aes(x=cut, y= n,   fill=color) + 
  geom_col(position = "dodge")

diamonds %>% 
  count(cut, color) %>% 
  ggplot() + 
  aes(x=cut, y= n,   fill=color) + 
  geom_col(position = "fill")

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

  
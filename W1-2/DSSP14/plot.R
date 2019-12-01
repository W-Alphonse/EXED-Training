library(tidyverse)
theme_set(theme_minimal())


# ggplot fonctionne par couche et nuage de point => je definie mes parametres esthetiques et je travaille par couche
ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width) +
  geom_point()

ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width, color=Species) +
  geom_point()

ggplot(iris) + 
  aes(x=Sepal.Length, y=Sepal.Width, color=Species, shape = Species) +
  geom_point()

# Geometrie 'boxplot'
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length) +
  geom_boxplot()

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, color=Species,) +
  geom_boxplot()

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
  geom_boxplot()

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
  geom_boxplot(show.legend = FALSE)  # ceci enlève la legende par type geometrie 

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
  geom_boxplot() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

# x<- 3 rm(x)  ceci enlève la valeur de x de l'environnement

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
  geom_boxplot() +
  scale_fill_manual(values = c(setosa="blue", versicolor="red", virginica = "green"))
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

  ggplot(iris) + 
    aes(x=Species, y = Sepal.Length, fill=Species,) +
    geom_boxplot() +
    scale_fill_manual(values = c(setosa="#a45f1c", 
                                 versicolor="red", 
                                 virginica = "green"))
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe
  
ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  theme(legend.position = "none") # ceci enlève tout les legendes du graphe

ggplot(iris) + 
  aes(x=Species, y = Sepal.Length, fill=Species,) +
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
  aes(x=Sepal.Length) + 
  geom_density(fill = "blue")

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density()

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  # param de transparence

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species, color = Species) + 
  geom_density(alpha = 0.4)  # param de transparence

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(. ~Species)  # a-droite est "." car Il n'ya rien, on met unquement la avleru à gauche
# le "." correspond à tout dire que c'est tt le reste

ggplot(iris) + 
  aes(x=Sepal.Length, fill = Species) + 
  geom_density(alpha = 0.4)  +
  facet_grid(Species ~.)  +
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

  
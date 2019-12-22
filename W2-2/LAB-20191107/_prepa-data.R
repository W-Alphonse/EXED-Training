
library(tidyverse)
library(lubridate)

#  Num_Acc  date  lum    agg    int   atm   col    com   adr   gps     lat  long dep 
# as_tibble(caracteristiques)
#m_caracteristiques <- select(caracteristiques, Num_Acc, date, agg, int, lum, dep )
m_caracteristiques = mutate( select(caracteristiques, Num_Acc, date, lum, agg, atm, int, col), 
        month = floor_date(date, "month") 
      )  

# Table d Usagers #########################################
m_usagers <- usagers
#Enlever colonnes hors "cat" et "an_nais"
m_usagers <- subset(m_usagers, select=-c(place,grav,sexe,trajet,secup,secuu,locp,actp,etatp,num_veh))
#Garder que les lignes conducteurs 
m_usagers_conducteur<- m_usagers[m_usagers$catu == "Conducteur",]
#m_usagers_conducteur <- m_usagers %>% filter(catu == "Conducteur")
#rajouter une colonne age
m_usagers_conducteur<-m_usagers_conducteur%>%mutate(age_conduct=2019-an_nais)
#supprimer an_nais
m_usagers_conducteur<-subset(m_usagers_conducteur, select=-c(an_nais))
m_usagers_conducteur<-subset(m_usagers_conducteur, select=-c(catu))
#supprimer "catu"
#m_usagers_conducteur<-pivot_wider(m_usagers_conducteur,age_conduct,age_conduct2)

#
m_joined <- m_usagers_conducteur %>% left_join(m_caracteristiques)
m_joined <- m_joined %>% left_join(caracteristiques)
m_joined<-subset(m_joined, select=-c(date))

#  Num_Acc  month  lum    agg    int   atm   col    com   adr   gps     lat  long dep 

# caracteristiques, Num_Acc, month, lum, agg, atm
## group names : month=month, lum+atm=meteo, col=Collision, int+agg=Topography, age=age
m_joined_cat <- select(m_joined,  # Num_Acc,  
                       month,
                       lum,atm,
                       #
                       col,
                       agg,int, 
                       #
                       age_conduct)


m_joined_cat <- m_joined_cat     %>% 
   filter(month >= "2005-01-01") %>% 
   filter(month <= "2005-02-01")

view(m_joined_cat)
# View(caracteristiques)
# View(usagers)
# View(lieux)
# View(vehicules)

library(FactoMineR)
m_joined_catMFA <- MFA(m_joined_cat, 
                 group=c(1,2,  1,2,   1),     # on indique combien de variable il y a ds chaque groupe 
                                              # EX: 1 colonne, 2 cols, 1 col, 2 cols, 1 col
                 type=c("n", "n", "n", "n", "s"),
                                              # s:variable continue, n:variable categorielle. Type de variable par groupe
                 name.group=c("Month","Meteo","Collision","Lieu", "Age_Conduct"))

m_joined_catMFA <- MFA(m_joined_cat, 
                       group=c(1,2,  1,2,   1),    
                       type=c("n", "n", "n", "n", "s"),
                       name.group=c("Month","Meteo","Collision","Lieu", "Age_Conduct"))
summary(m_joined_catMFA)


# Transform lumiere_str into lumiere_int
# lumiere_int <- c(1, 2, 3, 4, 5)
# lumiere_str <- c("Plein jour", "Crépuscule ou aube", "Nuit sans éclairage public", "Nuit avec éclairage public non allumé", "Nuit avec éclairage public allumé")
# df <- data.frame(lumiere_str, lumiere_int)
# df$lumiere_str <- as.factor(df$lumiere_str)
# factor()

# Transform agglomeration_str into agglomeration_int
# agglomeration_int <- c(1, 2)
# agglomeration_str <- c("Hors agglomération", "En agglomération")
# df <- data.frame(agglomeration_str, agglomeration_int)
# df$agglomeration_str <- as.factor(df$agglomeration_str)



# Intersection :
#   1 – Hors interse
# 1 – Hors intersection
# 2 – Intersection en X
# 3 – Intersection en T
# 4 – Intersection en Y
# 5 - Intersection à plus de 4 branches
# 6 - Giratoire
# 7 - Place
# 8 – Passage à niveau
# 9 – Autre intersection

# Conditions atmosphériques :
#   1 – Normale
# 2 – Pluie légère
# 3 – Pluie forte
# 4 – Neige - grêle
# 5 – Brouillard - fumée
# 6 – Vent fort - tempête
# 7 – Temps éblouissant
# 8 – Temps couvert
# 9 – Autre


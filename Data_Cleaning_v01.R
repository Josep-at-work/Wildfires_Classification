#### Inspiration
## Have wildfires become more or less frequent over time?
## What counties are the most and least fire-prone?
## Given the size, location and date, can you predict the cause of a fire wildfire?

# library("readxl")
library(tidyverse)
library(magrittr)
library(lubridate)
library(dplyr)
library(ggplot2)
library(ggcorrplot)

rm(list=ls())
cat("\014")

rm(path, wildfires)
path = 'Trabajo/Incendios2.csv'
wildfires<- as.data.frame(read.csv(path))

glimpse(wildfires)

########################## Select features
# Read all features, understand them and evaluate its relevance. I've talked with an expert  to understand all features
unique(wildfires$Estado)
unique(wildfires$Comunidad)
unique(wildfires$Provincia)
unique(wildfires$ComarcaIsla)
sort(unique(wildfires$Municipio))
sort(unique(wildfires$EntidadMenor))
unique(wildfires$NumeroMunicipiosAfectados) #1,2,3
table(wildfires$NumeroMunicipiosAfectados)
unique(wildfires$Hoja) #1004  604
unique(wildfires$Cuadricula)
unique(wildfires$Huso) #31  3
unique(wildfires$Datum) #"ETRS89" "WGS84"  ""
table(wildfires$NumeroPuntosInicioIncendio)
unique(wildfires$NumeroPuntosInicioIncendio) # 1 12 NA
unique(wildfires$Causa) #46 distintos son demasiados, debería haber unos 6 tops.
#Reproducidos son wildfiresque vuelven una vez apagados

unique(wildfires$Motivacion)
unique(wildfires$X)
unique(wildfires$OtrasSuperficiesNoforestales) #0
dim(wildfires)
table(wildfires$NumeroMunicipiosAfectados)

######################################
## New dataset
## I drop repeated, irrelevant information and not disperse information.
## Then I give correct data types to the selected features.
######################################

# wildfires<- as.data.frame(read.csv(path), dec='.')
wildfires%<>% select(c(ID=NumeroParte, Year=Campania, Municipality=Municipio, Square=Cuadricula,
                        X=CoordenadaX, Y=CoordenadaY, Detected=Detectado, Extinted=Extinguido,
                        Cause=Causa, Motivation=Motivacion, FarmingArea=SuperficieAgricola, 
                        NoTreesArea=SuperficieNoArbolada, ForestalTotalArea=SuperficieTotalForestal))
attach(wildfires)
glimpse(wildfires)
## Dates
wildfires%<>% mutate(Extinted=as.POSIXct(Extinted, format='%d/%b/%Y %H:%M:%S'),
                      Detected=as.POSIXct(Detected, format='%d/%b/%Y %H:%M:%S'))
## Numbers 
to_num <- function(feat){
  feat %<>% gsub(',','.',.) %>% as.numeric
} # decimals are separated by commas so first I replace them with points. 
wildfires%<>% mutate(FarmingArea=sapply(FarmingArea, FUN=to_num))
wildfires%<>% mutate(NoTreesArea=sapply(NoTreesArea, FUN=to_num))
wildfires%<>% mutate(ForestalTotalArea=sapply(ForestalTotalArea, FUN=to_num))

attach(wildfires)
glimpse(wildfires)

######################################
## Feature engineering
## Create more suited variables.
######################################

# 1. Time it took to extinguish the fire (in minutes)

wildfires %<>% mutate(Duration = difftime(Extinted, Detected, units='mins'))

class(wildfires$Duration)
sp <- ggplot(wildfires, aes(Duration, ForestalTotalArea)) +
  geom_point() 
sp
sp + xlim(0,7500) + ylim(0,500) + 
  ggtitle("Duration of fires vs Burnt Area") + xlab('Duration(min)') + ylab('Forestal Area (ha)') +
  theme_minimal()

wildfires %<>% mutate(Duration=as.numeric(Duration))

# 2. From the detected feature I will extract the month of the day of the week in which the fire occurred
library(lubridate)

wildfires %<>% mutate(Month = month(Detected), Day = wday(Detected))
table(wildfires$Day)

wildfires %<>% select(-c(Extinted,Detected))

######################################
## Dependant variable
## Use of two metrics to group the causes in bigger groups
######################################
## 1.Causes by frequency

as.data.frame(table(wildfires$Cause)) -> causes
causes %<>% arrange(desc(Freq))
head(causes)
filter(causes, Freq>5)

causes %>% filter(Freq>5) %>% ggplot(aes(Var1,Freq))+
  geom_bar(stat='identity')+
  theme(axis.text.x = element_text(angle=45, hjust = 1, size=5)) #not clean labels

causes %>% filter(Freq==1) -> drop 
# The causes which happened just once in 5 years are too specific and would only make the problem more complex, 
# it would be more difficult for the algorithms, the trade-off of loosing these observations is the removal of noise. 

## 2.Causes by damage (Forestal burned hectares)
wildfires%>% group_by(Cause) %>% summarise(area = sum(ForestalTotalArea)) -> hec_burned
arrange(hec_burned, desc(area))
hec_burned %>% filter(area<1000) %>% ggplot(aes(Cause,area))+
  geom_bar(stat='identity')+
  theme(axis.text.x = element_text(angle=45, hjust = 1, size=5)) #not clean labels

hec_burned %>% filter(Cause %in% drop$Var1)
# definitely these observations can be dropped as they are too specific and don't cause a significant damage
wildfires %<>% filter(!Cause %in% drop$Var1)

# Regroup all causes
wildfires$Cause[wildfires$Cause == "[400] Intencionado"] <- "Arson"
wildfires$Cause[wildfires$Cause == "[100] Rayo"] <- "Lightning"
wildfires$Cause[wildfires$Cause == as.character(causes[3,1])] <- "Smokers"
wildfires$Cause[wildfires$Cause == as.character(causes[6,1])] <- "Children"
# All causes related with burnign of vegetation
Veg_Bur=c(as.character(causes[5,1]), as.character(causes[7:10,1]), 
          as.character(causes[17,1]),as.character(causes[19,1]),
          as.character(causes[22,1]), as.character(causes[24:25,1]),
          as.character(causes[28,1]), as.character(causes[33,1]), as.character(causes[38:39,1]))
wildfires$Cause[wildfires$Cause %in% Veg_Bur] <- "Vegetation Burning"
# All causes related with vehicles or tools with engine
Engines = c(as.character(causes[12,1]), as.character(causes[15,1]),
            as.character(causes[26:27,1]), as.character(causes[36,1])) #vehicles and tools
wildfires$Cause[wildfires$Cause %in% Engines] <- "Engines"
# Mix of causes which aren't relevant enough to have it's own group or not specified
Miscellaneous = c(as.character(causes[11,1]), as.character(causes[14,1]),
                 as.character(causes[16,1]), as.character(causes[18,1]),
                 as.character(causes[20:21,1]), as.character(causes[23,1]), 
                 as.character(causes[29:32,1]), as.character(causes[34:35,1]))#Group the minor causes (240,600,399,320, 241, 286, 260, 270, 292,243,244,291,322)
wildfires$Cause[wildfires$Cause %in% Miscellaneous] <- "Miscellaneous"
Unknown = c(as.character(causes[4,1]), as.character(causes[13,1])) #290,500
wildfires$Cause[wildfires$Cause %in% Unknown] <- "Unknown"

ggplot(wildfires, aes(Cause)) +
  geom_bar(stat = 'count', aes(fill=Cause)) +
  theme_minimal() +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
  ggtitle("Number of fires by Cause") 

# The causes for the algorithm to predict have been reduced from 46 to 7.
# True is the causes feature has 8 groups, yet I believe that the unkown group shoulden be used to train the algorithm
# as it's an undefined group. 

# Motivation is only a description added to the arson wildfires.

wildfires %<>% select(-Motivation)
attach(wildfires)

sum(is.na(as.matrix(wildfires))==T) #no null values 

######################################
## EDA
## Explore the data
######################################

#1. Categorical(Qualitative) Variable
mun = as.data.frame(table(Municipality))
mun$color <- ifelse(mun$Freq >= 50, 'red',
                    ifelse(mun$Freq >= 30, 'tomato3',
                           ifelse(mun$Freq >= 20, 'tomato1',
                                  ifelse(mun$Freq >= 10,'orange', 'tan1'))))

ggplot(df=mun, mapping = aes(x=reorder(mun$Municipality,mun$Freq),y=mun$Freq))+
  geom_bar(stat='identity', fill=mun$color)+
  theme_minimal()+
  theme(axis.text.y = element_text(angle=0, hjust = 0, size=5))+
  xlab('Municipality') + ylab('Count') + ggtitle('Number of wildfires by Municipality') +
  coord_flip()

#Some algorithms can not deal with categorical variables, and due to the 
#amount of different municipalities, an OHE is not convinient. However
#it won't be a problem as the geolocation of the fires is also described as a
#numeric features, square, X and Y (coordenates). Also, municipality can be 
#and to a numeric aftewards.

#2. Burned area by day of the week and month
ggplot(wildfires, aes(Day, ForestalTotalArea))+
  geom_bar(stat='identity', aes(fill=Day))+
  theme_minimal() +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
  xlab('Forestal Area (ha)') + ggtitle('Total Forestal Area burnt', subtitle = 'By day of the week from 2005 to 2019')
  

ggplot(wildfires, aes(Month, ForestalTotalArea)) +
  geom_bar(stat='identity', fill=Month) + 
  scale_x_discrete(limit=c(1:12))+
  theme_minimal()+
  ylab('Forestal Area (ha)') + ggtitle('Total Forestal Area burnt', subtitle = 'By Month of the Year from 2005 to 2019')
  

#3. Causes
boxplot(Duration~Cause, ylim=c(0,1200))
ggplot(wildfires, aes(X,Duration))+
  geom_point(aes(col=Cause))+
  ylim(0,500)

#4. Correlation
# for the correlation analysis I will remove the categorical variables as well as the ID

corr <- cor(wildfires[-c(1,3,4,7)])
ggcorrplot(corr, method = 'circle', type = 'lower', lab = FALSE, hc.order = TRUE, ggtheme = ggplot2::theme_gray,
           colors = c("ivory2", "white", "indianred4")) +
  ggtitle("Correolograma") +
  theme_bw() +
  theme(axis.text.x = element_text(angle=90, hjust = 1))

p.mat <- cor_pmat(wildfires[-c(1,3,4,7)])
ggcorrplot(corr, hc.order = TRUE, method = 'circle', type = "lower",
           p.mat = p.mat, ggtheme = ggplot2::theme_dark,
           colors = c("ivory2", "white", "indianred4")) +
  ggtitle("Correlation p-values") +
  theme_bw()+
  theme(axis.text.x = element_text(angle=90, hjust = 1))

# The correlation of the total forestal area burned and the forestal area without trees is 
# almost perfectly correlated with estatistical significance. Correlation with farming are burned 
# seems to be correlated too with the total forestal area, yet I will keep it for the model building.

wildfires %<>% select(-c(ID,NoTreesArea))

table(Cause)
table(wildfires$Year)

write.csv(wildfires, 'Trabajo/wildfires.csv')

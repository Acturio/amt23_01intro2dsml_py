import numpy as np
from siuba import *
from plotnine import data as p9d
from plotnine import *
from mizani.formatters import comma_format, dollar_format
#, dollar_format
from sklearn import datasets

# import some data to play with
diamonds = p9d.diamonds
diamonds

(
 diamonds >>
 ggplot(aes(x = "cut", y = "price")) +
 geom_boxplot(aes(color = "cut"), fill = "#0c315e") +
 labs(title = "Precio por calidad de corte",
      x = "corte",
      y = "precio") +
 guides(color = guide_legend(title = "Calidad de corte")) +
 theme(axis_text_x=element_text(angle=20, size = 5)) +
 scale_y_continuous(labels=comma_format(), limits = np.array([0, 20000]))
)




(
diamonds >>
  ggplot(aes(x = "price")) +
  geom_histogram(color = "pink", fill = "purple", bins=50) +
  scale_x_continuous(labels=dollar_format(big_mark=',')) + 
  scale_y_continuous(labels=comma_format()) + 
  ggtitle("Distribución de precio") +
  labs(x = "Precio", y = "Conteo") + 
  theme(axis_text_x=element_text(angle=10))
)





(
 diamonds >>
  ggplot(aes(x = "price")) +
  geom_histogram(aes(y='stat(density)'), 
  bins = 30, fill = 'purple', color = "red") +
  geom_density(colour = "black", size = 1)
)




(
diamonds >>
 ggplot() +
 geom_boxplot(aes(x = 0, y = "price"), color= "blue", fill= "lightblue") +
 scale_y_continuous(labels = dollar_format(prefix='$', digits=0, big_mark=',')) +
 theme(axis_text_x=element_blank()) +
 ggtitle("Distribución de precio")
)




(
diamonds >>
 ggplot(aes(x = 0, y = "carat")) +
 geom_boxplot(color= "purple", fill= "pink", alpha = 0.1) +
 scale_y_continuous(labels = comma_format(digits=1)) +
 theme(axis_text_y=element_blank()) +
 coord_flip() +
 ggtitle("Distribución del peso de los diamantes")
)


#### VARIABLES CATEGÓRICAS ####

(
 diamonds >> 
  ggplot( aes( x = "cut")) + 
  geom_bar( color= "darkblue", fill= "cyan", alpha= 0.7) +
  scale_y_continuous(labels = comma_format()) +
  ggtitle("Distribución de calidad de corte") +
  labs(x = "Corte", y = "Conteo") +
  theme_dark()
)



(
diamonds >> 
  ggplot( aes( x = "clarity")) + 
  geom_bar( fill= "darkblue", color= "black", alpha= 0.7) +
  geom_text(
   aes(label=after_stat('count'), group=1),
     stat='count',
     nudge_x=0,
     nudge_y=5,
     va='bottom',
     size = 5,
     format_string='{:,.0f}') +
  scale_y_continuous(labels = comma_format(), 
  limits = np.array([0, 15000])) +
  ggtitle("Distribución claridad") 
)



(
diamonds >> 
  ggplot(mapping = aes( x = "clarity")) + 
  geom_bar(aes(fill= "cut"), color= "black", alpha= 0.9) +
  geom_text(
   aes(label=after_stat('count / sum(count) * 100'), group="cut"),
     color = "black",
     stat='count',
     va='bottom',
     #position = position_dodge2(width =1),
     #ha='center',
     size = 6,
     format_string='{:.1f}%') +
  scale_y_continuous(labels = comma_format()) +
  coord_flip() +
  facet_wrap("cut", scales = "free", ncol = 2) +
  ggtitle("Distribución claridad") #+
)


#### ANÁLISIS MULTIVARIADO ####

(
diamonds >> 
  ggplot(aes(y = "price", x = "cut", color = "cut"))  + 
  geom_jitter(size = 0.3, alpha = 0.3)
)



(
diamonds >> 
  ggplot(aes(y = "price", x = "cut", color = "cut"))  + 
  geom_boxplot(size=1, alpha= 0.3)
)



(
diamonds >> 
  ggplot(aes(x = "price" ,fill = "cut"))  + 
  geom_histogram(position = 'identity', alpha = 0.5) +
  theme(
   axis_text_x=element_text(size = 5),
   axis_text_y=element_text(size = 5),
   ) +
  facet_wrap("cut", scales = "free_y", ncol =2) +
  scale_fill_discrete(guide=False)
)



(
diamonds >> 
  ggplot(aes(x= "price" ,fill = "cut"))  + 
  geom_histogram(position = 'identity', alpha = 0.5) +
  facet_wrap("cut", ncol = 1)
)



(
diamonds >> 
  ggplot( aes(x = "carat", y = "price")) +
  geom_point(aes(color = "clarity"), size = 0.5, alpha = 0.3 ) +
  geom_smooth(stat='smooth') +
  ylim(0, 20000)
)




(
diamonds >> 
  ggplot( aes(x = "carat", y = "price")) +
  geom_point(aes(color = "clarity"), size = 0.3, alpha = 0.3 ) +
  facet_wrap("clarity")+
  geom_smooth() +
  ylim(0, 20000)
)





















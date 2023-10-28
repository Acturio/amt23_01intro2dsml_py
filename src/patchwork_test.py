import patchworklib as pw
from plotnine import *
from plotnine.data import *



g1 = (ggplot(mtcars) + geom_point(aes(x = "hp", y = "mpg")))
g1 = pw.load_ggplot(g1, figsize=(4,4))

g = (
 ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)')) + 
 geom_point() + 
 stat_smooth(method='lm') + 
 facet_wrap('~gear')
 )
g2 = pw.load_ggplot(g, figsize=(4,4))

g12 = (g1|g2)
g12.savefig("test2.png")



































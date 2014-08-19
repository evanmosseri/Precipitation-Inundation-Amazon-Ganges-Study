"""dumping ground for code for unique coloring for larger datasets"""
import numpy as np
import colorsys


SEED = 42
prng = np.random.RandomState(SEED)

#http://stackoverflow.com/a/9701141
def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + prng.rand() * 10)/100.
        saturation = (90 + prng.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
    
    
colors = {1:'SteelBlue', #January
         2:'LightSkyBlue', #February
         3:'GreenYellow', #March
         4:'ForestGreen', #April
         5:'OliveDrab', #May
         6:'Pink', #June
         7:'DeepPink', #July
         8:'MediumVioletRed',#August
         9:'Gold', #September
         10:'Orange',#October, 
         11:'OrangeRed', #November
         12: 'MidnightBlue', #December
         }
         
colors_named = {'January':'SteelBlue', #January
                'February':'LightSkyBlue', #February
                'March':'GreenYellow', #March
                'April':'ForestGreen', #April
                'May':'OliveDrab', #May
                'June':'Pink', #June
                'July':'DeepPink', #July
                'August':'MediumVioletRed',#August
                'September':'Gold', #September
                'October':'Orange',#October, 
                'November':'OrangeRed', #November
                'December': 'MidnightBlue', #December
                }
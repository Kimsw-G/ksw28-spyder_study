# -*- coding: utf-8 -*-
labels = ['개구리','돼지','개','다람쥐']
sizes = [15,30,45,10]
colors = ['yellow','gold','lightskyblue','red']
explode = (0,0.1,0,0)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.title('title')
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)
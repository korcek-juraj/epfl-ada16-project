# Command line version of the IPhython Notebook
# Used for parallelizing computations
import pandas as pd
import folium
import math
import sys

input_file_name = sys.argv[1]
df = pd.read_csv(input_file_name + '.csv')

# Map of Switzerland
map = folium.Map(location=[46.8, 8.2], zoom_start=8, tiles='Stamen Toner')

# Add outline
state_geo = r'./ch-cantons.topojson.json'
map.choropleth(geo_path=state_geo, topojson='objects.cantons',
             fill_color='#000', fill_opacity=0.05, line_opacity=.8)

for row in df.values:
    # Add circle
    if row[4] < .7:
        row[4] += .3
    color_part = row[4]
    opacity = row[4]
    if color_part < .5:
        opacity = 1 - opacity
    red = math.floor(255 * (1 - color_part))
    others = str(255 - red)
    red = str(red)
    map.circle_marker(
                    location=[row[1], row[2]], radius=10000 * math.log(row[3]),
                    popup="%s\nCount %.0f, average happyness %.2f%%" % (row[5], row[3], row[4] * 100),
                    line_color=None,
                    fill_color="rgb(" + red + ", " + others + ", " + others + ")",
                    fill_opacity=opacity*.7)

map.save(input_file_name + '.html')
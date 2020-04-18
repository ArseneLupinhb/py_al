import plotly.graph_objects as go

categories = ['processing cost', 'mechanical properties', 'chemical stability',
              'thermal stability', 'device integration']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
	r=[1, 5, 2, 2, 3],
	theta=categories,
	fill='toself',
	name='Product A'
))
fig.add_trace(go.Scatterpolar(
	r=[4, 3, 2.5, 1, 2],
	theta=categories,
	fill='toself',
	name='Product B'
))

fig.update_layout(
	polar=dict(
		radialaxis=dict(
			visible=True,
			range=[0, 5]
		)),
	showlegend=False
)

fig.show()

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Funnel(
	name='Montreal',
	y=["Website visit", "Downloads", "Potential customers", "Requested price"],
	x=[120, 60, 30, 20],
	textinfo="value+percent initial"))

fig.add_trace(go.Funnel(
	name='Toronto',
	orientation="h",
	y=["Website visit", "Downloads", "Potential customers", "Requested price", "invoice sent"],
	x=[100, 60, 40, 30, 20],
	textposition="inside",
	textinfo="value+percent previous"))

fig.add_trace(go.Funnel(
	name='Vancouver',
	orientation="h",
	y=["Website visit", "Downloads", "Potential customers", "Requested price", "invoice sent", "Finalized"],
	x=[90, 70, 50, 30, 10, 5],
	textposition="outside",
	textinfo="value+percent total"))

fig.show()

# import plotly.graph_objects as go
#
# import pandas as pd
#
# # Read data from a csv
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
#
# fig = go.Figure(data=[go.Surface(z=z_data.values)])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90)
# )
#
# fig.show()

import plotly.express as px

df = px.data.gapminder()
px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100, 100000], range_y=[25, 90])

import plotly.express as px

df = px.data.tips()
fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')
fig.show()

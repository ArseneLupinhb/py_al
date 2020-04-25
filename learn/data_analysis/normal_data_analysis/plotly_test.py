import pandas as pa
import plotly
import plotly.graph_objects as go

path_random = r'C:\Users\AL\Desktop\test\test.csv'
test_data_df = pa.read_csv(path_random)
test_data_df.head()

# trace0 = Scatter(x=[1,2,3,4], y=[1,2,3,4])
# trace1 = Scatter(x=[1,2,3,4], y=[5,6,7,8])
# data = [trace0, trace1]
# plotly.offline.plot(data, filename='tfh.html')

trace = [go.Pie(labels=test_data_df.index.tolist(), values=test_data_df.score.tolist(), hole=0.2)]
fig = go.Figure(data=trace)
pyplot = plotly.offline.plot
pyplot(fig)

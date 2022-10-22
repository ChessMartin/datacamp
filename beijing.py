import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
import os
import io
import pydeck as pdk
import json
import streamlit as st
import math
from datetime import time

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from typing import Tuple, List
#from networkx.classes.multidigraph import MultiDiGraph


### GLOBAL VARIABLE ###
JAM_MAP_PATH = "assets/data/traffic_jam_map.csv"
TAXIS_FREQUENCY_PATH = "assets/data/allTaxisByDayByHour.csv"

with open("assets/utils/metrics.json", 'r', encoding='utf-8') as f:
    TAXIS_METRICS = json.loads(f.read())


def create_vector_data(df):
    vector = []
    norm_array = []

    # computing the vector's norm
    for i in range(len(df)):
        a = [df.latitude[i], df.longitude[i]]
        l1 = norm(a, 1)
        norm_array.append(l1)
        vector.append(a)

    # creating the new dataframe with vector and norm inside

    df.date_time = pd.to_datetime(df.date_time)
    geo = pd.DataFrame({
        "date_time": list(df.date_time),
        "vector": list(vector),
        "norm": list(norm_array)
    })
    geo['hour'] = [x.hour for x in df['date_time']]
    geo.sort_values(by='hour', ascending=True)

    return geo

# Calculating Manhattan Distance from Scratch

def manhattan_distance(point1, point2):
    distance = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        distance += absolute_difference

    return distance

def weight(df):
    distance = [0]
    for i in range(len(df.vector)-1):
        distance.append(manhattan_distance(
            df.vector[i], df.vector[i+1])*100000)
    df['distance_meters'] = list(distance)
    traffic_jam_coord = []
    traffic_jam_lat = []
    traffic_jam_lon = []
    temp = []
    temp2 = []
    date_time = []
    counter = 0
    weight_list = []
    total_lat = 0
    total_lon = 0
    mean_lat = 0
    mean_lon = 0
    for i in range(len(df.vector)):
        if 30 < df.distance_meters[i] < 50:
            counter += 1
            temp.append(df.vector[i])
            temp2.append(df.date_time[i])
        elif df.distance_meters[i] == 0:
            for i in range(len(temp)):
                total_lat += temp[i][0]
                total_lon += temp[i][1]
                mean_lat = total_lat/len(temp)
                mean_lon = total_lon/len(temp)
            traffic_jam_coord.append([mean_lat, mean_lon])
            traffic_jam_lat.append(mean_lat)
            traffic_jam_lon.append(mean_lon)
            weight_list.append(counter)
            if len(temp2) > 0:
                date_time.append(temp2[i])
            temp2 = []
            total_lat = 0
            total_lon = 0
            mean_lat = 0
            mean_lon = 0
            counter = 0
            temp = []
    return traffic_jam_coord, weight_list, traffic_jam_lat, traffic_jam_lon, date_time

def calc_weight(df):
    # creating the vector dataframe
    df_vector = create_vector_data(df)
    distance = [0]
    # calculating distances between 2 locations
    for i in range(len(df_vector.vector)-1):
        distance.append(manhattan_distance(
            df_vector.vector[i], df_vector.vector[i+1])*100000)

    # adding the new column distance_meters in the datafram
    df_vector['distance_meters'] = list(distance)

    # calculating the traffic jam coordinates as well as their weight
    traffic_jam_coord, weight_list, traffic_jam_lat, traffic_jam_lon, date_time = weight(
        df_vector)

    # removing static point from the traffic jam list
    traffic_jam_coord_clean = []
    for i in range(len(traffic_jam_coord)):
        if weight_list[i] != 0:
            traffic_jam_coord_clean.append(traffic_jam_coord[i])

    weight_list = [i for i in weight_list if i != 0]
    traffic_jam_lat = [i for i in traffic_jam_lat if i != 0]
    traffic_jam_lon = [i for i in traffic_jam_lon if i != 0]

    # creating the new dataframe
    weight_map_total = pd.DataFrame({
        "date_time": list(date_time),
        "vector": list(traffic_jam_coord_clean),
        "traffic_intensity": list(weight_list),
        "latitude": list(traffic_jam_lat),
        "longitude": list(traffic_jam_lon)
    })
    weight_map_total.date_time = pd.to_datetime(weight_map_total.date_time)

    return weight_map_total

@st.experimental_memo
def complete_weight_map():
    dir_path = r'taxi_log_2008_by_id/'
    df_list = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            df_list.append(path)

    df_1 = pd.read_csv('taxi_log_2008_by_id/1.txt',
                       names=['taxi_id', 'date_time', 'longitude', 'latitude'])
    result = calc_weight(df_1)

    for i in range(3000, 3200):
        df = pd.read_csv('taxi_log_2008_by_id/{}'.format(df_list[i]), names=[
                         'taxi_id', 'date_time', 'longitude', 'latitude'])
        traffic_jam = calc_weight(df)
        result = pd.concat([result, traffic_jam])

    result.loc[result.traffic_intensity.between(1, 2), 'color'] = 'low traffic'
    result.loc[result.traffic_intensity.between(
        3, 5), 'color'] = 'medium traffic'
    result.loc[result.traffic_intensity > 5, 'color'] = 'high traffic'

    #result.to_csv("traffic_jam_map.csv", index=False)

    return result

@st.experimental_memo
def filterdata(df, hour_selected):
    return df[df["date_time"].dt.hour == hour_selected]

@st.experimental_memo
def mpoint(lat, lon):
    return (np.average(lat), np.average(lon))

COLOR_RANGE = [
    [102, 205, 0],
    [255, 255, 0],
    [255, 0, 0]
]

def color_scale(val):
    if 1 <= val <= 2:
        return COLOR_RANGE[0]
    elif 3 <= val <= 5:
        return COLOR_RANGE[1]
    else:
        return COLOR_RANGE[2]

def traffic_scatter(df):
    df["fill_color"] = df.traffic_intensity.apply(lambda x: color_scale(x))
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=39.9042,
            longitude=116.4074,
            zoom=11,
            pitch=80,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position=['longitude', 'latitude'],
                get_fill_color='fill_color',
                get_radius=50,
            ),
        ],
    ))

def hash_io(input_io):
    data = input_io.read()
    input_io.seek(0)
    if isinstance(data, str):
        data = data.encode("utf-8")
        return hashlib.md5(data).hexdigest()

@st.cache(hash_funcs={io.BytesIO: hash_io, io.StringIO: hash_io})
def load_vectors(f_path):
    df = pd.read_csv(f_path, header = 0)
    df.distance = pd.to_numeric(df.distance, errors='coerce')
    df.deltatime = pd.to_numeric(df.deltatime, errors='coerce')
    df.theta = pd.to_numeric(df.theta, errors='coerce')
    df.lat = pd.to_numeric(df.lat, errors='coerce')
    df.lon = pd.to_numeric(df.lon, errors='coerce')
    df.speed = pd.to_numeric(df.speed, errors='coerce')
    df.dropna(inplace=True)
    return df

def prediction_model(df):
    df = load_vectors('./vectors.csv')
    x = df[['distance', 'theta', 'lat', 'lon']]
    y = df[['deltatime']]
    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    reg = model.fit(x, y)
    return reg

def load_data(PATH):
    return pd.read_csv(PATH)

# IF THE SLIDER CHANGES, UPDATE THE QUERY PARAM
def update_query_params():
    hour_selected = st.session_state["gps_time"]
    st.experimental_set_query_params(gps_time=hour_selected)

# -----------------------------------------------------
######              MAIN              ################
# -----------------------------------------------------

# Load dataset
highest_congestion_data = load_data(JAM_MAP_PATH)
car_density_data = load_data(TAXIS_FREQUENCY_PATH)

# Data Preparation
highest_congestion_data.date_time = pd.to_datetime(
    highest_congestion_data.date_time)

# Secure process
if not st.session_state.get("url_synced", False):
    try:
        gps_time = int(st.experimental_get_query_params()["gps_time"][0])
        st.session_state["gps_time"] = gps_time
        st.session_state["url_synced"] = True
    except KeyError:
        pass

st.title("Beijing taxi data analysis")

# Initialisation of tabs
tab1, tab2, tab3 = st.tabs(
    ["📈 Highest Congestion", "🗃 Average Time", "📈 Average Car Density"])

# Highest Congestion
with tab1:
    st.subheader("The street of highest congestion rate")
    hour_selected = st.slider(
        "Select hour of pickup", 0, 23, key="gps_time", on_change=update_query_params
    )
    st.write(
        f"""**All Beijing Taxi data from {hour_selected}:00 and {(hour_selected + 1) % 24}:00**"""
    )
    traffic_scatter(filterdata(highest_congestion_data, hour_selected))

    st.write("")
    st.write('This version of the app is using a pre generated map traffic csv created by our functions in order to load the app faster')
    st.write('However you can still run the file locacally and with a simple modification render a traffic_jam dataframe file each time you run the streamlit')




with tab2:
    st.subheader("Transport time estimation")
    st.write("The inputs are not very user friendly but it allows the user to understand how the model is working, it relies on vectors, that are represented by a point, a norm and an angle, it is also ponderated by the time of the day and the day of the week")
    df = load_vectors('./vectors.csv')
    reg = prediction_model(df)
    lat = st.slider("Select your latitude",
                    float(np.percentile(df.lat, 25)),
                    float(np.percentile(df.lat, 75)),
                    float(np.percentile(df.lat, 50)))

    lon = st.slider("Select your longitude",
                    float(np.percentile(df.lon, 25)),
                    float(np.percentile(df.lon, 75)),
                    float(np.percentile(df.lon, 50)))

    theta = st.slider("Select the angle you are from the position",
            (-1)*math.pi, math.pi, 0.0)

    distance = st.slider("Select the distance you are from the point",
            1.0, 30.0, 15.0)

    day = st.selectbox('What day of the week is it ?',
            ('Monday', 'Tuesday', 'Wesdnesday', 'Thrusday',
             'Friday', 'Saturday', 'Sunday'))

    time = st.slider(
        "What time is it ?",
        value=(time(12, 45)))

    values = pd.DataFrame({
                           'distance': [distance],
                           'theta':    [theta],
                           'lat':      [lat],
                           'lon':      [lon]
            })
    print(values)
    st.write("The time predicted for the itenarary is : {} minutes".format(
            reg.predict(values)[0][0] * 60))


# Average Car Density
with tab3:
    st.subheader("The average car density of the city")
    st.write("One of the project's goal is to estimate the density of cars in the city of Beijing. However, our dataset only provides us the number of cabs at a time t. Thus, we will make an approximation using the New York's taxis number.")
    row3_1, row3_2 = st.columns((2, 2))
    with row3_1:
        st.write("We have several metrics at our disposal for New York City:")
        st.write("- The average number of Taxis: ", 14_000)
        st.write("- The number of buses in the city: ", 1_400_000)

        n_day_selected = st.selectbox(
            "Select day of car density",
            ("Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"),
             key="ny_day"
        )

        st.write(f"""**Beijing avegare density at {n_day_selected}: car/km2**""")

        NY_TAXIS_COEFF = (TAXIS_METRICS["NewYork"]["TAXIS_IN_NY"]
                          * 100 / TAXIS_METRICS["NewYork"]["CAR_IN_NY"]) / 100

        st.area_chart(car_density_data[car_density_data["dayOfWeek"] == n_day_selected].apply(lambda x: (
            x.taxis_id / NY_TAXIS_COEFF) / TAXIS_METRICS["Beijing"]["KM_ROAD_BEJING"],
            axis=1
        ))

    with row3_2:
        st.write("We have several metrics at our disposal for Beijing:")
        st.write("- The average number of Taxis: ", 66_000)
        st.write("- The number of buses in the city: ", 3_300_000)

        b_day_selected = st.selectbox(
            "Select day of car density",
            ("Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"),
             key="bj_day"
        )

        st.write(f"""**Beijing avegare density at {b_day_selected}: car/km2**""")

        st.area_chart(car_density_data[car_density_data["dayOfWeek"] == b_day_selected]
                      .apply(lambda x: (x.taxis_id * TAXIS_METRICS["Beijing"]
                                        ["COEFF_TAXIS_BY_ALL_TAXIS"]
                                        * TAXIS_METRICS["Beijing"]["COEFF_TAXIS_BY_ALL_CAR"])
                             / TAXIS_METRICS["Beijing"]["KM_ROAD_BEJING"],
                             axis=1
                             )
                      )




from dash import Dash, html, dcc, Input, Output, State, callback_context, ALL, no_update
import plotly.graph_objs as go
import geopandas as gpd
from shapely.geometry import Point, LineString
import r5py
from datetime import datetime, date
import numpy as np
import os
import subprocess 
from pathlib import Path


# Set environment variables for R5
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["R5_JAR"] = "../R5.jar"

# # Check Java installation
java_path = os.environ.get("JAVA_HOME", "Not Set")
print(f"JAVA_HOME is set to: {java_path}")

# Verify Java version
try:
    java_version = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    print("Java is installed successfully:", java_version.stderr)
except Exception as e:
    print("Java installation check failed:", e)


TRANSIT_FARE_PER_RIDE = 1.00

def calculate_fare(base_fare, cost_per_mile, cost_per_minute, service_fee, distance, duration, additional_fees=0):
    fare = base_fare + (cost_per_mile * distance) + (cost_per_minute * duration) + service_fee + additional_fees
    return fare

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

ROOT = Path(__file__).resolve().parents[1]
osm_path  = ROOT / "durham_new.osm.pbf"
gtfs_path = ROOT / "gtfs.zip"

for p in (osm_path, gtfs_path):
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")
        
# gtfs_path = 'gtfs.zip'
# osm_path = 'durham_new.osm.pbf'
transport_network = r5py.TransportNetwork(osm_path, [gtfs_path])

lat_start, lat_end = 35.88, 36.08
lon_start, lon_end = -78.98, -78.85

lat_points = np.linspace(lat_start, lat_end, 50)
lon_points = np.linspace(lon_start, lon_end, 50)
lat, lon = np.meshgrid(lat_points, lon_points)
lat = lat.flatten()
lon = lon.flatten()

fig = go.Figure(go.Scattermapbox(
    mode='markers',
    lon=lon,
    lat=lat,
    marker={'size': 5, 'opacity': 0},
    hoverinfo='none',
    showlegend=False
))

fig.update_layout(
    mapbox={
        'style': "carto-positron",
        'center': {'lat': 35.9940, 'lon': -78.8986},
        'zoom': 11
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    clickmode='event+select'
)

hours_options = [{'label': f'{i:02d}', 'value': f'{i:02d}'} for i in range(24)]
minutes_options = [{'label': f'{i:02d}', 'value': f'{i:02d}'} for i in range(0, 60, 5)]
optimization_options = [
    {'label': 'Total Time', 'value': 'total_time'},
    {'label': 'Number of Transfers', 'value': 'transfers'},
    {'label': 'Wait Time', 'value': 'wait_time'},
    {'label': 'Walking/Biking Distance', 'value': 'walking_biking_distance'}
]

app.layout = html.Div(className='main-container', children=[
    html.Div([
        html.H1("Trip Planner", className='title'),
    ], className='title-container'),
    html.Div([
        html.Div([
            html.Label('Enter your origin (lat, lon):', className='input-label'),
            dcc.Input(id='input-origin', type='text', placeholder='Enter origin lat, lon', className='input-box'),
            html.Label('Enter your destination (lat, lon):', className='input-label'),
            dcc.Input(id='input-destination', type='text', placeholder='Enter destination lat, lon', className='input-box'),
            html.Button('Add Destination', id='add-destination-button', n_clicks=0, className='hidden-button'),
            html.Div(id='new-destinations-container'),
            html.Label('Select Trip Mode:', className='input-label'),
            dcc.RadioItems(
                id='trip-mode-radio',
                options=[
                    {'label': 'Same Mode', 'value': 'same'},
                    {'label': 'Different Mode', 'value': 'different'}
                ],
                value='same',
                className='radio-items'
            ),
            html.Br(),
            html.Label('Departure Time:', className='input-label'),
            dcc.RadioItems(
                id='departure-time-radio',
                options=[
                    {'label': 'Leave Now', 'value': 'now'},
                    {'label': 'Choose Time', 'value': 'future'}
                ],
                value='now',
                className='radio-items'
            ),
            html.Div(id='departure-time-div', children=[
                dcc.DatePickerSingle(
                    id='departure-date-picker',
                    min_date_allowed=date.today(),
                    initial_visible_month=date.today(),
                    date=date.today(),
                    className='datepicker'
                ),
                html.Div([
                    dcc.Dropdown(id='departure-hour', options=hours_options, placeholder='Hour', className='time-dropdown'),
                    dcc.Dropdown(id='departure-minute', options=minutes_options, placeholder='Minute', className='time-dropdown')
                ], className='time-container')
            ], className='hidden-div'),
            html.Br(),
            html.Label('Optimization Criteria:', className='input-label'),
            dcc.Dropdown(id='optimization-criteria', options=optimization_options, value='total_time', className='dropdown'),
            html.Button('Calculate Travel Time', id='calculate-button', n_clicks=0, className='calculate-button'),
            html.Button('Start Over', id='start-over-button', n_clicks=0, className='start-over-button')
        ], className='left-panel'),
        html.Div([
            html.Label('Select your transport mode:', className='input-label'),
            html.Div([
                html.Button('Transit + Walk 🚶🏻🚌', id='mode-transit-walk', n_clicks=0, className='mode-button'),
                html.Button('Transit + Bike 🚲🚌', id='mode-transit-bike', n_clicks=0, className='mode-button'),
                html.Button('Car 🚗', id='mode-car', n_clicks=0, className='mode-button'),
                html.Button('Bike 🚲', id='mode-bike', n_clicks=0, className='mode-button'),
                html.Button('Shared Ride 🚕', id='mode-shared-ride', n_clicks=0, className='mode-button'),
                html.Button('Walk 🚶🏻', id='mode-walk', n_clicks=0, className='mode-button'),
            ], className='mode-button-container'),
            html.Div(id='travel-time', className='info-container'),
            html.Div(id='selected-modes-summary', className='info-container')
        ], className='right-panel'),
    ], id='main-content', className='main-content'),
    html.Div([
        dcc.Graph(id='map-graph', figure=fig, className='map')
    ], className='map-container'),
])

@app.callback(
    [Output('add-destination-button', 'style')],
    [Input('input-origin', 'value'),
     Input('input-destination', 'value')]
)
def toggle_add_destination_button(origin, destination):
    if origin and destination:
        return [{'margin': '5px', 'background-color': '#1c293a', 'color': 'white'}]
    return [{'display': 'none'}]

@app.callback(
    Output('new-destinations-container', 'children'),
    [Input('add-destination-button', 'n_clicks')],
    [State('new-destinations-container', 'children')]
)
def add_new_destination(n_clicks, children):
    if children is None:
        children = []
    if n_clicks > 0:
        new_destination = html.Div([
            html.Label(f'Enter your destination {n_clicks} (lat, lon):', className='input-label'),
            dcc.Input(id={'type': 'dynamic-destination', 'index': n_clicks}, type='text', placeholder=f'Enter destination {n_clicks} lat, lon', className='input-box')
        ], style={'margin-top': '10px'})
        children.append(new_destination)
    return children

@app.callback(
    Output('segment-mode-container', 'style'),
    [Input('trip-mode-radio', 'value')]
)
def toggle_segment_mode_container(trip_mode):
    if trip_mode == 'different':
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('segment-mode-container', 'children'),
    [Input('trip-mode-radio', 'value'),
     Input('add-destination-button', 'n_clicks')],
    [State('input-origin', 'value'),
     State('input-destination', 'value'),
     State({'type': 'dynamic-destination', 'index': ALL}, 'value')]
)
def update_segment_mode_dropdowns(trip_mode, n_clicks, origin, destination, dynamic_destinations):
    if trip_mode == 'same' or not origin or not destination:
        return []

    segments = [{'label': 'Transit + Walk 🚶🏻🚌', 'value': 'TRANSIT_WALK'},
                {'label': 'Transit + Bike 🚲🚌', 'value': 'TRANSIT_BIKE'},
                {'label': 'Car 🚗', 'value': 'CAR'},
                {'label': 'Bike 🚲', 'value': 'BICYCLE'},
                {'label': 'Shared Ride 🚕', 'value': 'SHARED_RIDE'},
                {'label': 'Walk 🚶🏻', 'value': 'WALK'}]

    segment_dropdowns = []
    total_segments = len(dynamic_destinations) + 1
    for i in range(total_segments):
        segment_dropdowns.append(html.Div([
            html.Label(f'Segment {i + 1} Mode:', className='input-label'),
            dcc.Dropdown(id={'type': 'segment-mode', 'index': i}, options=segments, value='CAR', className='dropdown')
        ]))

    return segment_dropdowns

@app.callback(
    [Output('input-origin', 'value'),
     Output('input-destination', 'value'),
     Output('map-graph', 'figure'),
     Output({'type': 'dynamic-destination', 'index': ALL}, 'value'),
     Output('travel-time', 'children'),
     Output('mode-transit-walk', 'style'),
     Output('mode-transit-bike', 'style'),
     Output('mode-car', 'style'),
     Output('mode-bike', 'style'),
     Output('mode-shared-ride', 'style'),
     Output('mode-walk', 'style'),
     Output('selected-modes-summary', 'children')],
    [Input('map-graph', 'clickData'),
     Input('calculate-button', 'n_clicks'),
     Input('start-over-button', 'n_clicks'),
     Input('mode-transit-walk', 'n_clicks'),
     Input('mode-transit-bike', 'n_clicks'),
     Input('mode-car', 'n_clicks'),
     Input('mode-bike', 'n_clicks'),
     Input('mode-shared-ride', 'n_clicks'),
     Input('mode-walk', 'n_clicks'),
     Input('optimization-criteria', 'value')],
    [State('input-origin', 'value'),
     State('input-destination', 'value'),
     State({'type': 'dynamic-destination', 'index': ALL}, 'value'),
     State({'type': 'segment-mode', 'index': ALL}, 'value'),
     State('trip-mode-radio', 'value'),
     State('departure-time-radio', 'value'),
     State('departure-date-picker', 'date'),
     State('departure-hour', 'value'),
     State('departure-minute', 'value'),
     State('map-graph', 'figure'),
     State('mode-transit-walk', 'style'),
     State('mode-transit-bike', 'style'),
     State('mode-car', 'style'),
     State('mode-bike', 'style'),
     State('mode-shared-ride', 'style'),
     State('mode-walk', 'style')]
)
def update_inputs_and_calculate_travel_time(clickData, n_clicks, start_over_clicks, transit_walk_clicks, transit_bike_clicks, car_clicks, bike_clicks, shared_ride_clicks, walk_clicks, optimization_criteria, origin, destination, dynamic_destinations, segment_modes, trip_mode, departure_time_radio, departure_date, departure_hour, departure_minute, current_figure, transit_walk_style, transit_bike_style, car_style, bike_style, shared_ride_style, walk_style):

    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    mode_styles = {
        'mode-transit-walk': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'},
        'mode-transit-bike': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'},
        'mode-car': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'},
        'mode-bike': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'},
        'mode-shared-ride': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'},
        'mode-walk': {'margin': '5px', 'background-color': '#1c293a', 'color': 'white'}
    }

    def parse_coordinates(coords_str):
        try:
            lat, lon = map(float, coords_str.split(','))
            return lat, lon
        except:
            return None

    if trigger == 'map-graph' and clickData:
        filled_boxes = len([d for d in dynamic_destinations if d])
        if not origin:
            coords = clickData['points'][0]
            lat, lon = coords['lat'], coords['lon']
            coords_str = f"{lat}, {lon}"
            new_figure = go.Figure(data=current_figure['data'], layout=current_figure['layout'])
            new_figure.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=[lon],
                lat=[lat],
                marker={'size': 10, 'color': 'red'},
                text=["Origin"],
                textposition="bottom right",
                showlegend=False
            ))
            return coords_str, destination, new_figure, dynamic_destinations, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        if not destination:
            coords = clickData['points'][0]
            lat, lon = coords['lat'], coords['lon']
            coords_str = f"{lat}, {lon}"
            new_figure = go.Figure(data=current_figure['data'], layout=current_figure['layout'])
            new_figure.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=[float(origin.split(',')[1]), lon],
                lat=[float(origin.split(',')[0]), lat],
                marker={'size': 10, 'color': ['red', '#93979C']},
                text=["Origin", "Destination"],
                textposition="bottom right",
                showlegend=False
            ))
            return origin, coords_str, new_figure, dynamic_destinations, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        if filled_boxes < len(dynamic_destinations):
            for i in range(len(dynamic_destinations)):
                if not dynamic_destinations[i]:
                    coords = clickData['points'][0]
                    lat, lon = coords['lat'], coords['lon']
                    coords_str = f"{lat}, {lon}"
                    dynamic_destinations[i] = coords_str
                    new_figure = go.Figure(data=current_figure['data'], layout=current_figure['layout'])
                    new_figure.add_trace(go.Scattermapbox(
                        mode='markers+text',
                        lon=[float(origin.split(',')[1]), float(destination.split(',')[1]), lon],
                        lat=[float(origin.split(',')[0]), float(destination.split(',')[0]), lat],
                        marker={'size': 10, 'color': ['red', '#93979C', '#93979C']},
                        text=["Origin", "Destination", f"Destination {i + 1}"],
                        textposition="bottom right",
                        showlegend=False
                    ))
                    return origin, destination, new_figure, dynamic_destinations, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        return origin, destination, current_figure, dynamic_destinations, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trigger == 'start-over-button':
        return '', '', fig, [], no_update, mode_styles['mode-transit-walk'], mode_styles['mode-transit-bike'], mode_styles['mode-car'], mode_styles['mode-bike'], mode_styles['mode-shared-ride'], mode_styles['mode-walk'], ''

    mode_of_travel = 'TRANSIT_WALK'
    if trip_mode == 'same':
        if trigger == 'mode-transit-bike' or (trigger == 'optimization-criteria' and transit_bike_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'TRANSIT_BIKE'
            transit_bike_style['background-color'] = '#74bf0c'
            transit_bike_style['color'] = 'black'
        elif trigger == 'mode-car' or (trigger == 'optimization-criteria' and car_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'CAR'
            car_style['background-color'] = '#74bf0c'
            car_style['color'] = 'black'
        elif trigger == 'mode-bike' or (trigger == 'optimization-criteria' and bike_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'BICYCLE'
            bike_style['background-color'] = '#74bf0c'
            bike_style['color'] = 'black'
        elif trigger == 'mode-transit-walk' or (trigger == 'optimization-criteria' and transit_walk_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'TRANSIT_WALK'
            transit_walk_style['background-color'] = '#74bf0c'
            transit_walk_style['color'] = 'black'
        elif trigger == 'mode-shared-ride' or (trigger == 'optimization-criteria' and shared_ride_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'SHARED_RIDE'
            shared_ride_style['background-color'] = '#74bf0c'
            shared_ride_style['color'] = 'black'
        elif trigger == 'mode-walk' or (trigger == 'optimization-criteria' and walk_style['background-color'] == '#74bf0c'):
            mode_of_travel = 'WALK'
            walk_style['background-color'] = '#74bf0c'
            walk_style['color'] = 'black'
        elif trigger == 'calculate-button':
            # Determine mode_of_travel based on max clicks if no direct trigger was found
            if car_clicks >= bike_clicks and car_clicks >= transit_walk_clicks and car_clicks >= transit_bike_clicks and car_clicks >= shared_ride_clicks and car_clicks >= walk_clicks:
                mode_of_travel = 'CAR'
                mode_styles['mode-car']['background-color'] = '#74bf0c'
                mode_styles['mode-car']['color'] = 'black'
            elif bike_clicks >= car_clicks and bike_clicks >= transit_walk_clicks and bike_clicks >= transit_bike_clicks and bike_clicks >= shared_ride_clicks and bike_clicks >= walk_clicks:
                mode_of_travel = 'BICYCLE'
                mode_styles['mode-bike']['background-color'] = '#74bf0c'
                mode_styles['mode-bike']['color'] = 'black'
            elif transit_walk_clicks >= car_clicks and transit_walk_clicks >= bike_clicks and transit_walk_clicks >= transit_bike_clicks and transit_walk_clicks >= shared_ride_clicks and transit_walk_clicks >= walk_clicks:
                mode_of_travel = 'TRANSIT_WALK'
                mode_styles['mode-transit-walk']['background-color'] = '#74bf0c'
                mode_styles['mode-transit-walk']['color'] = 'black'
            elif transit_bike_clicks >= car_clicks and transit_bike_clicks >= bike_clicks and transit_bike_clicks >= transit_walk_clicks and transit_bike_clicks >= shared_ride_clicks and transit_bike_clicks >= walk_clicks:
                mode_of_travel = 'TRANSIT_BIKE'
                mode_styles['mode-transit-bike']['background-color'] = '#74bf0c'
                mode_styles['mode-transit-bike']['color'] = 'black'
            elif shared_ride_clicks >= car_clicks and shared_ride_clicks >= bike_clicks and shared_ride_clicks >= transit_walk_clicks and shared_ride_clicks >= transit_bike_clicks and shared_ride_clicks >= walk_clicks:
                mode_of_travel = 'SHARED_RIDE'
                mode_styles['mode-shared-ride']['background-color'] = '#74bf0c'
                mode_styles['mode-shared-ride']['color'] = 'black'
            elif walk_clicks >= car_clicks and walk_clicks >= bike_clicks and walk_clicks >= transit_walk_clicks and walk_clicks >= transit_bike_clicks and walk_clicks >= shared_ride_clicks:
                mode_of_travel = 'WALK'
                mode_styles['mode-walk']['background-color'] = '#74bf0c'
                mode_styles['mode-walk']['color'] = 'black'

    if trigger in ['calculate-button', 'mode-transit-walk', 'mode-transit-bike', 'mode-car', 'mode-bike', 'mode-shared-ride', 'mode-walk', 'optimization-criteria'] and n_clicks > 0 and origin and destination:
        try:
            total_travel_time_seconds = 0
            total_distance_miles = 0
            total_cost = 0.0
            all_segments_details = []
            all_route_traces = []

            coords_list = [origin] + [destination] + dynamic_destinations
            coords_list = [coords for coords in coords_list if coords]

            def meters_to_miles(meters):
                return meters * 0.000621371

            def select_best_option(details, criterion):
                details['total_time'] = details['travel_time'] + details['wait_time']
                walking_biking_time = details.groupby('option', group_keys=False).apply(lambda x: x.iloc[0]['travel_time'] + x.iloc[-1]['travel_time']).reset_index(name='walking_biking_time')
                details['num_transfers'] = details.groupby('option')['segment'].transform('count') - 3

                # Compute walking_biking_distance similar to walking_biking_time
                walking_biking_distance = details.groupby('option', group_keys=False).apply(
                    lambda x: x.iloc[0]['distance'] + x.iloc[-1]['distance']
                ).reset_index(name='walking_biking_distance')

                grouped = details.groupby('option').agg(
                    total_time=('total_time', 'sum'),
                    wait_time=('wait_time', 'sum'),
                    num_transfers=('num_transfers', 'first'),
                    distance=('distance', 'sum')
                ).reset_index().merge(walking_biking_time, on='option').merge(walking_biking_distance, on='option')

                if criterion == 'total_time':
                    return grouped.loc[grouped['total_time'].idxmin()]
                elif criterion == 'transfers':
                    return grouped.loc[grouped['num_transfers'].idxmin()]
                elif criterion == 'wait_time':
                    return grouped.loc[grouped['wait_time'].idxmin()]
                elif criterion == 'walking_biking_distance':
                    return grouped.loc[grouped['walking_biking_distance'].idxmin()]
                else:
                    return grouped.loc[grouped['total_time'].idxmin()]

            for i in range(len(coords_list) - 1):
                origin_coords = parse_coordinates(coords_list[i])
                destination_coords = parse_coordinates(coords_list[i + 1])
                if not origin_coords or not destination_coords:
                    raise ValueError("Invalid coordinates")

                origin_lat, origin_lon = origin_coords
                destination_lat, destination_lon = destination_coords
                origin_point = Point(origin_lon, origin_lat)
                destination_point = Point(destination_lon, destination_lat)
                origins = gpd.GeoDataFrame([{'id': 'origin', 'geometry': origin_point}], crs="EPSG:4326")
                destinations = gpd.GeoDataFrame([{'id': 'destination', 'geometry': destination_point}], crs="EPSG:4326")

                if departure_time_radio == 'future' and departure_date and departure_hour and departure_minute:
                    departure_datetime = datetime.combine(datetime.fromisoformat(departure_date).date(), datetime.strptime(f'{departure_hour}:{departure_minute}', '%H:%M').time())
                else:
                    departure_datetime = datetime.now()

                current_segment_mode = segment_modes[i] if trip_mode == 'different' and i < len(segment_modes) else mode_of_travel

                if current_segment_mode == 'SHARED_RIDE':
                    car_itineraries_computer = r5py.DetailedItinerariesComputer(
                        transport_network,
                        origins=origins,
                        destinations=destinations,
                        departure=departure_datetime,
                        transport_modes=['CAR']
                    )
                    car_travel_details = car_itineraries_computer.compute_travel_details()
                    min_car_travel_time = car_travel_details['travel_time'].min()

                    wait_time = np.random.normal(loc=8, scale=3)
                    wait_time = max(1, min(15, wait_time))
                    additional_travel_time = np.random.exponential(scale=2)
                    additional_travel_time = max(1, min(8, additional_travel_time))

                    total_segment_travel_time_seconds = min_car_travel_time.total_seconds() + wait_time * 60 + additional_travel_time * 60
                    total_travel_time_seconds += total_segment_travel_time_seconds
                    hours = int(total_segment_travel_time_seconds // 3600)
                    minutes = int((total_segment_travel_time_seconds % 3600) // 60)
                    seconds = int(total_segment_travel_time_seconds % 60)

                    total_distance_miles += car_travel_details['distance'].sum() * 0.000621371

                    distance_miles = car_travel_details['distance'].sum() * 0.000621371
                    duration_minutes = total_segment_travel_time_seconds / 60
                    shared_ride_fare = calculate_fare(base_fare=2.36, cost_per_mile=0.76, cost_per_minute=0.25, service_fee=3.58, distance=distance_miles, duration=duration_minutes)
                    total_cost += shared_ride_fare

                    all_segments_details.append(
                        html.Div([
                            html.Li(f"Segment {i + 1} Travel Time: {hours} hours, {minutes} minutes, and {seconds} seconds", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px'}),
                            html.Ul([
                                html.Li(f"Base Travel Time: {min_car_travel_time}", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px'}),
                                html.Li(f"Wait Time: {wait_time:.2f} minutes", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px'}),
                                html.Li(f"Additional Travel Time: {additional_travel_time:.2f} minutes", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px'})
                            ]),
                            html.Li(f"Shared Ride Fare: ${shared_ride_fare:.2f}", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px'})
                        ])
                    )

                    route_geometry = car_travel_details.loc[car_travel_details['travel_time'] == min_car_travel_time, 'geometry']
                    if not route_geometry.empty:
                        route_coords = list(route_geometry.values[0].coords)
                        route_lons = [coord[0] for coord in route_coords]
                        route_lats = [coord[1] for coord in route_coords]

                        route_trace = go.Scattermapbox(
                            mode='lines',
                            lon=route_lons,
                            lat=route_lats,
                            line=dict(width=2, color='blue'),
                            name=f'Route {i + 1}'
                        )
                        all_route_traces.append(route_trace)

                else:
                    if current_segment_mode == 'TRANSIT_WALK':
                        transport_modes = [r5py.TransportMode.TRANSIT]
                        access_modes = [r5py.TransportMode.WALK]
                        egress_modes = [r5py.TransportMode.WALK]
                    elif current_segment_mode == 'TRANSIT_BIKE':
                        transport_modes = [r5py.TransportMode.TRANSIT, r5py.TransportMode.BICYCLE]
                        access_modes = [r5py.TransportMode.BICYCLE]
                        egress_modes = [r5py.TransportMode.BICYCLE]
                    else:
                        transport_modes = [r5py.TransportMode[current_segment_mode]]
                        access_modes = []
                        egress_modes = []

                    detailed_itineraries_computer = r5py.DetailedItinerariesComputer(
                        transport_network,
                        origins=origins,
                        destinations=destinations,
                        departure=departure_datetime,
                        transport_modes=transport_modes,
                        access_modes=access_modes,
                        egress_modes=egress_modes
                    )
                    travel_details = detailed_itineraries_computer.compute_travel_details()

                    if current_segment_mode == 'TRANSIT_BIKE':
                        travel_details = travel_details[1:]

                    if current_segment_mode in ['TRANSIT_WALK', 'TRANSIT_BIKE']:
                        # Use the updated select_best_option function
                        min_option = select_best_option(travel_details, optimization_criteria)
                        selected_option_details = travel_details[travel_details['option'] == min_option['option']]

                        total_segment_travel_time_seconds = selected_option_details['total_time'].sum().total_seconds()
                        total_wait_time_seconds = selected_option_details['wait_time'].sum().total_seconds()
                        total_time_seconds = total_segment_travel_time_seconds + total_wait_time_seconds
                        total_distance_meters = selected_option_details['distance'].sum()
                        total_distance_miles += meters_to_miles(total_distance_meters)
                        total_travel_time_seconds += total_time_seconds

                        first_row = selected_option_details.iloc[0]
                        last_row = selected_option_details.iloc[-1]
                        total_walking_biking_distance_meters = first_row['distance'] + last_row['distance']
                        total_walking_biking_distance_miles = meters_to_miles(total_walking_biking_distance_meters)
                        total_walking_biking_time = first_row['travel_time'] + last_row['travel_time']
                        total_out_of_vehicle_time = total_walking_biking_time + selected_option_details['wait_time'].sum()

                        num_segments = len(selected_option_details)
                        transit_fare = TRANSIT_FARE_PER_RIDE * (num_segments - 2)
                        total_cost += transit_fare

                        first_segment = selected_option_details.iloc[0]
                        last_segment = selected_option_details.iloc[-1]
                        transit_segments = selected_option_details.iloc[1:-1]

                        first_segment_coords = list(first_segment['geometry'].coords)
                        last_segment_coords = list(last_segment['geometry'].coords)
                        transit_coords = [coord for segment in transit_segments['geometry'] for coord in segment.coords]

                        first_segment_lons = [coord[0] for coord in first_segment_coords]
                        first_segment_lats = [coord[1] for coord in first_segment_coords]
                        last_segment_lons = [coord[0] for coord in last_segment_coords]
                        last_segment_lats = [coord[1] for coord in last_segment_coords]
                        transit_lons = [coord[0] for coord in transit_coords]
                        transit_lats = [coord[1] for coord in transit_coords]

                        first_segment_trace = go.Scattermapbox(
                            mode='lines',
                            lon=first_segment_lons,
                            lat=first_segment_lats,
                            line=dict(width=2, color='lightblue'),
                            name=f'Segment {i + 1} Walking/Biking Segment'
                        )

                        transit_trace = go.Scattermapbox(
                            mode='lines',
                            lon=transit_lons,
                            lat=transit_lats,
                            line=dict(width=2, color='blue'),
                            name=f'Segment {i + 1} Transit Segment'
                        )

                        last_segment_trace = go.Scattermapbox(
                            mode='lines',
                            lon=last_segment_lons,
                            lat=last_segment_lats,
                            line=dict(width=2, color='lightblue'),
                            name=f'Segment {i + 1} Walking/Biking Segment'
                        )

                        all_route_traces.extend([first_segment_trace, transit_trace, last_segment_trace])

                        hours = int(total_time_seconds // 3600)
                        minutes = int((total_time_seconds % 3600) // 60)
                        seconds = int(total_time_seconds % 60)

                        output = [
                            html.Li(f"Calculated Travel Time: {hours} hours, {minutes} minutes, and {seconds} seconds", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                            html.Li(f"Total Walking/Biking Distance: {total_walking_biking_distance_miles:.2f} miles", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                            html.Li(f"Total Out of Vehicle Time: {total_out_of_vehicle_time}", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                            html.Li(f"Total Walking/Biking Time: {total_walking_biking_time}", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'})
                        ]

                        all_segments_details.append(
                            html.Li(f"Segment {i + 1} Travel Time: {hours} hours, {minutes} minutes, and {seconds} seconds", style={'padding': '10px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                        )
                        all_segments_details.extend(output)

                    else:
                        # For modes like CAR, BICYCLE, WALK (non-transit)
                        min_travel_time = travel_details['travel_time'].min()
                        total_distance_meters = travel_details['distance'].sum()
                        total_distance_miles += meters_to_miles(total_distance_meters)

                        route_geometry = travel_details.loc[travel_details['travel_time'] == min_travel_time, 'geometry']
                        if not route_geometry.empty:
                            route_coords = list(route_geometry.values[0].coords)
                            route_lons = [coord[0] for coord in route_coords]
                            route_lats = [coord[1] for coord in route_coords]

                            route_trace = go.Scattermapbox(
                                mode='lines',
                                lon=route_lons,
                                lat=route_lats,
                                line=dict(width=2, color='blue'),
                                name=f'Segment {i + 1} Route'
                            )
                            all_route_traces.append(route_trace)

                        total_segment_travel_time_seconds = min_travel_time.total_seconds()
                        total_travel_time_seconds += total_segment_travel_time_seconds
                        hours = int(total_segment_travel_time_seconds // 3600)
                        minutes = int((total_segment_travel_time_seconds % 3600) // 60)
                        seconds = int(total_segment_travel_time_seconds % 60)

                        all_segments_details.append(
                            html.Li(f"Segment {i + 1} Travel Time: {hours} hours, {minutes} minutes, and {seconds} seconds", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                        )

            total_hours = int(total_travel_time_seconds // 3600)
            total_minutes = int((total_travel_time_seconds % 3600) // 60)
            total_seconds = int(total_travel_time_seconds % 60)

            total_details = [
                html.Li(f"Total Travel Time: {total_hours} hours, {total_minutes} minutes, and {total_seconds} seconds", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                html.Li(f"Total Distance: {total_distance_miles:.2f} miles", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}),
                html.Li(f"Total Cost: ${total_cost:.2f}", style={'padding': '3px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'})
            ]

            all_details = all_segments_details + total_details

            current_figure = go.Figure(current_figure)
            current_figure.data = [trace for trace in current_figure.data if trace.mode != 'lines']

            for trace in all_route_traces:
                current_figure.add_trace(trace)

            origin_lon, origin_lat = parse_coordinates(origin)
            destination_lon, destination_lat = parse_coordinates(destination)
            all_lons = [origin_lon, destination_lon] + [parse_coordinates(dest)[1] for dest in dynamic_destinations if dest]
            all_lats = [origin_lat, destination_lat] + [parse_coordinates(dest)[0] for dest in dynamic_destinations if dest]
            all_texts = ["Origin", "Destination"] + [f"Destination {i + 1}" for i in range(len(dynamic_destinations)) if dynamic_destinations[i]]

            current_figure.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=all_lons,
                lat=all_lats,
                marker={'size': 10, 'color': ['red'] + ['#93979C'] * (len(all_lons) - 1)},
                text=all_texts,
                textposition="bottom right",
                showlegend=False
            ))

            selected_mode_key = {
                'TRANSIT_WALK': 'mode-transit-walk',
                'TRANSIT_BIKE': 'mode-transit-bike',
                'CAR': 'mode-car',
                'BICYCLE': 'mode-bike',
                'SHARED_RIDE': 'mode-shared-ride',
                'WALK': 'mode-walk'
            }.get(mode_of_travel)

            if trip_mode == 'different':
                selected_mode_summary = "Your modes of transport are: " + ", ".join(segment_modes[:len(coords_list) - 1])
                if selected_mode_key:
                    mode_styles[selected_mode_key]['background-color'] = '#1c293a'
                    mode_styles[selected_mode_key]['color'] = 'white'
            else:
                selected_mode_summary = ""
                if selected_mode_key:
                    mode_styles[selected_mode_key]['background-color'] = '#74bf0c'
                    mode_styles[selected_mode_key]['color'] = 'black'

            return origin, destination, current_figure, dynamic_destinations, html.Ul(all_details), mode_styles['mode-transit-walk'], mode_styles['mode-transit-bike'], mode_styles['mode-car'], mode_styles['mode-bike'], mode_styles['mode-shared-ride'], mode_styles['mode-walk'], selected_mode_summary

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return origin, destination, current_figure, dynamic_destinations, html.Li(f"Error calculating travel time: {str(e)}", style={'padding': '10px', 'background-color': '#f4f4f9', 'color': 'black', 'margin-bottom': '3px', 'text-align': 'left'}), mode_styles['mode-transit-walk'], mode_styles['mode-transit-bike'], mode_styles['mode-car'], mode_styles['mode-bike'], mode_styles['mode-shared-ride'], mode_styles['mode-walk'], ''

    if trigger in ['mode-transit-walk', 'mode-transit-bike', 'mode-car', 'mode-bike', 'mode-shared-ride', 'mode-walk']:
        origin_coords = parse_coordinates(origin)
        destination_coords = parse_coordinates(destination)
        if not origin_coords or not destination_coords:
            raise ValueError("Invalid coordinates")

        origin_lon, origin_lat = origin_coords
        destination_lon, destination_lat = destination_coords
        all_lons = [origin_lon, destination_lon] + [parse_coordinates(dest)[1] for dest in dynamic_destinations if dest]
        all_lats = [origin_lat, destination_lat] + [parse_coordinates(dest)[0] for dest in dynamic_destinations if dest]
        all_texts = ["Origin", "Destination"] + [f"Destination {i + 1}" for i in range(len(dynamic_destinations)) if dynamic_destinations[i]]

        current_figure = go.Figure(data=current_figure['data'], layout=current_figure['layout'])
        current_figure.data = [trace for trace in current_figure.data if trace.mode != 'lines']
        current_figure.add_trace(go.Scattermapbox(
            mode='markers+text',
            lon=all_lons,
            lat=all_lats,
            marker={'size': 10, 'color': ['red'] + ['#93979C'] * (len(all_lons) - 1)},
            text=all_texts,
            textposition="bottom right",
            showlegend=False
        ))

        selected_mode_key = {
            'TRANSIT_WALK': 'mode-transit-walk',
            'TRANSIT_BIKE': 'mode-transit-bike',
            'CAR': 'mode-car',
            'BICYCLE': 'mode-bike',
            'SHARED_RIDE': 'mode-shared-ride',
            'WALK': 'mode-walk'
        }.get(mode_of_travel)
        if selected_mode_key:
            mode_styles[selected_mode_key]['background-color'] = '#74bf0c'
            mode_styles[selected_mode_key]['color'] = 'black'

        return origin, destination, current_figure, dynamic_destinations, no_update, mode_styles['mode-transit-walk'], mode_styles['mode-transit-bike'], mode_styles['mode-car'], mode_styles['mode-bike'], mode_styles['mode-shared-ride'], mode_styles['mode-walk'], ''

    return origin, destination, current_figure, dynamic_destinations, no_update, no_update, no_update, no_update, no_update, no_update, no_update, ''

@app.callback(
    Output('departure-time-div', 'style'),
    [Input('departure-time-radio', 'value')]
)
def toggle_departure_time_div(radio_value):
    if radio_value == 'future':
        return {'display': 'block'}
    return {'display': 'none'}

# if __name__ == '__main__':
#     app.run_server(debug=True)
# Get the port from the environment (Render assigns one dynamically)
PORT = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT is not set

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

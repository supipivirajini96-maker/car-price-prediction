from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import numpy as np
from model_pipeline import predict_car_price, fill_missing_values

# Initialize Dash app
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    suppress_callback_exceptions=True
)
app.title = "Car Price Predictor"

# Layout with URL routing
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Main Page Layout
main_page = dbc.Container([
    html.H1("Welcome to Car Price Predictor", className="text-center my-5"),

    html.P(
        "This web application predicts the estimated price of a car based on a few key features. "
        "Simply provide the required inputs on the Prediction Page and get an instant estimate!",
        className="lead text-center mb-4"
    ),

    dbc.Card([
        dbc.CardHeader(html.H4("How Prediction Works")),
        dbc.CardBody([
            html.P(
                "The app uses a trained machine learning model to predict car prices. "
                "The model has learned from historical car data including features like transmission type and engine power."
            ),
            html.Ul([
                html.Li(html.B("Transmission Type:"), " Manual (0) or Automatic (1)."),
                html.Li(html.B("Max Power:"), " Maximum engine power in bhp. Higher power often increases the price."),
            ]),
            html.P(
                "After entering these details on the Prediction Page, "
                "the model outputs an estimated car price in dollars."
            ),
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(
            dbc.Button("Go to Prediction Page", href="/predict", color="primary", className="d-block mx-auto"),
            width=12
        )
    ])
], className="my-4")

# Prediction Page Layout
prediction_page = dbc.Container([
    html.H1("Car Price Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Transmission Type"),
            dcc.Dropdown(
                id="transmission",
                options=[
                    {"label": "Manual", "value": 0},
                    {"label": "Automatic", "value": 1}
                ],
                placeholder="Select Transmission Type",
                className="mb-3"
            ),

            html.Label("Max Power (bhp)"),
            dcc.Input(
                id="max_power", type="number",
                placeholder="Enter Max Power",
                className="form-control mb-3"
            ),

            dbc.Button("Predict", id="submit-btn", color="primary", className="mt-3"),
            html.Br(),
            html.Br(),
            dbc.Button("Try Again", id="try-again-btn", color="warning", className="me-2"),
            dbc.Button("Back to Home", href="/", color="secondary")
        ], md=6)
    ]),

    html.Hr(),
    html.Div(id="prediction-output", className="h4 text-success text-center mt-3")
])

# Routing Callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/predict":
        return prediction_page
    return main_page

# Prediction Callback
@app.callback(
    Output("transmission", "value"),
    Output("max_power", "value"),
    Output("prediction-output", "children"),
    Input("submit-btn", "n_clicks"),
    Input("try-again-btn", "n_clicks"),
    State("transmission", "value"),
    State("max_power", "value"),
    prevent_initial_call=True
)

# Handle Prediction or Reset
def handle_prediction_or_reset(submit_clicks, try_again_clicks, transmission, max_power):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "submit-btn":
        try:
            # Fill missing values from defaults
            transmission, max_power = fill_missing_values(transmission, max_power)

            # Prepare sample
            features = [transmission, max_power]
            sample = np.array([features])

            # do prediction
            prediction = predict_car_price(sample)
            return transmission, max_power, f"Predicted Car Price: ${prediction:,.2f}"

        except Exception as e:
            return transmission, max_power, f"Error: {str(e)}"

    elif button_id == "try-again-btn":
        # Reset inputs
        return None, None, ""

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)

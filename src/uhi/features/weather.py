"""Weather-station features from the NY Mesonet stations in the Bronx and Manhattan."""

import numpy as np
import pandas as pd

# Two points on the straight line that separates the Bronx from Manhattan in the study area
BRONX_LINE = ((-73.92, 40.78), (-73.94, 40.825))
WEATHER_COLUMNS = ["Air Temp at Surface [degC]", "Relative Humidity [percent]", "Avg Wind Speed [m/s]",
                   "Wind Direction [degrees]", "Solar Flux [W/m^2]"]


def is_bronx(longitude, latitude, line=BRONX_LINE):
    """1 for points north of the Bronx / Manhattan line, 0 otherwise."""
    (x1, y1), (x2, y2) = line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return np.where(latitude > m * longitude + b, 1, 0)


def read_station(weather_xls, sheet):
    """Readings of one station (sheet 'Bronx' or 'Manhattan') with a parsed 'Date / Time' column."""
    df = pd.read_excel(weather_xls, sheet_name=sheet) if not isinstance(weather_xls, pd.ExcelFile) \
        else weather_xls.parse(sheet)
    df["Date / Time"] = pd.to_datetime(df["Date / Time"])
    return df


def survey_hour_means(station_df):
    """Mean of every reading from 15:00 to 16:00 inclusive, the time window of the ground survey."""
    t = station_df["Date / Time"].dt
    in_window = (t.hour == 15) & (t.minute >= 0) | (t.hour == 16) & (t.minute == 0)
    return station_df[in_window].drop(columns=["Date / Time"]).mean()


def add_station_weather(df, bronx_means, manhattan_means):
    """Add the borough flag and give every pixel the survey-hour means of its station."""
    df["is_bronx"] = is_bronx(df["Longitude"], df["Latitude"])
    for col in manhattan_means.index:
        df[col] = np.where(df["is_bronx"] == 0, manhattan_means[col], bronx_means[col])
    return df


def add_interactions(df):
    """Weather interaction terms and two spectral interactions (in place)."""
    df["Wind_Solar_Impact"] = df["Avg Wind Speed [m/s]"] * df["Solar Flux [W/m^2]"]
    df["Temp_Humidity_Ratio"] = df["Air Temp at Surface [degC]"] / (df["Relative Humidity [percent]"] + 1e-6)

    # Wind direction in radians, and the U (east-west) / V (north-south) components
    df["Wind_Dir_Radians"] = np.radians(df["Wind Direction [degrees]"])
    df["Wind_U_Component"] = df["Avg Wind Speed [m/s]"] * np.sin(df["Wind_Dir_Radians"])
    df["Wind_V_Component"] = df["Avg Wind Speed [m/s]"] * np.cos(df["Wind_Dir_Radians"])

    # Vegetation x albedo, built-up x water
    df["Green_Heat_Reflection"] = df["NDVI_single"] * df["Albedo_single"]
    df["Urban_Water_Buffer"] = df["NDBI_single"] * df["NDWI_single"]

    # More interaction terms (Temp_Humidity_Ratio is recomputed without the epsilon)
    df["Temp_Humidity_Ratio"] = df["Air Temp at Surface [degC]"] / df["Relative Humidity [percent]"]
    df["Temp_Humidity_Diff"] = df["Air Temp at Surface [degC]"] - df["Relative Humidity [percent]"]
    df["Temp_Solar_Impact"] = df["Air Temp at Surface [degC]"] * df["Solar Flux [W/m^2]"]
    df["Wind_Temperature_Ratio"] = df["Avg Wind Speed [m/s]"] / df["Air Temp at Surface [degC]"]
    df["NDVI_NDBI_Ratio"] = df["NDVI_single"] / df["NDBI_single"]
    df["Wind_Direction_Speed_Impact"] = df["Avg Wind Speed [m/s]"] * df["Wind Direction [degrees]"]
    return df

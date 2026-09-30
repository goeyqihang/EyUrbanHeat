import numpy as np
import pandas as pd
import pytest

from uhi.features import weather


def test_is_bronx_splits_along_the_line():
    # the line runs through (-73.92, 40.78) and (-73.94, 40.825); at -73.93 it is at 40.8025
    lon = np.array([-73.90, -73.97, -73.93, -73.93])
    lat = np.array([40.86, 40.77, 40.803, 40.802])
    line_lat = 40.78 + (lon + 73.92) * (40.825 - 40.78) / (-73.94 + 73.92)
    expected = (lat > line_lat).astype(int)
    assert list(weather.is_bronx(lon, lat)) == list(expected) == [1, 0, 1, 0]


def test_survey_hour_means_uses_15_00_to_16_00_inclusive():
    times = pd.to_datetime(["2021-07-24 14:55", "2021-07-24 15:00", "2021-07-24 15:30",
                            "2021-07-24 16:00", "2021-07-24 16:05"])
    station = pd.DataFrame({"Date / Time": times, "Air Temp at Surface [degC]": [1.0, 2.0, 3.0, 4.0, 5.0]})
    means = weather.survey_hour_means(station)
    assert means["Air Temp at Surface [degC]"] == pytest.approx(3.0)
    assert "Date / Time" not in means.index


def test_read_station_parses_dates(tmp_path):
    path = tmp_path / "weather.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Date / Time": ["2021-07-24 15:00:00"], "Solar Flux [W/m^2]": [400]}).to_excel(
            writer, sheet_name="Bronx", index=False)
    df = weather.read_station(pd.ExcelFile(path), "Bronx")
    assert pd.api.types.is_datetime64_any_dtype(df["Date / Time"])


def test_add_station_weather_and_interactions():
    df = pd.DataFrame({"Longitude": [-73.90, -73.97], "Latitude": [40.86, 40.77],
                       "NDVI_single": [0.5, 0.2], "Albedo_single": [0.2, 0.1],
                       "NDBI_single": [-0.1, 0.3], "NDWI_single": [-0.4, -0.2]})
    bronx = pd.Series({"Air Temp at Surface [degC]": 30.0, "Relative Humidity [percent]": 40.0,
                       "Avg Wind Speed [m/s]": 2.0, "Wind Direction [degrees]": 90.0, "Solar Flux [W/m^2]": 500.0})
    manhattan = bronx * 0.5
    weather.add_station_weather(df, bronx, manhattan)
    assert list(df["is_bronx"]) == [1, 0]
    assert list(df["Air Temp at Surface [degC]"]) == [30.0, 15.0]

    weather.add_interactions(df)
    # recomputed without the epsilon
    assert df.loc[0, "Temp_Humidity_Ratio"] == 30.0 / 40.0
    assert df.loc[0, "Wind_U_Component"] == pytest.approx(2.0)
    assert df.loc[0, "Wind_V_Component"] == pytest.approx(0.0, abs=1e-12)
    assert df.loc[0, "Green_Heat_Reflection"] == pytest.approx(0.5 * 0.2)

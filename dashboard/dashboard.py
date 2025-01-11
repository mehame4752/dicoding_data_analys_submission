import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.ticker as ticker
import numpy as np

import os
from operator import itemgetter

st.set_page_config(
    page_title="Bike Rents Dashboard",
    page_icon=":bike:",
    initial_sidebar_state="expanded",
)


sns.set(style="dark")

dirname = os.path.dirname(__file__)

day_df = pd.read_csv(os.path.join(dirname, "assets/data/day.csv"))
hour_df = pd.read_csv(os.path.join(dirname, "assets/data/hour.csv"))

DATETIME_COLUMNS = ["dteday"]

for column in DATETIME_COLUMNS:
    day_df[column] = pd.to_datetime(day_df[column])
    hour_df[column] = pd.to_datetime(hour_df[column])

min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

# START CLEANING DATA

seasons_mapping = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

weather_mapping = {
    1: "Clear/Cloudy",
    2: "Foggy/Cloudy",
    3: "Light Rain/Light Snow",
    4: "Extreme Weather",
}

# Categorized temperature
# temp < 10         Cold
# 10 <= temp <= 20  Cool
# 20 < temp <= 30   Warm
# 30 < temp         Hot
temperature_mapping = {
    "Cold": 10,
    "Cool": 20,
    "Warm": 30,
    "Hot": np.Infinity,
}

# Categorized humidity
# hum < 30         : Very Dry
# 30 <= hum <= 50  : Dry
# 50 < hum <= 70   : Moderate
# 70 < hum <= 85   : Humid
# 85 < hum         : Very Humid
humidity_mapping = {
    "Very Dry": 30,
    "Dry": 50,
    "Moderate": 70,
    "Humid": 85,
    "Very Humid": np.Infinity,
}

# Categorized windspeed
# ws < 10           : Calm
# 10 <= ws <= 30    : Moderate
# 30 < ws <= 50     : Gusty
# 50 < ws           : Strong
windspeed_mapping = {
    "Calm": 10,
    "Moderate": 30,
    "Gusty": 50,
    "Strong": np.Infinity,
}

# Categorized hour
# 0 <= hour < 5    : Early Morning
# 5 <= hour < 11   : Morning
# 11 <= hour < 15  : Noon
# 15 <= hour < 18  : Afternoon
# 18 <= hour < 24  : Evening
hour_mapping = {
    "Early Morning": [0, 5],
    "Morning": [5, 11],
    "Noon": [11, 15],
    "Afternoon": [15, 18],
    "Evening": np.Infinity,
}

# based holiday column
day_type = {True: "Holiday", False: "Workday"}


def categorize_with_loop(categories=list(), values=list(), target_value=0):
    category = categories[0]

    index = 0
    for value in values:
        is_list = isinstance(value, list)

        if not is_list and target_value < value and index == 0:
            break

        category = categories[index]

        # when list we use range
        # otherwise use <= value
        is_done = (not is_list and target_value <= value) or (
            is_list and value[0] <= target_value and target_value < value[1]
        )

        if is_done:
            break

        index += 1

    return category


# function on documentation
#  normalized_temp = (original_temp - t_min)/(t_max - t_min)
#
# so origin temperature can getting by
#   origin_temp = (normalized_temp * (t_max - t_min)) + (t_min)
# where t_max = 39 and t_min =-8
def get_origin_temperature(normalized_temperature, tmax=39, tmin=-8):
    return (normalized_temperature * (tmax - (tmin))) + tmin


# function on documentation
#   normalized_humidity = origin_humidity / 100
#
# so origin humidity can getting by
#   origin humidity = normalized_humidity * 100
def get_origin_humidity(normalized_humidity, scala=100):
    return normalized_humidity * 100


# function on documentation
#   normalized_windspeed = origin_windspeed / 67
#
# so use derive function for getting origin_windspeed such as
#   origin_windspeed = normalized_windspeed * 67
def get_origin_windspeed(normalized_windspeed, scala=67):
    return normalized_windspeed * 67


# categorize temperature based on documentation
# where temp normalized with function
#   temp_normalized = (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39, t = original temperature
def categorize_temp(t, tmax=39, tmin=-8):
    original_temp = get_origin_temperature(t, tmax=tmax, tmin=tmin)

    temp_keys_map = list(temperature_mapping.keys())
    temp_vals_map = list(temperature_mapping.values())

    return categorize_with_loop(temp_keys_map, temp_vals_map, original_temp)


# categorize humidity based on documentation
# where humidity nomalized gained with function
#   humidity_normalized = humidity_origin / 100
def categorize_humid(humid, scala=100):
    original_humid = get_origin_humidity(humid, scala=scala)

    humid_keys = list(humidity_mapping.keys())
    humid_values = list(humidity_mapping.values())

    return categorize_with_loop(humid_keys, humid_values, original_humid)


# categorize windspeed based on documentation
# where the windspeed normalized gained using function
#   windspeed normalized = windspeed original / 67
def categorize_windspeed(windspeed, scala=67):
    original_windspeed = get_origin_windspeed(windspeed, scala=scala)

    windspeed_keys = list(windspeed_mapping.keys())
    windspeed_values = list(windspeed_mapping.values())

    return categorize_with_loop(windspeed_keys, windspeed_values, original_windspeed)


# categorize hour based time
def categorize_hour(hour):
    hour_keys = list(hour_mapping.keys())
    hour_values = list(hour_mapping.values())

    return categorize_with_loop(hour_keys, hour_values, hour)


def grouping_sum_user(dataframe, by, sort_values=True, ascending=False):
    result_df = dataframe.groupby(by).agg(
        {
            "cnt": "sum",
            "casual": "sum",
            "registered": "sum",
        }
    )

    if not sort_values:
        return result_df

    return result_df.sort_values(by="cnt", ascending=ascending)


def grouping_sum_user_v2(
    dataframe, by, show_columns="*", sort_values=False, ascending=False
):
    result_df = (
        dataframe.groupby(by)
        .agg(
            row_count=("cnt", "count"),
            rent_count=("cnt", "sum"),
            rent_mean=("cnt", "mean"),
            casual_count=("casual", "sum"),
            registered_count=("registered", "sum"),
        )
        .reset_index()
        .sort_values(by="rent_count", ascending=ascending)
    )

    if show_columns == "*" or not isinstance(show_columns, list):
        return result_df

    return result_df[show_columns]


# configs = [
#         {
#             'dataframe': 'test',
#             'by': 'by',
#             'is_ordinal': False,
#             'y_axis': 'test',
#             'x_axis': 'test',
#             'plot_type': 'barplot',
#         }
#     ]
def visualize_multiple_subplots(
    nrows,
    ncols,
    subplots,
    separate=False,
    suptitle="",
):
    DEFAULT_SUBPLOT = {
        "y_axis": "cnt",
        "plot_type": "barplot",
        "is_ordinal": False,
        "horizontal": False,
        "grouping_func": grouping_sum_user,
        "template": "{0} {1}",
        "force_yaxis_to_category": False,
        "only_head": False,
        "invert_xaxis": False,
        "right_yaxis": False,
        "ascending": False,
        "mapping": None,
        "reindex": None,
        "title": None,
    }

    width_size = ncols * 8
    height_size = nrows * 7

    figs = []

    if not separate:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(width_size, height_size)
        )

    current_index = 0
    for row_index in range(nrows):
        for col_index in range(ncols):
            subplot = subplots[current_index]
            if separate:
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

            target_axe = axes

            if not separate and nrows == 1 and ncols > 1:
                target_axe = axes[col_index]

            elif not separate and nrows > 1 and ncols > 1:
                target_axe = axes[row_index, col_index]

            # remove axis
            if subplot == None:
                target_axe.set_visible(False)
                continue

            subplot = {**DEFAULT_SUBPLOT, **subplot}

            (
                dataframe,
                by,
                is_ordinal,
                y_axis,
                x_axis,
                plot_type,
                title,
                horizontal,
                grouping_func,
                template,
                only_head,
                invert_xaxis,
                right_yaxis,
                ascending,
                reindex,
            ) = itemgetter(
                "dataframe",
                "by",
                "is_ordinal",
                "y_axis",
                "x_axis",
                "plot_type",
                "title",
                "horizontal",
                "grouping_func",
                "template",
                "only_head",
                "invert_xaxis",
                "right_yaxis",
                "ascending",
                "reindex",
            )(
                subplot
            )

            target_df = grouping_func(
                dataframe, by, sort_values=is_ordinal, ascending=ascending
            )

            if only_head:
                target_df = target_df.head()

            if isinstance(reindex, list) and len(reindex):
                target_df = target_df.reindex(reindex)

            target_df = target_df.reset_index()

            value_axis = x_axis if horizontal else y_axis
            category_axis = y_axis if horizontal else x_axis

            max_index = target_df[value_axis].idxmax()
            max_value = target_df[y_axis][max_index]

            min_index = target_df[value_axis].idxmin()
            min_value = target_df[y_axis][min_index]

            target_value = min_value if ascending else max_value

            colors = [
                "#d3d3d3" if target_df[y_axis][i] != target_value else "#ff6347"
                for i in target_df.index
            ]

            if plot_type == "barplot":
                sns.barplot(
                    y=y_axis,
                    x=x_axis,
                    ax=target_axe,
                    data=target_df,
                    palette=colors,
                    # hue=y_axis if horizontal else x_axis,
                    orient="y" if horizontal else "x",
                    order=target_df[category_axis],
                )

            elif plot_type == "lineplot":
                sns.lineplot(
                    data=target_df,
                    ax=target_axe,
                    y=y_axis,
                    x=x_axis,
                    marker="o",
                    linestyle="-",
                    markersize=6,
                    linewidth=1,
                    color="#72BCD4",
                )

                plt.xticks(target_df[category_axis])

                # target_axe.grid(True, axis='x', linestyle='--', alpha=0.6)

                plt.text(
                    max_index,
                    max_value,
                    template.format(max_index, max_value),
                    fontsize=10,
                    ha="left",
                    va="top",
                    color="#ff6347",
                )

                plt.scatter(
                    max_index,
                    max_value,
                    color="#ff6347",
                    s=30,
                    zorder=5,
                    label="Max value",
                )

            if horizontal:
                target_axe.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
                )
            else:
                target_axe.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
                )

            target_axe.set(xlabel=None, ylabel=None, title=title)

            if invert_xaxis:
                target_axe.invert_xaxis()

            if right_yaxis:
                target_axe.yaxis.set_label_position("right")
                target_axe.yaxis.tick_right()

            figs.append(fig)

            current_index += 1

    if suptitle:
        plt.suptitle(suptitle)

    return figs


def visualize_multiple_plots(nrows, plots, suptitle=""):
    return visualize_multiple_subplots(
        nrows=nrows, ncols=len(plots), subplots=plots, separate=True, suptitle=suptitle
    )


# END CLEANING DATA

# START PREPARING DATA

hour_df["season_category"] = hour_df["season"].map(seasons_mapping)
hour_df["weather_category"] = hour_df["weathersit"].map(weather_mapping)
hour_df["temp_category"] = hour_df["temp"].apply(categorize_temp)
hour_df["humid_category"] = hour_df["hum"].apply(categorize_humid)
hour_df["windspeed_category"] = hour_df["windspeed"].apply(categorize_windspeed)

hour_df["hour_category"] = hour_df["hr"].sort_values().apply(categorize_hour)

day_df["season_category"] = day_df["season"].map(seasons_mapping)

# END PREPARING DATA


# START SIDEBAR

with st.sidebar:
    st.title("Bike Rent")
    st.image(os.path.join(dirname, "assets/images/bicycle-icon.png"))

    # date input
    date_input = st.date_input(
        label="Time Range",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
    )

    start_date = min_date
    end_date = max_date

    if len(date_input) == 2:
        start_date, end_date = date_input

main_day_df = day_df[
    (day_df["dteday"] >= str(start_date)) & (day_df["dteday"] <= str(end_date))
]

main_hour_df = hour_df[
    (hour_df["dteday"] >= str(start_date)) & (hour_df["dteday"] <= str(end_date))
]

# END SIDEBAR


# dashboard

st.header("Bike Rents Dashboard :bike:")


# START COUNTS

st.subheader("Rent Amounts")

column1, column2, column3 = st.columns(3)

with column1:
    total_registered_rents = main_day_df.registered.sum()
    st.metric("Total registered renters", value="{:,}".format(total_registered_rents))

with column2:
    total_casual_rents = main_day_df.casual.sum()
    st.metric("Total new renters", value="{:,}".format(total_casual_rents))

with column3:
    total_rents = main_day_df.cnt.sum()
    st.metric("Total rents", value="{:,}".format(total_rents))

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    main_day_df["dteday"], main_day_df["cnt"], marker="o", linewidth=1, color="#90CAF9"
)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

st.pyplot(fig)

# END COUNTS

# START ENVIROMENTAL FACTORS

st.subheader("Renter Distributions based on Environmental Factors")

fig1, fig2, fig3, fig4, fig5 = visualize_multiple_plots(
    nrows=1,
    plots=[
        {
            "dataframe": main_day_df,
            "by": "season_category",
            "x_axis": "season_category",
            "reindex": list(seasons_mapping.values()),
        },
        {
            "dataframe": main_hour_df,
            "by": "weather_category",
            "x_axis": "weather_category",
            "reindex": list(weather_mapping.values()),
        },
        {
            "dataframe": main_hour_df,
            "by": "temp_category",
            "x_axis": "temp_category",
            "reindex": list(temperature_mapping.keys()),
        },
        {
            "dataframe": main_hour_df,
            "by": "humid_category",
            "x_axis": "humid_category",
            "reindex": list(humidity_mapping.keys()),
        },
        {
            "dataframe": main_hour_df,
            "by": "windspeed_category",
            "x_axis": "windspeed_category",
            "reindex": list(windspeed_mapping.keys()),
        },
    ],
)

st.text("By Season")
st.pyplot(fig1)

st.text("By Weather")
st.pyplot(fig2)

st.text("By Temperature")
st.pyplot(fig3)

st.text("By Humidity")
st.pyplot(fig4)

st.text("By Windspeed")
st.pyplot(fig5)

# END ENVIROMENTAL FACTORS

# START HOURS

st.subheader("Renter Distributions based on Hours")

st.text("Trends of Rental Bikes by hour")
(fig1,) = visualize_multiple_plots(
    nrows=1,
    plots=[
        {
            "dataframe": main_hour_df,
            "by": "hr",
            "x_axis": "hr",
            "title": "Total Bike Rents per Hour (2011 - 2012)",
            "plot_type": "lineplot",
            "template": "   Hour {0}:00\n   Max: {1:,}",
        },
    ],
)
st.pyplot(fig1)

st.text("By hour category")
(fig1,) = visualize_multiple_plots(
    nrows=1,
    plots=[
        {
            "dataframe": main_hour_df,
            "by": "hour_category",
            "x_axis": "hour_category",
            "reindex": list(hour_mapping.keys()),
        },
    ],
)
st.pyplot(fig1)

st.text("Most and Least Bike Rents by Hour")
fig1, fig2 = visualize_multiple_subplots(
    nrows=1,
    ncols=2,
    subplots=[
        {
            "dataframe": main_hour_df,
            "by": "hr",
            "x_axis": "cnt",
            "y_axis": "hr",
            "title": "Most Productive Hour",
            "horizontal": True,
            "is_ordinal": True,
            "only_head": True,
        },
        {
            "dataframe": main_hour_df,
            "by": "hr",
            "x_axis": "cnt",
            "y_axis": "hr",
            "title": "Least Productive Hour",
            "horizontal": True,
            "is_ordinal": True,
            "only_head": True,
            "invert_xaxis": True,
            "right_yaxis": True,
            "ascending": True,
        },
    ],
    suptitle="Most and Least Bike Rents by Hour",
)

st.pyplot(fig1)

# END HOURS

st.caption("Copyright © mehame 2025")

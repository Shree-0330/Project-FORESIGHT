# ============================================================
# PROJECT FORESIGHT
# AI-POWERED DEMAND FORECASTING & INVENTORY INTELLIGENCE
# ============================================================

import os
import pickle

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
PREDICTION_FILE = "foresight_predictions.csv"
INVENTORY_FILE = "foresight_inventory_analysis.csv"
MODEL_FILE = "foresight_demand_model.pkl"

MODEL_LOADED_TEXT = "Loaded ✅"
MODEL_ERROR_TEXT = "Error ❌"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0f1117;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Headings */
    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.7rem !important;
    }

    h3 {
        font-size: 1.3rem !important;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background-color: #181c25;
        border: 1px solid #303746;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aeb7c6 !important;
    }

    div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171a21;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
    }

    /* Download button */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Info box */
    .project-box {
        background-color: #182235;
        border-left: 4px solid #4da3ff;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    return train, test, predictions, inventory


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None, "Model file not found."

    # First try joblib
    try:
        model = joblib.load(MODEL_FILE)
        return model, None

    except Exception as joblib_error:

        # Then try pickle
        try:
            with open(MODEL_FILE, "rb") as file:
                model = pickle.load(file)

            return model, None

        except Exception as pickle_error:

            return (
                None,
                f"Model could not be loaded. "
                f"Joblib: {joblib_error} | "
                f"Pickle: {pickle_error}"
            )


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    PREDICTION_FILE,
    INVENTORY_FILE,
    MODEL_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Some required project files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD PROJECT DATA
# ============================================================

try:

    train, test, predictions, inventory = load_data()

except Exception as error:

    st.error("❌ Could not load project datasets.")

    st.exception(error)

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model, model_error = load_model()


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

train_display = train.copy()

if "date" in train_display.columns:

    train_display["date"] = pd.to_datetime(
        train_display["date"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔮 FORESIGHT")

    st.caption(
        "AI-Powered Demand Forecasting & Inventory Intelligence"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Demand Predictions",
            "Demand Analysis",
            "Inventory Intelligence",
            "Data Explorer",
            "About Project"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader("🤖 Model Status")

    if model is not None:

        st.success(MODEL_LOADED_TEXT)

    else:

        st.error(MODEL_ERROR_TEXT)

    st.markdown("---")

    st.caption(
        "Project FORESIGHT uses machine learning "
        "for demand forecasting and business intelligence."
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 Business Dashboard")

    st.markdown(
        """
        <div class="project-box">
        <b>Project FORESIGHT</b> uses historical demand data,
        machine learning predictions and analytical insights
        to support demand forecasting and inventory planning.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Demand Filters")

    filtered_data = train_display.copy()

    # --------------------------------------------------------
    # STORE SELECTOR
    # --------------------------------------------------------

    if "store" in filtered_data.columns:

        store_values = sorted(
            filtered_data["store"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_store = st.sidebar.selectbox(
            "🏪 Select Store",
            ["All Stores"] + store_values
        )

        if selected_store != "All Stores":

            filtered_data = filtered_data[
                filtered_data["store"] == selected_store
            ]

    # --------------------------------------------------------
    # ITEM SELECTOR
    # --------------------------------------------------------

    if "item" in filtered_data.columns:

        item_values = sorted(
            filtered_data["item"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_item = st.sidebar.selectbox(
            "📦 Select Item",
            ["All Items"] + item_values
        )

        if selected_item != "All Items":

            filtered_data = filtered_data[
                filtered_data["item"] == selected_item
            ]

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    if "date" in filtered_data.columns:

        valid_dates = filtered_data["date"].dropna()

        if not valid_dates.empty:

            minimum_date = valid_dates.min().date()
            maximum_date = valid_dates.max().date()

            selected_dates = st.sidebar.date_input(
                "📅 Select Date Range",
                value=(minimum_date, maximum_date),
                min_value=minimum_date,
                max_value=maximum_date
            )

            if isinstance(selected_dates, tuple):

                if len(selected_dates) == 2:

                    start_date = pd.Timestamp(
                        selected_dates[0]
                    )

                    end_date = pd.Timestamp(
                        selected_dates[1]
                    )

                    filtered_data = filtered_data[
                        (filtered_data["date"] >= start_date)
                        &
                        (filtered_data["date"] <= end_date)
                    ]

    # --------------------------------------------------------
    # SALES RANGE SLIDER
    # --------------------------------------------------------

    if (
        "sales" in filtered_data.columns
        and not filtered_data.empty
    ):

        sales_min = int(
            filtered_data["sales"].min()
        )

        sales_max = int(
            filtered_data["sales"].max()
        )

        if sales_min < sales_max:

            selected_sales = st.sidebar.slider(
                "📊 Sales / Demand Range",
                min_value=sales_min,
                max_value=sales_max,
                value=(sales_min, sales_max)
            )

            filtered_data = filtered_data[
                (filtered_data["sales"] >= selected_sales[0])
                &
                (filtered_data["sales"] <= selected_sales[1])
            ]

    # ========================================================
    # MAIN KPIs
    # ========================================================

    st.subheader("📌 Key Performance Indicators")

    total_training_records = len(train)

    total_test_records = len(test)

    total_forecast_records = len(predictions)

    total_stores = (
        train["store"].nunique()
        if "store" in train.columns
        else 0
    )

    total_items = (
        train["item"].nunique()
        if "item" in train.columns
        else 0
    )

    total_sales = (
        train["sales"].sum()
        if "sales" in train.columns
        else 0
    )

    average_sales = (
        train["sales"].mean()
        if "sales" in train.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📚 Training Records",
            f"{total_training_records:,}"
        )

    with col2:

        st.metric(
            "🧪 Test Records",
            f"{total_test_records:,}"
        )

    with col3:

        st.metric(
            "🔮 Forecast Records",
            f"{total_forecast_records:,}"
        )

    with col4:

        st.metric(
            "🤖 Model Status",
            MODEL_LOADED_TEXT
            if model is not None
            else MODEL_ERROR_TEXT
        )

    # ========================================================
    # SECOND KPI ROW
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🏪 Stores",
            f"{total_stores:,}"
        )

    with col2:

        st.metric(
            "📦 Items",
            f"{total_items:,}"
        )

    with col3:

        st.metric(
            "💰 Total Sales",
            f"{total_sales:,.0f}"
        )

    with col4:

        st.metric(
            "📊 Average Sales",
            f"{average_sales:,.2f}"
        )

    st.markdown("---")

    # ========================================================
    # FILTERED RESULTS
    # ========================================================

    st.subheader("🎯 Filtered Demand Overview")

    filtered_records = len(filtered_data)

    if (
        "sales" in filtered_data.columns
        and not filtered_data.empty
    ):

        filtered_total_sales = filtered_data["sales"].sum()

        filtered_average_sales = filtered_data["sales"].mean()

        filtered_max_sales = filtered_data["sales"].max()

    else:

        filtered_total_sales = 0
        filtered_average_sales = 0
        filtered_max_sales = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Filtered Records",
            f"{filtered_records:,}"
        )

    with col2:

        st.metric(
            "Filtered Demand",
            f"{filtered_total_sales:,.0f}"
        )

    with col3:

        st.metric(
            "Average Demand",
            f"{filtered_average_sales:,.2f}"
        )

    with col4:

        st.metric(
            "Highest Demand",
            f"{filtered_max_sales:,.0f}"
        )

    st.markdown("---")

    # ========================================================
    # INTERACTIVE DEMAND TREND
    # ========================================================

    if (
        "date" in filtered_data.columns
        and "sales" in filtered_data.columns
        and not filtered_data.empty
    ):

        daily_sales = (
            filtered_data
            .groupby("date")["sales"]
            .sum()
            .reset_index()
        )

        st.subheader("📈 Demand Trend")

        fig = px.line(
            daily_sales,
            x="date",
            y="sales",
            title="Demand Over Time",
            markers=False
        )

        fig.update_traces(
            hovertemplate=
            "Date: %{x}<br>"
            "Demand: %{y:,.0f}"
            "<extra></extra>"
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No data available for the selected filters."
        )

    # ========================================================
    # STORE AND ITEM ANALYSIS
    # ========================================================

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # STORE SALES
    # --------------------------------------------------------

    with col1:

        if (
            "store" in filtered_data.columns
            and "sales" in filtered_data.columns
            and not filtered_data.empty
        ):

            store_sales = (
                filtered_data
                .groupby("store")["sales"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            st.subheader("🏪 Sales by Store")

            store_chart = px.bar(
                store_sales,
                x="store",
                y="sales",
                title="Store-wise Demand"
            )

            store_chart.update_traces(
                hovertemplate=
                "Store: %{x}<br>"
                "Demand: %{y:,.0f}"
                "<extra></extra>"
            )

            st.plotly_chart(
                store_chart,
                use_container_width=True
            )

    # --------------------------------------------------------
    # ITEM SALES
    # --------------------------------------------------------

    with col2:

        if (
            "item" in filtered_data.columns
            and "sales" in filtered_data.columns
            and not filtered_data.empty
        ):

            item_sales = (
                filtered_data
                .groupby("item")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            st.subheader("📦 Top Items")

            item_chart = px.bar(
                item_sales,
                x="item",
                y="sales",
                title="Top 10 Items by Demand"
            )

            item_chart.update_traces(
                hovertemplate=
                "Item: %{x}<br>"
                "Demand: %{y:,.0f}"
                "<extra></extra>"
            )

            st.plotly_chart(
                item_chart,
                use_container_width=True
            )

    # ========================================================
    # FILTERED DATA PREVIEW
    # ========================================================

    st.markdown("---")

    st.subheader("📋 Filtered Data Preview")

    st.dataframe(
        filtered_data.head(100),
        use_container_width=True
    )


# ============================================================
# DEMAND PREDICTIONS
# ============================================================

elif page == "Demand Predictions":

    st.title("📈 Demand Predictions")

    st.write(
        "Machine-learning forecast results generated by "
        "Project FORESIGHT."
    )

    st.markdown("---")

    # ========================================================
    # FIND NUMERIC COLUMNS
    # ========================================================

    numeric_prediction_columns = (
        predictions
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    # ========================================================
    # KPIs
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Forecast Records",
            f"{len(predictions):,}"
        )

    with col2:

        st.metric(
            "Prediction Columns",
            f"{len(predictions.columns):,}"
        )

    with col3:

        if numeric_prediction_columns:

            prediction_column = (
                numeric_prediction_columns[-1]
            )

            st.metric(
                "Average Prediction",
                f"{predictions[prediction_column].mean():,.2f}"
            )

        else:

            st.metric(
                "Average Prediction",
                "N/A"
            )

    with col4:

        st.metric(
            "Model",
            MODEL_LOADED_TEXT
            if model is not None
            else MODEL_ERROR_TEXT
        )

    st.markdown("---")

    # ========================================================
    # PREDICTION TABLE
    # ========================================================

    st.subheader("📋 Forecast Results")

    st.dataframe(
        predictions,
        use_container_width=True
    )

    # ========================================================
    # PREDICTION VISUALIZATION
    # ========================================================

    if numeric_prediction_columns:

        selected_prediction_column = st.selectbox(
            "Select Prediction Column",
            numeric_prediction_columns
        )

        prediction_chart = px.line(
            predictions,
            y=selected_prediction_column,
            title=f"{selected_prediction_column} Forecast"
        )

        prediction_chart.update_traces(
            hovertemplate=
            "Record: %{x}<br>"
            "Value: %{y:,.2f}"
            "<extra></extra>"
        )

        prediction_chart.update_layout(
            xaxis_title="Record",
            yaxis_title=selected_prediction_column
        )

        st.plotly_chart(
            prediction_chart,
            use_container_width=True
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        label="⬇️ Download Forecast Results",
        data=predictions.to_csv(index=False),
        file_name="foresight_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# DEMAND ANALYSIS
# ============================================================

elif page == "Demand Analysis":

    st.title("📊 Demand Analysis")

    st.write(
        "Analyze historical sales and demand patterns."
    )

    st.markdown("---")

    # ========================================================
    # SALES KPIs
    # ========================================================

    if "sales" in train.columns:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Sales",
                f"{train['sales'].sum():,.0f}"
            )

        with col2:

            st.metric(
                "Average Sales",
                f"{train['sales'].mean():,.2f}"
            )

        with col3:

            st.metric(
                "Maximum Sales",
                f"{train['sales'].max():,.0f}"
            )

        with col4:

            st.metric(
                "Minimum Sales",
                f"{train['sales'].min():,.0f}"
            )

    st.markdown("---")

    # ========================================================
    # DAILY DEMAND
    # ========================================================

    if (
        "date" in train_display.columns
        and "sales" in train_display.columns
    ):

        daily_sales = (
            train_display
            .groupby("date")["sales"]
            .sum()
            .reset_index()
        )

        st.subheader("📅 Historical Demand Trend")

        fig = px.line(
            daily_sales,
            x="date",
            y="sales",
            title="Historical Daily Demand"
        )

        fig.update_traces(
            hovertemplate=
            "Date: %{x}<br>"
            "Sales: %{y:,.0f}"
            "<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # STORE ANALYSIS
    # ========================================================

    if (
        "store" in train.columns
        and "sales" in train.columns
    ):

        store_sales = (
            train
            .groupby("store")["sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.subheader("🏪 Store Demand Analysis")

        fig = px.bar(
            store_sales,
            x="store",
            y="sales",
            title="Total Sales by Store"
        )

        fig.update_traces(
            hovertemplate=
            "Store: %{x}<br>"
            "Sales: %{y:,.0f}"
            "<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # ITEM ANALYSIS
    # ========================================================

    if (
        "item" in train.columns
        and "sales" in train.columns
    ):

        item_sales = (
            train
            .groupby("item")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        st.subheader("📦 Top 10 Items")

        fig = px.bar(
            item_sales,
            x="item",
            y="sales",
            title="Top 10 Items by Sales"
        )

        fig.update_traces(
            hovertemplate=
            "Item: %{x}<br>"
            "Sales: %{y:,.0f}"
            "<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # DAY-OF-WEEK ANALYSIS
    # ========================================================

    if (
        "date" in train_display.columns
        and "sales" in train_display.columns
    ):

        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_data = train_display.copy()

        day_data["day_name"] = (
            day_data["date"].dt.day_name()
        )

        day_sales = (
            day_data
            .groupby("day_name")["sales"]
            .mean()
            .reindex(day_order)
            .reset_index()
        )

        st.subheader("📅 Average Demand by Day")

        fig = px.bar(
            day_sales,
            x="day_name",
            y="sales",
            title="Average Sales by Day"
        )

        fig.update_traces(
            hovertemplate=
            "Day: %{x}<br>"
            "Average Demand: %{y:,.2f}"
            "<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# INVENTORY INTELLIGENCE
# ============================================================

elif page == "Inventory Intelligence":

    st.title("📦 Inventory Intelligence")

    st.write(
        "Explore the inventory analysis generated by "
        "Project FORESIGHT."
    )

    st.markdown("---")

    # ========================================================
    # INVENTORY KPIs
    # ========================================================

    numeric_inventory_columns = (
        inventory
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Inventory Records",
            f"{len(inventory):,}"
        )

    with col2:

        st.metric(
            "Inventory Columns",
            f"{len(inventory.columns):,}"
        )

    with col3:

        st.metric(
            "Numeric Metrics",
            f"{len(numeric_inventory_columns):,}"
        )

    with col4:

        st.metric(
            "Model Status",
            MODEL_LOADED_TEXT
            if model is not None
            else MODEL_ERROR_TEXT
        )

    st.markdown("---")

    # ========================================================
    # INVENTORY DATA
    # ========================================================

    st.subheader("📋 Inventory Analysis Data")

    st.dataframe(
        inventory,
        use_container_width=True
    )

    # ========================================================
    # INVENTORY METRIC SELECTOR
    # ========================================================

    if numeric_inventory_columns:

        selected_inventory_metric = st.selectbox(
            "Select Inventory Metric",
            numeric_inventory_columns
        )

        fig = px.line(
            inventory,
            y=selected_inventory_metric,
            title=f"{selected_inventory_metric} Analysis"
        )

        fig.update_traces(
            hovertemplate=
            "Record: %{x}<br>"
            "Value: %{y:,.2f}"
            "<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric inventory metrics were found."
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        label="⬇️ Download Inventory Analysis",
        data=inventory.to_csv(index=False),
        file_name="foresight_inventory_analysis.csv",
        mime="text/csv"
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("🔎 Data Explorer")

    st.write(
        "Explore the datasets used in Project FORESIGHT."
    )

    st.markdown("---")

    # ========================================================
    # DATASET SELECTOR
    # ========================================================

    dataset_name = st.selectbox(
        "Select Dataset",
        [
            "Training Data",
            "Test Data",
            "Predictions",
            "Inventory Analysis"
        ]
    )

    if dataset_name == "Training Data":

        selected_data = train

    elif dataset_name == "Test Data":

        selected_data = test

    elif dataset_name == "Predictions":

        selected_data = predictions

    else:

        selected_data = inventory

    # ========================================================
    # DATASET KPIs
    # ========================================================

    missing_values = (
        selected_data
        .isnull()
        .sum()
        .sum()
    )

    duplicate_rows = (
        selected_data
        .duplicated()
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(selected_data):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(selected_data.columns):,}"
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}"
        )

    st.markdown("---")

    # ========================================================
    # DATA TABLE
    # ========================================================

    st.subheader(
        f"📋 {dataset_name}"
    )

    st.dataframe(
        selected_data,
        use_container_width=True
    )

    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    st.subheader("🔍 Column Information")

    column_information = pd.DataFrame(
        {
            "Column": selected_data.columns,
            "Data Type": [
                str(dtype)
                for dtype in selected_data.dtypes
            ],
            "Missing Values": [
                selected_data[column].isnull().sum()
                for column in selected_data.columns
            ],
            "Unique Values": [
                selected_data[column].nunique()
                for column in selected_data.columns
            ]
        }
    )

    st.dataframe(
        column_information,
        use_container_width=True
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("ℹ️ About Project FORESIGHT")

    st.markdown("---")

    st.subheader(
        "🔮 AI-Powered Demand Forecasting & Inventory Intelligence"
    )

    st.write(
        """
        Project FORESIGHT is a machine-learning based demand
        forecasting and analytics platform.

        The system uses historical demand data to generate
        forecasts and provide analytical insights that can
        support demand planning and inventory-related decisions.
        """
    )

    # ========================================================
    # BUSINESS OBJECTIVE
    # ========================================================

    st.subheader("🎯 Business Objective")

    st.write(
        """
        The objective is to understand historical demand,
        forecast future demand and provide useful information
        for better business planning.
        """
    )

    # ========================================================
    # TECHNOLOGIES
    # ========================================================

    st.subheader("🛠️ Technologies")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.info(
            "🐍 Python\n\n"
            "Data Processing"
        )

    with col2:

        st.info(
            "🤖 Machine Learning\n\n"
            "Demand Forecasting"
        )

    with col3:

        st.info(
            "📊 Pandas\n\n"
            "Data Analysis"
        )

    with col4:

        st.info(
            "🌐 Streamlit\n\n"
            "Dashboard"
        )

    # ========================================================
    # PROJECT WORKFLOW
    # ========================================================

    st.subheader("🔄 Project Workflow")

    st.markdown(
        """
        **Historical Data**

        ↓

        **Data Preprocessing**

        ↓

        **Exploratory Data Analysis**

        ↓

        **Feature Engineering**

        ↓

        **Machine Learning Model**

        ↓

        **Model Evaluation**

        ↓

        **Demand Prediction**

        ↓

        **Inventory Analysis**

        ↓

        **Streamlit Dashboard**
        """
    )

    # ========================================================
    # PROJECT FILES
    # ========================================================

    st.subheader("📁 Project Files")

    project_files = [
        TRAIN_FILE,
        TEST_FILE,
        PREDICTION_FILE,
        INVENTORY_FILE,
        MODEL_FILE,
        "APP.py"
    ]

    for file in project_files:

        if os.path.exists(file):

            st.success(
                f"✓ {file}"
            )

        else:

            st.warning(
                f"⚠️ {file} not found"
            )

    st.markdown("---")

    st.success(
        "🚀 Project FORESIGHT is running successfully."
    )
    
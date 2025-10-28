# -*- coding: utf-8 -*-
"""
Combined and Enhanced Flight Fare Predictor App
-------------------------------------------------
This Streamlit application predicts flight fares based on user input,
incorporating robust data preprocessing, model training pipelines,
target variable standardization, and insightful visualizations.

**Modern UI & Text Updates:**
- UI styling modernized with a new CSS block.
- All user-facing text "Price" has been updated to "Fare".
- Core functionality remains unchanged.

Structure:
1.  Import Libraries
2.  Page Configuration
3.  Custom Modern CSS (Updated for attractiveness)
4.  Data Loading & Target Standardization
5.  Feature Definition
6.  Sidebar Controls
7.  Helper Function (Prediction Summary)
8.  Cached Model Training Functions
9.  Main Application Logic (Pages - All Fixes Included)
10. Footer (on Data Overview)
"""

# --- 1. Import Necessary Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore') # Suppress minor warnings

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Flight Fare Predictor", # <-- Updated
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Custom Modern CSS ---
st.markdown("""
<style>
    /* Base & Font */
    body {
        font-family: 'Inter', sans-serif;
        color: #333333;
    }

    /* Main Container */
    .main .block-container {
        padding: 1.5rem 2.5rem 2rem 2.5rem;
        background-color: #F8F9FA; /* Light grey background */
    }

    /* Sidebar */
    .stSidebar {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #0D6EFD; /* Modern Blue */
    }
    /* Sidebar Radio Button Items */
    .stSidebar .stRadio > label {
        padding: 0.6rem 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease-in-out;
    }
    /* Sidebar Radio Hover (Unselected) */
    .stSidebar .stRadio > label:hover {
        background-color: #E9ECEF;
        color: #0D6EFD;
        transform: translateX(2px);
    }
    /* Sidebar Radio Selected */
    .stSidebar .stRadio [aria-selected="true"] {
        background-color: #0D6EFD;
        color: #FFFFFF;
        font-weight: 500;
    }
    .stSidebar .stRadio [aria-selected="true"]:hover {
        background-color: #0A58CA;
        color: #FFFFFF;
        transform: none; /* No hover transform when selected */
    }
    .stSidebar .stMarkdown {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* Page Titles */
    h1, h2, h3 {
        color: #1A3A5A; /* Deep Navy Blue */
        font-weight: 600;
    }

    /* Page Section Containers (Cards) & Form Containers */
    .page-section, .stForm {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 0.75rem;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        /* Softer, more modern shadow */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    .page-section:hover, .stForm:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.07);
    }


    /* Metric Cards */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 0.75rem;
        padding: 1.25rem 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease-in-out;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .stMetric > label {
        color: #495057; /* Medium Grey Text */
        font-weight: 500;
    }
    .stMetric > div {
        color: #0D6EFD; /* Modern Blue */
        font-weight: 600;
        font-size: 2rem; /* Make metric value larger */
    }

    /* Buttons */
    .stButton > button {
        background-color: #0D6EFD;
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 1.25rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #0A58CA;
        color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(13, 110, 253, 0.2);
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.3);
    }

    /* Expander */
    .stExpander {
        border: 1px solid #DEE2E6;
        border-radius: 0.5rem;
        background-color: #FDFDFD;
    }
    .stExpander > details > summary {
        font-weight: 500;
        color: #0A58CA;
        padding: 0.5rem 0;
    }
    .stExpander > details {
        padding: 0 1rem 1rem 1rem;
    }

    /* Info/Success/Error Messages */
    .stAlert {
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    .stAlert[data-baseweb="alert-success"] {
        background-color: #E6F7F0;
        color: #0D6B42;
    }
    .stAlert[data-baseweb="alert-info"] {
        background-color: #E7F3FF;
        color: #0D6EFD;
    }
    .stAlert[data-baseweb="alert-warning"] {
        background-color: #FFF8E1;
        color: #B35B00;
    }
    .stAlert[data-baseweb="alert-error"] {
        background-color: #FDEDED;
        color: #D92D2D;
    }

</style>
""", unsafe_allow_html=True)

# --- 4. Data Loading ---
@st.cache_data
def load_data():
    """Loads data, cleans, standardizes 'price', returns df and scaler."""
    try:
        file_path = 'Clean_Dataset.csv'
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        if 'flight' in df.columns: df = df.drop(columns=['flight'], errors='ignore')
        target_scaler = StandardScaler()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['price'], inplace=True)
        if 'price' in df.columns and pd.api.types.is_numeric_dtype(df['price']):
             # 'price' column is used for logic, 'fare' is for display
             df['price_standardized'] = target_scaler.fit_transform(df[['price']])
        else: st.error("Fare column ('price') error."); st.stop()
        return df, target_scaler
    except FileNotFoundError: st.error(f"**Error: 'Clean_Dataset.csv' not found!**"); st.stop()
    except Exception as e: st.error(f"Error loading data: {str(e)}"); st.stop()

df_original, target_scaler = load_data()

# --- 5. Feature Definition ---
def define_feature_types(df):
    """Identifies features, excluding targets and duration."""
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # 'price' and 'price_standardized' are targets
    for col in ['price', 'price_standardized', 'duration']:
        if col in numerical_columns: numerical_columns.remove(col)
    return categorical_columns, numerical_columns

categorical_columns_all, numerical_columns_all = define_feature_types(df_original)

# --- 6. Sidebar Controls ---
st.sidebar.title("✈️ Flight Fare Predictor")
st.sidebar.header("⚙️ Controls & Settings")
st.sidebar.subheader("Filter Data by Class")
class_filter = st.sidebar.radio(
    "Select Flight Class (for Overview & Analysis):",
    ['All Classes', 'Economy', 'Business'], index=0, key='class_filter', label_visibility="collapsed"
)
if class_filter == 'Economy':
    df_display = df_original[df_original['class'] == 'Economy'].copy()
    st.sidebar.caption("Showing **Economy** flights for Overview/Analysis.")
elif class_filter == 'Business':
    df_display = df_original[df_original['class'] == 'Business'].copy()
    st.sidebar.caption("Showing **Business** flights for Overview/Analysis.")
else:
    df_display = df_original.copy()
    st.sidebar.caption("Showing **All Classes** for Overview/Analysis.")
if df_display.empty: st.warning(f"No data for '{class_filter}'."); st.stop()

st.sidebar.subheader("App Navigation")
app_mode = st.sidebar.radio(
    "Go to:",
    ["Data Overview", "Fare Analysis", "Model Performance", "Predict Fare"], # <-- Updated
    index=0, key='app_mode', label_visibility="collapsed"
)

st.sidebar.subheader("Group # 1 Details")
st.sidebar.markdown("""
<div style="font-size: 0.9em; line-height: 1.4; background-color: #F8F9FA; padding: 0.75rem; border-radius: 0.5rem;">
- Govind Wattamwar: pgdsai25033<br>
- Mehul Jevani: pgdsai25028<br>
- Saurabh Chhabra: pgdsai25032<br>
- Sarfaraj Pansare: pgdsai25029<br>
- Amit Fegade: pgdsai25030
</div>
""", unsafe_allow_html=True)

# --- 7. Helper Function for Prediction Summary ---
def display_prediction_summary(input_data_dict, title="Prediction Input Summary"):
    """Displays the input values in a horizontal table."""
    st.subheader(title)
    display_dict = {key.replace('_', ' ').title(): value for key, value in input_data_dict.items()}
    summary_df = pd.DataFrame([display_dict])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# --- 8. Cached Model Training Functions ---
@st.cache_data(show_spinner=False)
def run_model_comparison(_X_data, _y_data, _models, _preprocessor):
    """Trains multiple models for comparison and returns performance metrics."""
    performance_results = []; progress_bar = st.progress(0, text="Initializing comparison..."); total_models = len(_models)
    X_train, X_test, y_train, y_test = train_test_split(_X_data, _y_data, test_size=0.2, random_state=42)
    for i, (name, model) in enumerate(_models.items()):
        progress_bar.progress((i) / total_models, text=f"Training {name}...")
        pipeline = Pipeline([('preprocessor', _preprocessor), ('regressor', model)])
        try:
            pipeline.fit(X_train, y_train); y_pred = pipeline.predict(X_test)
            mse=mean_squared_error(y_test, y_pred); rmse=np.sqrt(mse); mae=mean_absolute_error(y_test, y_pred); r2=r2_score(y_test, y_pred)
            performance_results.append({'Model': name, 'MSE (Scaled)': mse, 'RMSE (Scaled)': rmse, 'MAE (Scaled)': mae, 'R² (Scaled)': r2, 'Error': None})
        except Exception as e:
             performance_results.append({'Model': name, 'MSE (Scaled)': np.nan, 'RMSE (Scaled)': np.nan, 'MAE (Scaled)': np.nan, 'R² (Scaled)': np.nan, 'Error': str(e)})
    progress_bar.progress(1.0, text="Comparison complete!")
    return pd.DataFrame(performance_results)

@st.cache_resource(show_spinner=False)
def get_prediction_models_dict(_df_full_data, _cat_features, _num_features):
    """Trains pipelines for multiple regression models using the full dataset."""
    prediction_models = {
        'Linear Regression': LinearRegression(), 'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    required_cols = _cat_features + _num_features
    missing = [c for c in required_cols if c not in _df_full_data.columns]
    if missing: st.error(f"Error: Missing columns: {missing}"); return {name: None for name in prediction_models.keys()}
    if 'price_standardized' not in _df_full_data.columns: st.error("Error: 'price_standardized' missing."); return {name: None for name in prediction_models.keys()}
    X_train_all = _df_full_data[required_cols]; y_train_all_scaled = _df_full_data['price_standardized']
    preprocessor_all = ColumnTransformer(transformers=[('num', StandardScaler(), _num_features), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), _cat_features)], remainder='passthrough')
    trained_pipelines = {}
    with st.spinner("Training prediction models (one-time setup)..."):
        for name, model in prediction_models.items():
            pipeline = Pipeline([('preprocessor', preprocessor_all), ('regressor', model)])
            try:
                pipeline.fit(X_train_all, y_train_all_scaled)
                trained_pipelines[name] = pipeline
                if 'processed_feature_names_pred' not in st.session_state:
                     try:
                        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
                        ohe_names = ohe.get_feature_names_out(_cat_features)
                        st.session_state['processed_feature_names_pred'] = _num_features + list(ohe_names)
                     except Exception: st.session_state['processed_feature_names_pred'] = None
            except Exception as e: st.warning(f"Could not train {name}: {e}"); trained_pipelines[name] = None
    return trained_pipelines

# --- Trigger Training of Prediction Models ---
categorical_features_for_pred_model = categorical_columns_all.copy()
numerical_features_for_pred_model = numerical_columns_all.copy() # Excludes duration
trained_prediction_pipelines = get_prediction_models_dict(
    df_original, categorical_features_for_pred_model, numerical_features_for_pred_model
)


# --- 9. Main Application Logic (Pages) ---

# --- Header Image ---
try:
    with open("flightPrediction.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/jpeg;base64,{b64_string}" 
                     alt="Flight Fare Banner" 
                     width="900" 
                     height="400" 
                     style="border-radius: 12px; box-shadow: 0 6px 16px rgba(0,0,0,0.08);">
            </div>
            <br/>
            """,
            unsafe_allow_html=True
        )
        
except FileNotFoundError:
    st.warning("Could not load header image 'flightPrediction.jpg'. Make sure it's in the same folder as the script.")
except Exception as e:
    st.warning(f"An error occurred loading the header image: {e}")
# --- END Header ---


# =============================================================================
# PAGE 1: DATA OVERVIEW
# =============================================================================
if app_mode == "Data Overview":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"📊 Data Overview ({class_filter})")
    st.markdown("A quick look at the dataset structure and key statistics.")
    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Flights", f"{len(df_display):,}")
    with col2: st.metric("Fare Range (₹)", f"{df_display['price'].min():,.0f} - {df_display['price'].max():,.0f}") # <-- Updated
    with col3: st.metric("Average Fare (₹)", f"{df_display['price'].mean():,.0f}") # <-- Updated
    st.markdown("---")
    col_data, col_types = st.columns([3, 1])
    with col_data:
        st.subheader("Data Sample")
        # Display 'price' column as 'fare' for user
        display_sample = df_display.drop(columns=['price_standardized', 'duration'], errors='ignore').head(10)
        display_sample = display_sample.rename(columns={'price': 'fare'})
        st.dataframe(display_sample, use_container_width=True)
    with col_types:
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame(df_display.drop(columns=['price_standardized', 'duration'], errors='ignore').dtypes, columns=['Data Type']).astype(str)
        dtype_df.index.name = "Feature"
        dtype_df = dtype_df.rename(index={'price': 'fare'})
        st.dataframe(dtype_df, use_container_width=True)
    st.markdown("---")
    st.subheader("Fare Distribution Analysis") # <-- Updated
    col_hist, col_box = st.columns(2)
    with col_hist:
        st.markdown("**Fare Histogram (Original ₹)**") # <-- Updated
        fig_hist = px.histogram(df_display, x='price', nbins=50, title=f"Frequency of Fares ({class_filter})", labels={'price': 'Fare (₹)'}) # <-- Updated
        fig_hist.update_layout(bargap=0.1); st.plotly_chart(fig_hist, use_container_width=True)
    with col_box:
        st.markdown("**Fare Box Plot (Original ₹)**") # <-- Updated
        fig_box = px.box(df_display, y='price', title=f"Spread of Fares ({class_filter})", labels={'price': 'Fare (₹)'}) # <-- Updated
        st.plotly_chart(fig_box, use_container_width=True)
    with st.expander("ℹ️ How to Read These Distribution Charts"):
        st.markdown("""
            * **Histogram:** Shows how many flights fall into different fare buckets (ranges). High bars = common fares.
            * **Box Plot:** Shows fare spread. Box = middle 50%, line = median. Whiskers show typical range, dots are outliers.
        """)
    # Footer Moved Here
    st.markdown("---")
    st.markdown("##### 🛠️ App Technical Details")
    with st.expander("Show Implementation Notes"):
        st.markdown(f"""
        * **Data Source:** `Clean_Dataset.csv` loaded locally using `@st.cache_data`.
        * **Current Filter:** Analyzing **{class_filter}** data (Overview & Analysis pages only).
        * **Target Scaling:** `StandardScaler` applied to 'price' column *before* training. Predictions inverse-transformed to ₹.
        * **Preprocessing:** `Pipeline` with `ColumnTransformer` (Numerical: `StandardScaler`, Categorical: `OneHotEncoder`). **Duration** feature excluded.
        * **Model Comparison:** Evaluates multiple models, ranked by R² (using filtered data).
        * **Predict Fare Page:** Trains multiple models on **All Classes** data. User selects model, inputs details, and gets prediction in ₹.
        * **Model Caching:** Uses `@st.cache_resource` for pipelines, `@st.cache_data` for data/comparison results.
        * **UI:** Streamlit multi-page app with sidebar, Plotly charts, forms, modern CSS.
        """)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2: FARE ANALYSIS
# =============================================================================
elif app_mode == "Fare Analysis": # <-- Updated
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"🔍 Fare Analysis ({class_filter})") # <-- Updated
    st.markdown("Exploring factors affecting flight fares (using original fares in ₹).") # <-- Updated
    # Row 1: Airline Analysis
    st.subheader("Airline Analysis"); col_airline1, col_airline2 = st.columns(2)
    with col_airline1:
        st.markdown("**Fare Distribution by Airline**"); fig_airline_box = px.box(df_display, x='airline', y='price', color='airline', title=f"Fare Distribution by Airline ({class_filter})", labels={'airline': 'Airline', 'price': 'Fare (₹)'}) # <-- Updated
        fig_airline_box.update_layout(xaxis_tickangle=-45); st.plotly_chart(fig_airline_box, use_container_width=True)
        with st.expander("ℹ️ How to Read (Airline vs. Fare)"): # <-- Updated
            st.markdown("""
                * **What it shows:** This compares the fare spread (using box plots) for each airline.
                * **Compare Medians:** The line inside each box shows the median fare for that airline. Higher lines indicate typically more expensive airlines.
                * **Compare Box Sizes:** A taller box indicates greater variability in the middle 50% of fares for that airline.
                * **Use Case:** Helps visually identify budget vs. premium carriers.
            """)
    with col_airline2:
        st.markdown("**Airline Market Share**"); airline_counts = df_display['airline'].value_counts().reset_index(); airline_counts.columns = ['airline', 'count']
        fig_airline_pie = px.pie(airline_counts, names='airline', values='count', title=f'Airline Market Share ({class_filter})', hole=0.3)
        fig_airline_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value'); st.plotly_chart(fig_airline_pie, use_container_width=True)
        with st.expander("ℹ️ How to Read Pie Chart"):
            st.markdown("""
                * **What it shows:** Each slice represents an airline.
                * **Size of Slice:** Corresponds to the percentage of total flights (in the filtered data) operated by that airline.
            """)
    st.markdown("---")
    # Row 2: Time Analysis
   
    # # Row 3: Booking Time
    st.subheader("Fare vs. Booking Time") # <-- Updated
    st.markdown("**Average Fare vs. Days Before Departure (Trend)**") # <-- Updated
    daily_avg_price = df_display.groupby('days_left')['price'].mean().reset_index()
    fig_days_trend = px.line(daily_avg_price, x='days_left', y='price', markers=True, title=f"Average Fare Trend vs. Days Left ({class_filter})", labels={'days_left': 'Days Before Departure', 'price': 'Average Fare (₹)'}) # <-- Updated
    fig_days_trend.update_traces(line_color='#28A745'); st.plotly_chart(fig_days_trend, use_container_width=True)
    with st.expander("ℹ️ How to Read (Days Left vs. Fare Trend)"): # <-- Updated
        st.markdown("""
            * **Line:** Shows the *average* fare for all flights booked a specific number of days before departure.
            * **Shape:** Clearly visualizes the overall trend. A downward slope confirms booking earlier is cheaper on average.
        """)
    st.markdown("---")
    # Row 4: Correlation Heatmap
    st.subheader("Feature Correlation Heatmap"); st.markdown("Visualizes linear relationships between features and the **fare**.") # <-- Updated
    st.info("ℹ️ Note: Uses Label Encoding for categorical features to calculate correlation.")
    try:
        df_heatmap = df_display.copy(); label_encoders = {}
        for col in df_heatmap.select_dtypes(include=['object']).columns: le = LabelEncoder(); df_heatmap[col] = le.fit_transform(df_heatmap[col].astype(str)); label_encoders[col] = le
        cols_to_drop_heatmap = ['price_standardized', 'duration']; df_heatmap_numeric = df_heatmap.select_dtypes(include=np.number).drop(columns=cols_to_drop_heatmap, errors='ignore')
        if not df_heatmap_numeric.empty and 'price' in df_heatmap_numeric.columns:
            corr_matrix = df_heatmap_numeric.corr(); fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag_r', linewidths=.5, ax=ax_heatmap, annot_kws={"size": 8})
            plt.title(f'Correlation Matrix (Features vs. Fare) ({class_filter})', fontsize=16); st.pyplot(fig_heatmap)
            with st.expander("ℹ️ How to Read Heatmap"):
                st.markdown("""
                    * **Colors & Numbers:** Show correlation (-1 to +1). Red = positive, Blue = negative, Light = weak.
                    * **Focus on 'price' Row/Column:** Identifies features with the strongest *linear* association with fare. (Note: 'price' is the column name for 'fare').
                    * **Limitations:** Only shows linear relationships, not complex ones.
                """)
        else: st.warning("No numerical features for heatmap.")
    except Exception as e: st.error(f"Could not generate heatmap: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 3: MODEL PERFORMANCE COMPARISON
# =============================================================================
elif app_mode == "Model Performance":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"⚖️ Model Performance Comparison ({class_filter})")
    st.markdown("Comparing models using standardized **fare** prediction. Ranked by R².") # <-- Updated
    st.info("ℹ️ Models trained on standardized fare ('price_standardized'). R² is comparable regardless of scaling.") # <-- Updated
    models_to_compare = {
        'Linear Regression': LinearRegression(), 'Ridge Regression (L2)': Ridge(alpha=1.0),
        'Lasso Regression (L1)': Lasso(alpha=0.001), 'Elastic Net': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
    }
    current_categorical_comp = categorical_columns_all.copy(); current_numerical_comp = numerical_columns_all.copy()
    if class_filter != 'All Classes' and 'class' in current_categorical_comp: current_categorical_comp.remove('class')
    required_cols = current_categorical_comp + current_numerical_comp
    missing_cols = [col for col in required_cols if col not in df_display.columns]
    if missing_cols: st.error(f"Missing columns: {missing_cols}."); st.stop()
    if df_display.empty or len(df_display) < 50: st.warning(f"Insufficient data ({len(df_display)} rows)."); st.stop()
    X = df_display[required_cols]; y = df_display['price_standardized']
    preprocessor_comp = ColumnTransformer(transformers=[('num', StandardScaler(), current_numerical_comp), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), current_categorical_comp)], remainder='passthrough')
    if st.button("Compare All Models", type="primary", use_container_width=True):
        performance_df = run_model_comparison(X, y, models_to_compare, preprocessor_comp) # Cached
        st.session_state['performance_df'] = performance_df; st.session_state['comparison_run'] = True
    if st.session_state.get('comparison_run', False):
        if 'performance_df' in st.session_state:
            performance_df = st.session_state['performance_df']
            st.subheader("📊 Performance Metrics (Standardized Scale - Ranked by R²)")
            st.markdown("Lower **MSE**, **RMSE**, **MAE** better. Higher **R²** better.")
            display_df = performance_df.copy(); display_df = display_df.sort_values(by='R² (Scaled)', ascending=False).reset_index(drop=True)
            for col in ['MSE (Scaled)', 'RMSE (Scaled)', 'MAE (Scaled)', 'R² (Scaled)']:
                if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "Error")
            st.dataframe(display_df.drop(columns=['Error'], errors='ignore'), use_container_width=True)
            if 'Error' in performance_df.columns and performance_df['Error'].notna().any(): st.warning("Errors occurred:"); st.dataframe(performance_df[['Model', 'Error']].dropna(), use_container_width=True)
            st.markdown("---"); st.subheader("📈 Visual Performance Comparison")
            col_chart1, col_chart2 = st.columns(2)
            plot_df = performance_df.dropna(subset=['R² (Scaled)', 'RMSE (Scaled)']).sort_values(by='R² (Scaled)', ascending=False)
            if not plot_df.empty:
                 with col_chart1: st.markdown("**R² Score Comparison**"); fig_r2 = px.bar(plot_df, x='Model', y='R² (Scaled)', title='R² Score (Higher is Better)', color='R² (Scaled)', text_auto='.4f', color_continuous_scale='viridis'); min_r2_vis = max(0, plot_df['R² (Scaled)'].min() - 0.05) if pd.notna(plot_df['R² (Scaled)'].min()) else 0; max_r2_vis = min(1, plot_df['R² (Scaled)'].max() + 0.05) if pd.notna(plot_df['R² (Scaled)'].max()) else 1; fig_r2.update_layout(yaxis_range=[min_r2_vis ,max_r2_vis]); fig_r2.update_traces(textangle=0, textposition="outside"); st.plotly_chart(fig_r2, use_container_width=True)
                 with col_chart2: st.markdown("**RMSE Comparison**"); fig_rmse = px.bar(plot_df, x='Model', y='RMSE (Scaled)', title='RMSE (Lower is Better)', color='RMSE (Scaled)', text_auto='.4f', color_continuous_scale='plasma_r'); fig_rmse.update_traces(textangle=0, textposition="outside"); st.plotly_chart(fig_rmse, use_container_width=True)
                 with st.expander("ℹ️ How to Read Comparison Charts"):
                     st.markdown("""
                        * **R² Score Chart (Higher is Better):** Shows the proportion of the fare variation that each model can explain (on the scaled data). Bars closer to 1.0 indicate a better fit.
                        * **RMSE Chart (Lower is Better):** Shows the typical prediction error for each model (on the scaled data). Lower bars are better.
                    """)
            else: st.warning("No valid results to display charts.")
        else: st.warning("Performance data not found.")
    else: st.info("Click 'Compare All Models' button.")
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 4: PREDICT FARE
# =============================================================================
elif app_mode == "Predict Fare": # <-- Updated
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"🎯 Flight Fare Prediction") # <-- Updated
    st.markdown("Get a fare estimate using different regression models trained on **All Classes** data.") # <-- Updated
    st.info("ℹ️ Select your flight details (Source/Destination required) and choose a model to predict the fare in Rupees (₹).") # <-- Updated

    if not trained_prediction_pipelines or all(p is None for p in trained_prediction_pipelines.values()):
        st.error("Prediction models could not be trained or loaded. Cannot proceed.")
        st.markdown('</div>', unsafe_allow_html=True); st.stop()

    with st.form("prediction_form_unified"):
        st.subheader("Enter Flight Details for Prediction:")
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
            source_city = st.selectbox("Source City*", sorted(df_original['source_city'].unique()), key="pred_source", help="Mandatory: Select departure city.")
            airline_input = st.selectbox("Airline (Optional)", ["Any"] + sorted(df_original['airline'].unique()), key="pred_airline", help="Select 'Any' for typical airline.")
            stops_input = st.selectbox("Stops (Optional)", ["Any"] + sorted(df_original['stops'].unique(), key=lambda x: ("zero", "one", "two_or_more").index(x) if x != "Any" else -1), key="pred_stops", help="Select 'Any' for typical stops.")
        with col_form2:
            all_destinations = sorted(df_original['destination_city'].unique())
            available_destinations = [city for city in all_destinations if city != source_city] # Dynamic filter
            if not available_destinations: available_destinations = all_destinations
            destination_city = st.selectbox("Destination City*", available_destinations, key="pred_dest", help="Mandatory: Select arrival city.")
            class_options = sorted(df_original['class'].unique())
            try: default_class_index = class_options.index('Economy')
            except ValueError: default_class_index = 0
            flight_class_input = st.selectbox("Class*", class_options, index=default_class_index, key="pred_class", help="Mandatory: Select flight class.")
            days_left = st.slider("Days Before Departure", min_value=1, max_value=50, value=15, step=1, key="pred_days")
        with col_form3:
            available_models = {name: pipe for name, pipe in trained_prediction_pipelines.items() if pipe is not None}
            if not available_models: st.error("No models available."); st.stop()
            model_name_predict = st.selectbox( "Select Prediction Model*", list(available_models.keys()), key="pred_model_select", help="Mandatory: Choose algorithm.")
            # Duration removed
            
            # Add a placeholder to balance the column
            st.empty()
            st.empty()

        # Define defaults for hidden features
        default_departure_time = df_original['departure_time'].mode()[0]
        default_arrival_time = df_original['arrival_time'].mode()[0]
        
        st.markdown("---")
        predict_button = st.form_submit_button("Predict Fare", type="primary", use_container_width=True) # <-- Updated

        if predict_button:
            if source_city == destination_city: st.error("Source and Destination cannot be the same.")
            else:
                try:
                    input_airline = df_original['airline'].mode()[0] if airline_input == "Any" else airline_input
                    input_stops = df_original['stops'].mode()[0] if stops_input == "Any" else stops_input
                    input_class = flight_class_input
                    input_dict = {
                        'airline': input_airline, 'source_city': source_city, 'departure_time': default_departure_time,
                        'stops': input_stops, 'arrival_time': default_arrival_time, 'destination_city': destination_city,
                        'class': input_class, 'days_left': days_left #'duration' removed
                    }
                    input_df = pd.DataFrame([input_dict])
                    input_cols = categorical_features_for_pred_model + numerical_features_for_pred_model
                    input_df_filtered = input_df[input_cols]
                    selected_pipeline = available_models[model_name_predict]
                    predicted_price_scaled = selected_pipeline.predict(input_df_filtered)[0]
                    predicted_price_real = target_scaler.inverse_transform([[predicted_price_scaled]])[0][0]
                    
                    # Display Result
                    st.success(f"### Predicted Flight Fare ({model_name_predict}): **₹{predicted_price_real:,.2f}**") # <-- Updated
                    st.markdown("---"); st.subheader("Prediction Context")
                    compare_class = input_class
                    similar_flights_query = ((df_original['airline'] == input_airline) & (df_original['source_city'] == source_city) & (df_original['destination_city'] == destination_city) & (df_original['class'] == compare_class) & (df_original['stops'] == input_stops) & (df_original['days_left'].between(max(1, days_left - 3), days_left + 3)))
                    similar_flights = df_original[similar_flights_query]
                    st.markdown("**Comparison with Similar Historical Flights:**");
                    if not similar_flights.empty:
                        similar_avg=similar_flights['price'].mean(); similar_min=similar_flights['price'].min(); similar_max=similar_flights['price'].max(); similar_count=len(similar_flights)
                        st.markdown(f"* Found **{similar_count}** flights...\n* **Observed Fare Range:** ₹{similar_min:,.0f} - ₹{similar_max:,.0f}\n* **Observed Average Fare:** ₹{similar_avg:,.0f}\n* **Your Prediction:** **₹{predicted_price_real:,.2f}**"); # <-- Updated
                        with st.expander("ℹ️ How to Read Market Comparison"):
                             st.markdown("""
                                * **What it shows:** Compares your prediction to the *actual fares* of historical flights that match your selected Airline, Route, Class, Stops, and are within +/- 3 days of your selected booking time.
                                * **Prediction vs. Average:** How does the model's prediction compare to the average fare people actually paid for similar flights?
                                * **Prediction within Range:** Does the prediction fall within the minimum and maximum fares observed?
                            """) # <-- Updated
                    else: st.warning("No closely matching flights found in the historical data for comparison.")
                except Exception as e: st.error(f"Error during prediction: {str(e)}"); st.exception(e)
    st.markdown('</div>', unsafe_allow_html=True)


# --- 10. Footer/Technical Details ---
# (Footer is now only on Data Overview page)

# --- End of Script ---


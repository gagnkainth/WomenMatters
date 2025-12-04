import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_page_config(page_title="WomenMatters", layout="wide")

# --- Custom CSS for colorful, modern look --- 
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    /* Sub-header styling */
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    /* Insight box styling */
    .insight-box {
        background: linear-gradient(90deg, #ffdde1 0%, #ee9ca7 100%);
        padding: 1.2rem;
        border-radius: 14px;
        border-left: 6px solid #FF6B6B;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(78,205,196,0.07);
    }
    /* Data section card styling */
    .data-section {
        background: linear-gradient(90deg, #f7fff7 0%, #c9fffd 100%);
        padding: 1rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(78,205,196,0.07);
    }
    /* End note styling */
    .end-note {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: 2rem;
        letter-spacing: 1px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fff7 0%, #c9fffd 100%);
    }
    /* Table header styling */
    th {
        background-color: #4ECDC4 !important;
        color: #fff !important;
    }
    /* Chart title styling */
    .js-plotly-plot .gtitle {
        fill: #FF6B6B !important;
        font-weight: bold !important;
    }
    /* Colorful markdown headers */
    .gradient-header {
        background: linear-gradient(90deg, #ffdde1 0%, #ee9ca7 100%);
        padding: 0.7rem 1rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
        border-left: 6px solid #FF6B6B;
        box-shadow: 0 2px 8px rgba(78,205,196,0.07);
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-header">🌍 WomenMatters: Analyzing Global Issue</div>', unsafe_allow_html=True)

# --- Description ---
st.markdown("""
<div class="gradient-header">📊 Dashboard Overview</div>
<div style="font-size:1.2rem; color:#444; margin-bottom:1.5rem;">
    This dashboard analyzes <span style="color:#FF6B6B;font-weight:bold;">global survey data</span> on attitudes that justify violence against women.<br>
    <span style="color:#4ECDC4;">Explore how responses vary by country, gender, education, and other demographics.</span><br>
    Interactive charts and insights help uncover patterns and highlight areas for intervention.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("women_violence.csv")
    return df

df_raw = load_data()

# --- Data Cleaning ---
def clean_data(df):
    df_clean = df.copy()
    # Fill missing 'Value' with median
    if 'Value' in df_clean.columns:
        median_value = df_clean['Value'].median()
        df_clean['Value'] = df_clean['Value'].fillna(median_value)
    # Convert 'Survey Year' to datetime
    if 'Survey Year' in df_clean.columns:
        df_clean['Survey Year'] = pd.to_datetime(df_clean['Survey Year'], errors='coerce')
    # Drop rows with missing critical fields
    df_clean = df_clean.dropna(subset=['Value', 'Survey Year'])
    return df_clean

df_clean = clean_data(df_raw)

# --- Show First 5 Rows Before and After Cleaning ---
st.markdown('<div class="gradient-header">🗂 Data Preview</div>', unsafe_allow_html=True)
st.markdown('<div class="data-section">', unsafe_allow_html=True)
st.markdown("**First 5 Rows: Raw Data**")
st.dataframe(df_raw.head(5), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="data-section">', unsafe_allow_html=True)
st.markdown("**First 5 Rows: Cleaned Data**")
df_clean_display = df_clean.copy()
df_clean_display['Survey Year'] = df_clean_display['Survey Year'].dt.strftime('%Y-%m-%d')
st.dataframe(df_clean_display.head(5), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Quick Insights on Preprocessing ---
st.markdown("""
<div class="insight-box">
    <span style="font-size:1.3rem;font-weight:bold;">🛠️ Preprocessing Steps:</span>
    <ul>
        <li>Filled missing <b>Value</b> entries with <span style="color:#FF6B6B;">median</span></li>
        <li>Converted <b>Survey Year</b> to <span style="color:#4ECDC4;">datetime format</span></li>
        <li>Dropped rows with missing critical fields</li>
        <li>Ensured consistent data types for analysis</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.markdown('<div style="font-size:1.5rem;font-weight:bold;color:#4ECDC4;margin-bottom:1rem;">🔎 Interactive Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown("Customize your analysis by selecting specific parameters.")

st.sidebar.markdown("---")

# Year filter
year_min = int(df_clean['Survey Year'].dt.year.min())
year_max = int(df_clean['Survey Year'].dt.year.max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)

# Country filter
countries = sorted(df_clean['Country'].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries[:5]
)

# Gender filter
genders = sorted(df_clean['Gender'].dropna().unique())
selected_genders = st.sidebar.multiselect(
    "Select Gender",
    options=genders,
    default=genders
)

# Demographics Question filter
demographics_questions = sorted(df_clean['Demographics Question'].dropna().unique())
selected_demographics = st.sidebar.multiselect(
    "Select Demographics Question",
    options=demographics_questions,
    default=demographics_questions
)

# Submit button
st.sidebar.markdown("---")
submit_button = st.sidebar.button(
    "Apply Filters",
    type="primary",
    use_container_width=True,
    help="Click to apply your selected filters and update all charts"
)

# --- Filtered Data Logic ---
if 'df_filtered' not in st.session_state:
    st.session_state['df_filtered'] = df_clean.copy()

if submit_button:
    filtered = df_clean[
        (df_clean['Survey Year'].dt.year >= year_range[0]) &
        (df_clean['Survey Year'].dt.year <= year_range[1]) &
        (df_clean['Country'].isin(selected_countries)) &
        (df_clean['Gender'].isin(selected_genders)) &
        (df_clean['Demographics Question'].isin(selected_demographics))
    ]
    st.session_state['df_filtered'] = filtered
    st.sidebar.success(f"Filters applied! Showing {len(filtered):,} records.")

st.sidebar.write(f"**Records Shown:** {len(st.session_state['df_filtered']):,}")

# Use filtered data for all charts below
df_filtered = st.session_state['df_filtered']

# --- Chart 1: Distribution by Demographics Question ---
st.markdown('<div class="gradient-header">📊 Distribution of Value by Demographics Question</div>', unsafe_allow_html=True)
fig1 = px.box(df_filtered, x='Demographics Question', y='Value', title='Distribution of Value by Demographics Question')
fig1.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Value distributions vary widely across demographic categories, highlighting key areas for intervention.
</div>
""", unsafe_allow_html=True)

# --- Chart 2: Gender Distribution ---
st.markdown('<div class="gradient-header">🧑‍🤝‍🧑 Gender Distribution</div>', unsafe_allow_html=True)
gender_counts = df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
fig2 = px.pie(gender_counts, values='Count', names='Gender', title='Distribution of Gender')
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> The dataset contains responses from both genders, enabling comparative analysis.
</div>
""", unsafe_allow_html=True)

# --- Chart 3: Word Cloud of Questions ---
st.markdown('<div class="gradient-header">☁️ Common Survey Questions (Word Cloud)</div>', unsafe_allow_html=True)
text = " ".join(str(q) for q in df_filtered['Question'].dropna())
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Frequently asked questions focus on specific justifications for violence.
</div>
""", unsafe_allow_html=True)

# --- Chart 4: Average Value by Question ---
st.markdown('<div class="gradient-header">📈 Average Value by Question</div>', unsafe_allow_html=True)
df_question_avg = df_filtered.groupby('Question')['Value'].mean().reset_index()
fig4 = px.bar(df_question_avg, x='Question', y='Value', title='Average Value by Question')
fig4.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Some questions elicit higher justification rates than others.
</div>
""", unsafe_allow_html=True)

# --- Chart 5: Average Value Over Years ---
st.markdown('<div class="gradient-header">📅 Average Value Over Years</div>', unsafe_allow_html=True)
df_yearly_avg = df_filtered.groupby(df_filtered['Survey Year'].dt.year)['Value'].mean().reset_index()
fig5 = px.line(df_yearly_avg, x='Survey Year', y='Value', title='Average Value Over Years')
st.plotly_chart(fig5, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Trends over time reveal changes in societal attitudes.
</div>
""", unsafe_allow_html=True)

# --- Chart 6: Sunburst by Demographics ---
st.markdown('<div class="gradient-header">🌞 Demographics Hierarchy (Sunburst)</div>', unsafe_allow_html=True)
df_sunburst_data = df_filtered.groupby(['Demographics Question', 'Demographics Response'])['Value'].mean().reset_index()
fig6 = px.sunburst(df_sunburst_data, path=['Demographics Question', 'Demographics Response'], values='Value', title='Average Value by Demographics Hierarchy')
st.plotly_chart(fig6, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Sunburst chart shows how justification rates differ across demographic hierarchies.
</div>
""", unsafe_allow_html=True)

# --- Chart 7: Heatmap of Demographics vs Question ---
st.markdown('<div class="gradient-header">🔥 Heatmap: Demographics vs Question</div>', unsafe_allow_html=True)
df_heatmap_data = df_filtered.groupby(['Demographics Question', 'Question'])['Value'].mean().reset_index()
heatmap_data = df_heatmap_data.pivot(index='Demographics Question', columns='Question', values='Value')
fig7 = px.imshow(heatmap_data, labels=dict(x="Question", y="Demographics Question", color="Average Value"), title="Average Value by Demographics Question and Question")
fig7.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig7, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Heatmap highlights which questions are most justified by different demographic groups.
</div>
""", unsafe_allow_html=True)

# --- Chart 8: Distribution by Country ---
st.markdown('<div class="gradient-header">🌏 Distribution of Value by Country</div>', unsafe_allow_html=True)
fig8 = px.violin(df_filtered, x='Country', y='Value', title='Distribution of Value by Country')
fig8.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig8, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Country-wise distributions reveal regional differences in attitudes.
</div>
""", unsafe_allow_html=True)

# --- Chart 9: Gender Comparison by Demographics Question ---
st.markdown('<div class="gradient-header">👩‍🦰👨 Gender Comparison by Demographics Question</div>', unsafe_allow_html=True)
df_grouped_gender = df_filtered.groupby(['Demographics Question', 'Gender'])['Value'].mean().reset_index()
fig9 = px.bar(df_grouped_gender, x='Demographics Question', y='Value', color='Gender', title='Average Value by Demographics Question and Gender', barmode='group')
fig9.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig9, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Gender-based analysis uncovers differences in justification rates.
</div>
""", unsafe_allow_html=True)

# --- Chart 10: Stacked Bar of Demographics Responses ---
st.markdown('<div class="gradient-header">📊 Distribution of Demographics Responses</div>', unsafe_allow_html=True)
df_stacked_bar_data = df_filtered.groupby(['Demographics Question', 'Demographics Response']).size().reset_index(name='Count')
fig10 = px.bar(df_stacked_bar_data, x='Demographics Question', y='Count', color='Demographics Response', title='Distribution of Demographics Responses within Demographics Questions')
fig10.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig10, use_container_width=True)
st.markdown("""
<div class="insight-box">
    <b>Insight:</b> Stacked bar chart shows the composition of responses within each demographic category.
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# --- Insights Section (after charts) ---
st.markdown("""
<div class="gradient-header" style="margin-top:2rem;">
    <img src="https://img.icons8.com/color/48/000000/light-on--v1.png" width="36" style="vertical-align:middle;margin-right:10px;">
    Key Insights
</div>
""", unsafe_allow_html=True)

insight_cols = st.columns(2)
with insight_cols[0]:
    st.markdown("""
    <div class="insight-box">
        <b>🔎 Demographic Patterns:</b><br>
        Higher justification rates are observed in certain age groups and education levels, indicating the need for targeted awareness campaigns.
    </div>
    """, unsafe_allow_html=True)
with insight_cols[1]:
    st.markdown("""
    <div class="insight-box">
        <b>🌍 Regional Differences:</b><br>
        Some countries and regions show significantly higher rates, highlighting cultural and societal influences on attitudes toward violence.
    </div>
    """, unsafe_allow_html=True)

insight_cols2 = st.columns(2)
with insight_cols2[0]:
    st.markdown("""
    <div class="insight-box">
        <b>👩‍🦰👨 Gender Gap:</b><br>
        Gender-based analysis reveals differences in justification rates, emphasizing the importance of gender-sensitive interventions.
    </div>
    """, unsafe_allow_html=True)
with insight_cols2[1]:
    st.markdown("""
    <div class="insight-box">
        <b>📅 Time Trends:</b><br>
        Trends over years show gradual changes, suggesting the impact of policy, education, and advocacy efforts.
    </div>
    """, unsafe_allow_html=True)

# --- Project Conclusion Section ---
st.markdown("""
<div style="margin-top:2rem;">
    <div style="font-size:2rem; font-weight:bold; color:#FF6B6B; display:flex; align-items:center;">
        <img src="https://img.icons8.com/color/48/000000/goal--v1.png" width="36" style="margin-right:10px;">
        Project Conclusion
    </div>
    <hr style="border:1px solid #FF6B6B;margin-bottom:2rem;">
    <div style="background: linear-gradient(120deg, #e0c3fc 0%, #fcb7ff 100%);
                border-radius: 20px; padding: 1.2rem 2rem; margin-bottom:2rem;
                box-shadow: 0 2px 8px rgba(78,205,196,0.07); font-size:1.15rem; color:#333;">
        This analysis of global attitudes towards violence against women reveals significant demographic, regional, and temporal patterns.
        The dashboard provides actionable insights for policymakers, educators, and advocates to design effective interventions and promote positive societal change.
        Continued monitoring and targeted efforts are essential to reduce justification rates and foster safer environments for women worldwide.
    </div>
</div>
""", unsafe_allow_html=True)

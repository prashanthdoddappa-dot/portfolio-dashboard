import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================== PAGE CONFIGURATION ========================
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== MODERN CSS STYLING ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .profit-badge {
        background: linear-gradient(135deg, #00c851 0%, #00ff00 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .loss-badge {
        background: linear-gradient(135deg, #ff4444 0%, #ff6b6b 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .stock-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .filter-section {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: white;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
        border: 1px solid #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #f0f0f0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="metric-container"] > div {
        font-weight: 600;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .dataframe-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Custom styling for positive/negative values */
    .positive-value {
        color: #00c851;
        font-weight: 600;
    }
    
    .negative-value {
        color: #ff4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ======================== DATA PROCESSING FUNCTIONS ========================

ACCOUNT_CONFIG = {
    '13906-PRO': {
        'name': '13906-PRO',
        'columns': [0, 1, 2, 3, 4, 5, 6],
        'col_names': ['INE_Code', 'Stock', 'Quantity', 'Buy_Price', 'Buy_Value', 'Current_Price', 'Current_Value'],
        'color': '#667eea'
    },
    'Z00018-NKSQUARED': {
        'name': 'Z00018-NKSQUARED',
        'columns': [8, 9, 10, 11, 12, 13, 14],
        'col_names': ['INE_Code', 'Stock', 'Quantity', 'Buy_Price', 'Buy_Value', 'Current_Price', 'Current_Value'],
        'color': '#764ba2'
    },
    'Z00008-Kamath Associate': {
        'name': 'Z00008-Kamath Associate',
        'columns': [16, 17, 18, 19, 20, 21, 22],
        'col_names': ['INE_Code', 'Stock', 'Quantity', 'Buy_Price', 'Buy_Value', 'Current_Price', 'Current_Value'],
        'color': '#f50057'
    }
}

def safe_float(value):
    """Safely convert value to float"""
    if pd.isna(value) or value is None or value == '':
        return 0.0
    try:
        str_val = str(value).strip()
        if str_val.startswith('"') and str_val.endswith('"'):
            str_val = str_val[1:-1]
        str_val = str_val.replace(',', '').replace('"', '').replace("'", "").strip()
        if str_val == '' or str_val.lower() in ['nan', 'none', 'null']:
            return 0.0
        return float(str_val)
    except:
        return 0.0

def clean_text(value):
    """Clean text values"""
    if pd.isna(value) or value is None:
        return ''
    str_val = str(value).strip()
    if str_val.startswith('"') and str_val.endswith('"'):
        str_val = str_val[1:-1]
    return str_val.strip()

@st.cache_data
def load_and_process_data(file_path):
    """Load and process portfolio data"""
    try:
        df_raw = pd.read_csv(file_path, header=None)
        
        # Extract metadata
        filter_data = {}
        for i in range(min(9, len(df_raw))):
            key = clean_text(df_raw.iloc[i, 0])
            value = clean_text(df_raw.iloc[i, 1])
            if key and value:
                filter_data[key] = value
        
        # Process portfolio data
        portfolio_data = []
        for account_key, config in ACCOUNT_CONFIG.items():
            for row_idx in range(17, len(df_raw)):
                try:
                    row = df_raw.iloc[row_idx]
                    stock_data = {}
                    
                    for i, col_idx in enumerate(config['columns']):
                        if col_idx < len(row):
                            col_name = config['col_names'][i]
                            if col_name in ['Quantity', 'Buy_Price', 'Buy_Value', 'Current_Price', 'Current_Value']:
                                stock_data[col_name] = safe_float(row.iloc[col_idx])
                            else:
                                stock_data[col_name] = clean_text(row.iloc[col_idx])
                    
                    # Validate stock entry
                    if (stock_data.get('INE_Code', '').startswith('INE') and 
                        stock_data.get('Buy_Value', 0) > 0):
                        
                        # Add account info
                        stock_data['Account'] = config['name']
                        stock_data['Account_Color'] = config['color']
                        
                        # Calculate metrics
                        stock_data['Buy_Value_Cr'] = stock_data['Buy_Value'] / 10000000
                        stock_data['Current_Value_Cr'] = stock_data['Current_Value'] / 10000000
                        stock_data['Profit_Loss_Cr'] = stock_data['Current_Value_Cr'] - stock_data['Buy_Value_Cr']
                        stock_data['Returns_Percentage'] = (stock_data['Profit_Loss_Cr'] / stock_data['Buy_Value_Cr'] * 100) if stock_data['Buy_Value_Cr'] > 0 else 0
                        stock_data['Profit_Status'] = 'Profit' if stock_data['Returns_Percentage'] > 0 else 'Loss'
                        
                        portfolio_data.append(stock_data)
                except:
                    continue
        
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            df = df.drop_duplicates(subset=['Stock', 'Account'], keep='first')
            df = df.sort_values('Current_Value_Cr', ascending=False)
            return df, filter_data
        else:
            return create_sample_data(), {}
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return create_sample_data(), {}

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFOSYS', 'WIPRO', 'ITC', 'SBIN', 'BHARTIARTL', 'HCLTECH', 'MARUTI']
    accounts = list(ACCOUNT_CONFIG.keys())
    
    data = []
    for stock in stocks:
        for account in accounts:
            buy_value = np.random.uniform(50, 500)
            returns = np.random.uniform(-30, 150)
            current_value = buy_value * (1 + returns/100)
            
            data.append({
                'Stock': stock,
                'Account': account,
                'Account_Color': ACCOUNT_CONFIG[account]['color'],
                'Quantity': np.random.randint(1000, 100000),
                'Buy_Price': np.random.uniform(100, 5000),
                'Current_Price': np.random.uniform(100, 5000),
                'Buy_Value_Cr': buy_value,
                'Current_Value_Cr': current_value,
                'Profit_Loss_Cr': current_value - buy_value,
                'Returns_Percentage': returns,
                'Profit_Status': 'Profit' if returns > 0 else 'Loss'
            })
    
    return pd.DataFrame(data)

# ======================== VISUALIZATION FUNCTIONS ========================

def create_modern_metrics(df):
    """Create modern metric cards"""
    total_investment = df['Buy_Value_Cr'].sum()
    total_current = df['Current_Value_Cr'].sum()
    total_pnl = df['Profit_Loss_Cr'].sum()
    overall_return = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #666; margin: 0;">💰 Total Investment</h4>
            <h2 style="color: #333; margin: 0.5rem 0;">₹{:.2f} Cr</h2>
            <p style="color: #999; margin: 0;">Capital Deployed</p>
        </div>
        """.format(total_investment), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #666; margin: 0;">📈 Current Value</h4>
            <h2 style="color: #333; margin: 0.5rem 0;">₹{:.2f} Cr</h2>
            <p style="color: {}; margin: 0;">{}₹{:.2f} Cr</p>
        </div>
        """.format(total_current, 
                  "#00c851" if total_pnl > 0 else "#ff4444",
                  "+" if total_pnl > 0 else "",
                  total_pnl), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #666; margin: 0;">💎 Total Returns</h4>
            <h2 style="color: {}; margin: 0.5rem 0;">{}{:.2f}%</h2>
            <p style="color: #999; margin: 0;">Overall Performance</p>
        </div>
        """.format("#00c851" if overall_return > 0 else "#ff4444",
                  "+" if overall_return > 0 else "",
                  overall_return), unsafe_allow_html=True)
    
    with col4:
        profitable = len(df[df['Returns_Percentage'] > 0])
        total_stocks = len(df)
        win_rate = (profitable / total_stocks * 100) if total_stocks > 0 else 0
        
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #666; margin: 0;">🎯 Win Rate</h4>
            <h2 style="color: #333; margin: 0.5rem 0;">{:.1f}%</h2>
            <p style="color: #999; margin: 0;">{}/{} Profitable</p>
        </div>
        """.format(win_rate, profitable, total_stocks), unsafe_allow_html=True)

def create_advanced_charts(df):
    """Create advanced interactive charts"""
    
    # 1. Treemap of Portfolio
    fig_treemap = px.treemap(
        df,
        path=['Account', 'Stock'],
        values='Current_Value_Cr',
        color='Returns_Percentage',
        color_continuous_scale='RdYlGn',
        range_color=[-50, 50],
        title='Portfolio Composition - Interactive Treemap',
        hover_data={'Current_Value_Cr': ':.2f', 'Returns_Percentage': ':.2f'}
    )
    fig_treemap.update_layout(height=500)
    
    # 2. Scatter plot - Risk vs Return
    fig_scatter = px.scatter(
        df,
        x='Buy_Value_Cr',
        y='Returns_Percentage',
        size='Current_Value_Cr',
        color='Account',
        title='Risk-Return Analysis',
        labels={'Buy_Value_Cr': 'Investment (₹ Cr)', 'Returns_Percentage': 'Returns (%)'},
        hover_data={'Stock': True, 'Current_Value_Cr': ':.2f'}
    )
    fig_scatter.update_layout(height=400)
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # 3. Sunburst Chart
    fig_sunburst = px.sunburst(
        df,
        path=['Account', 'Stock'],
        values='Current_Value_Cr',
        color='Returns_Percentage',
        color_continuous_scale='RdYlGn',
        range_color=[-50, 50],
        title='Portfolio Hierarchy'
    )
    fig_sunburst.update_layout(height=500)
    
    # 4. Simple Returns Distribution
    fig_dist = px.histogram(
        df,
        x='Returns_Percentage',
        nbins=30,
        title='Returns Distribution Analysis',
        labels={'Returns_Percentage': 'Returns (%)', 'count': 'Frequency'},
        color_discrete_sequence=['rgba(102, 126, 234, 0.7)']
    )
    fig_dist.update_layout(
        height=400,
        showlegend=False,
        bargap=0.1
    )
    fig_dist.add_vline(x=df['Returns_Percentage'].mean(), 
                      line_dash="dash", 
                      line_color="red",
                      annotation_text=f"Avg: {df['Returns_Percentage'].mean():.1f}%")
    
    # 5. Waterfall Chart for P&L
    top_10 = df.nlargest(10, 'Profit_Loss_Cr')
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="P&L",
        orientation="v",
        x=top_10['Stock'].tolist() + ['Total'],
        y=top_10['Profit_Loss_Cr'].tolist() + [top_10['Profit_Loss_Cr'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgba(0, 200, 81, 0.7)"}},
        decreasing={"marker": {"color": "rgba(255, 68, 68, 0.7)"}},
        totals={"marker": {"color": "rgba(102, 126, 234, 0.7)"}}
    ))
    
    fig_waterfall.update_layout(
        title="Top 10 Stocks - P&L Waterfall",
        height=400
    )
    
    return fig_treemap, fig_scatter, fig_sunburst, fig_dist, fig_waterfall

def create_performance_dashboard(df):
    """Create performance analysis dashboard"""
    
    # Account Performance Comparison
    account_summary = df.groupby('Account').agg({
        'Buy_Value_Cr': 'sum',
        'Current_Value_Cr': 'sum',
        'Profit_Loss_Cr': 'sum',
        'Returns_Percentage': 'mean'
    }).round(2)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Account Holdings', 'Average Returns by Account', 
                       'P&L by Account', 'Investment vs Current Value'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=account_summary.index, 
               values=account_summary['Current_Value_Cr'],
               hole=0.4),
        row=1, col=1
    )
    
    # Bar chart - Returns
    fig.add_trace(
        go.Bar(x=account_summary.index, 
               y=account_summary['Returns_Percentage'],
               marker_color=['#667eea', '#764ba2', '#f50057']),
        row=1, col=2
    )
    
    # Bar chart - P&L
    colors = ['green' if x > 0 else 'red' for x in account_summary['Profit_Loss_Cr']]
    fig.add_trace(
        go.Bar(x=account_summary.index, 
               y=account_summary['Profit_Loss_Cr'],
               marker_color=colors),
        row=2, col=1
    )
    
    # Scatter - Investment vs Current
    fig.add_trace(
        go.Scatter(x=account_summary['Buy_Value_Cr'], 
                   y=account_summary['Current_Value_Cr'],
                   mode='markers+text',
                   text=account_summary.index,
                   textposition="top center",
                   marker=dict(size=20, color=['#667eea', '#764ba2', '#f50057'])),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    return fig

def format_dataframe_display(df):
    """Format dataframe for display without matplotlib dependency"""
    # Create a copy for display
    display_df = df.copy()
    
    # Round numeric columns
    numeric_cols = ['Buy_Value_Cr', 'Current_Value_Cr', 'Profit_Loss_Cr', 'Returns_Percentage']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    return display_df

# ======================== MAIN APPLICATION ========================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💎 Portfolio Analytics Dashboard</h1>
        <p>Advanced Investment Analysis & Performance Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 📊 Dashboard Controls")
        
        # File Upload Section
        with st.expander("📁 Data Upload", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Portfolio CSV",
                type=['csv'],
                help="Upload your portfolio data file"
            )
            
            if uploaded_file:
                st.success("✅ File uploaded successfully!")
            else:
                st.info("📌 Using sample data for demonstration")
        
        # Load Data
        if uploaded_file:
            with open("temp_portfolio.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            df, metadata = load_and_process_data("temp_portfolio.csv")
            import os
            os.remove("temp_portfolio.csv")
        else:
            df = create_sample_data()
            metadata = {}
    
    # Sidebar Filters
    with st.sidebar:
        st.markdown("## 🎯 Smart Filters")
        
        # Quick Filter Buttons
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        
        filter_type = None
        with col1:
            if st.button("🟢 Winners", use_container_width=True):
                filter_type = 'winners'
            if st.button("🏆 Top 10", use_container_width=True):
                filter_type = 'top10'
        
        with col2:
            if st.button("🔴 Losers", use_container_width=True):
                filter_type = 'losers'
            if st.button("💰 High Value", use_container_width=True):
                filter_type = 'highvalue'
        
        # Apply quick filters
        df_filtered = df.copy()
        if filter_type == 'winners':
            df_filtered = df_filtered[df_filtered['Returns_Percentage'] > 0]
        elif filter_type == 'losers':
            df_filtered = df_filtered[df_filtered['Returns_Percentage'] < 0]
        elif filter_type == 'top10':
            df_filtered = df_filtered.nlargest(10, 'Returns_Percentage')
        elif filter_type == 'highvalue':
            df_filtered = df_filtered.nlargest(10, 'Current_Value_Cr')
        
        st.markdown("---")
        
        # Advanced Filters
        st.markdown("### 🔍 Advanced Filters")
        
        # Stock Search
        all_stocks = sorted(df['Stock'].unique())
        selected_stocks = st.multiselect(
            "🔎 Search Stocks",
            options=all_stocks,
            default=all_stocks,
            placeholder="Type to search..."
        )
        
        # Account Filter
        selected_accounts = st.multiselect(
            "🏦 Select Accounts",
            options=df['Account'].unique(),
            default=df['Account'].unique()
        )
        
        # Performance Range
        st.markdown("#### 📊 Performance Range")
        returns_min, returns_max = st.slider(
            "Returns % Range",
            min_value=float(df['Returns_Percentage'].min()),
            max_value=float(df['Returns_Percentage'].max()),
            value=(float(df['Returns_Percentage'].min()), float(df['Returns_Percentage'].max())),
            step=1.0
        )
        
        # Investment Range
        st.markdown("#### 💵 Investment Range")
        inv_min, inv_max = st.slider(
            "Investment (₹ Cr)",
            min_value=0.0,
            max_value=float(df['Buy_Value_Cr'].max()),
            value=(0.0, float(df['Buy_Value_Cr'].max())),
            step=10.0
        )
        
        # Apply Filters
        df_filtered = df_filtered[
            (df_filtered['Stock'].isin(selected_stocks)) &
            (df_filtered['Account'].isin(selected_accounts)) &
            (df_filtered['Returns_Percentage'] >= returns_min) &
            (df_filtered['Returns_Percentage'] <= returns_max) &
            (df_filtered['Buy_Value_Cr'] >= inv_min) &
            (df_filtered['Buy_Value_Cr'] <= inv_max)
        ]
        
        st.markdown("---")
        st.markdown(f"**📌 Showing:** {len(df_filtered)} stocks")
    
    # Main Content Area
    if df_filtered.empty:
        st.warning("⚠️ No data matches your filters. Please adjust the filter criteria.")
        return
    
    # Metrics Row
    create_modern_metrics(df_filtered)
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Advanced Analytics", 
        "💼 Portfolio Details",
        "🎯 Performance Analysis",
        "📋 Data Table"
    ])
    
    with tab1:
        st.markdown("### 🌟 Portfolio Overview")
        
        # Get charts
        fig_treemap, fig_scatter, fig_sunburst, fig_dist, fig_waterfall = create_advanced_charts(df_filtered)
        
        # Display Treemap
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Two columns for smaller charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Advanced Portfolio Analytics")
        
        # Sunburst and Waterfall
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("---")
        
        # Top and Bottom Performers
        st.markdown("### 🏆 Performance Leaders & Laggards")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Top 5 Performers")
            top_5 = df_filtered.nlargest(5, 'Returns_Percentage')
            fig_top = px.bar(
                top_5, 
                x='Returns_Percentage', 
                y='Stock',
                orientation='h',
                color='Returns_Percentage',
                color_continuous_scale='Greens',
                text='Returns_Percentage'
            )
            fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_top.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Bottom 5 Performers")
            bottom_5 = df_filtered.nsmallest(5, 'Returns_Percentage')
            fig_bottom = px.bar(
                bottom_5, 
                x='Returns_Percentage', 
                y='Stock',
                orientation='h',
                color='Returns_Percentage',
                color_continuous_scale='Reds_r',
                text='Returns_Percentage'
            )
            fig_bottom.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bottom.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bottom, use_container_width=True)
    
    with tab3:
        st.markdown("### 💼 Detailed Portfolio Holdings")
        
        # Summary Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best = df_filtered.nlargest(1, 'Returns_Percentage').iloc[0]
            st.markdown(f"""
            <div class="info-box">
                <h4>📈 Best Performer</h4>
                <h3>{best['Stock']}</h3>
                <p class="positive-value">+{best['Returns_Percentage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            worst = df_filtered.nsmallest(1, 'Returns_Percentage').iloc[0]
            st.markdown(f"""
            <div class="warning-box">
                <h4>📉 Worst Performer</h4>
                <h3>{worst['Stock']}</h3>
                <p class="negative-value">{worst['Returns_Percentage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            largest = df_filtered.nlargest(1, 'Current_Value_Cr').iloc[0]
            st.markdown(f"""
            <div class="success-box">
                <h4>💰 Largest Holding</h4>
                <h3>{largest['Stock']}</h3>
                <p>₹{largest['Current_Value_Cr']:.2f} Cr</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_return = df_filtered['Returns_Percentage'].mean()
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Avg Return</h4>
                <h3 class="{'positive-value' if avg_return > 0 else 'negative-value'}">{'+' if avg_return > 0 else ''}{avg_return:.2f}%</h3>
                <p>Portfolio Average</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top Holdings
        st.markdown("### 🏆 Top 10 Holdings by Value")
        top_10 = df_filtered.nlargest(10, 'Current_Value_Cr')
        
        for _, row in top_10.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{row['Stock']}** ({row['Account']})")
            with col2:
                st.markdown(f"₹{row['Current_Value_Cr']:.2f} Cr")
            with col3:
                st.markdown(f"₹{row['Buy_Value_Cr']:.2f} Cr")
            with col4:
                color = "🟢" if row['Returns_Percentage'] > 0 else "🔴"
                st.markdown(f"{color} {row['Returns_Percentage']:.2f}%")
            with col5:
                if row['Returns_Percentage'] > 0:
                    st.markdown('<span class="profit-badge">Profit</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="loss-badge">Loss</span>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 🎯 Performance Analysis Dashboard")
        
        # Account Performance
        fig_perf = create_performance_dashboard(df_filtered)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("---")
        
        # Performance Metrics by Account
        st.markdown("### 📊 Account-wise Performance Metrics")
        
        account_metrics = df_filtered.groupby('Account').agg({
            'Buy_Value_Cr': 'sum',
            'Current_Value_Cr': 'sum',
            'Profit_Loss_Cr': 'sum',
            'Stock': 'count'
        }).round(2)
        
        account_metrics['Returns %'] = ((account_metrics['Current_Value_Cr'] - account_metrics['Buy_Value_Cr']) / 
                                        account_metrics['Buy_Value_Cr'] * 100).round(2)
        account_metrics.columns = ['Investment (₹Cr)', 'Current Value (₹Cr)', 'P&L (₹Cr)', 'Stocks', 'Returns (%)']
        
        # Display without background_gradient
        st.dataframe(account_metrics, use_container_width=True)
    
    with tab5:
        st.markdown("### 📋 Complete Portfolio Data")
        
        # Prepare display dataframe
        display_df = format_dataframe_display(df_filtered)
        display_df = display_df.sort_values('Current_Value_Cr', ascending=False)
        
        # Select and rename columns
        columns_to_display = ['Stock', 'Account', 'Quantity', 'Buy_Price', 'Current_Price', 
                             'Buy_Value_Cr', 'Current_Value_Cr', 'Profit_Loss_Cr', 'Returns_Percentage']
        
        # Filter columns that exist
        available_columns = [col for col in columns_to_display if col in display_df.columns]
        display_df = display_df[available_columns]
        
        # Rename columns for display
        column_names = {
            'Stock': 'Stock',
            'Account': 'Account',
            'Quantity': 'Quantity',
            'Buy_Price': 'Buy Price (₹)',
            'Current_Price': 'Current Price (₹)',
            'Buy_Value_Cr': 'Investment (₹Cr)',
            'Current_Value_Cr': 'Current Value (₹Cr)',
            'Profit_Loss_Cr': 'P&L (₹Cr)',
            'Returns_Percentage': 'Returns (%)'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # Display the dataframe
        st.dataframe(display_df, use_container_width=True, height=600)
        
        # Download Options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create summary report
            report = f"""Portfolio Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}

SUMMARY METRICS:
- Total Investment: ₹{df_filtered['Buy_Value_Cr'].sum():.2f} Cr
- Current Value: ₹{df_filtered['Current_Value_Cr'].sum():.2f} Cr
- Total P&L: ₹{df_filtered['Profit_Loss_Cr'].sum():.2f} Cr
- Overall Returns: {(df_filtered['Profit_Loss_Cr'].sum()/df_filtered['Buy_Value_Cr'].sum()*100):.2f}%

HOLDINGS:
- Total Stocks: {len(df_filtered)}
- Profitable: {len(df_filtered[df_filtered['Returns_Percentage'] > 0])}
- Loss-making: {len(df_filtered[df_filtered['Returns_Percentage'] < 0])}

TOP PERFORMERS:
{df_filtered.nlargest(5, 'Returns_Percentage')[['Stock', 'Returns_Percentage']].to_string()}

BOTTOM PERFORMERS:
{df_filtered.nsmallest(5, 'Returns_Percentage')[['Stock', 'Returns_Percentage']].to_string()}
            """
            
            st.download_button(
                "📄 Download Report",
                data=report,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>💎 Advanced Portfolio Analytics Dashboard v2.0</p>
        <p>Built with Streamlit • Plotly • Python</p>
        <p>© 2024 Portfolio Management System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
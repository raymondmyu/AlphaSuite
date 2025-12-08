import base64
from datetime import datetime
import os
import pandas as pd
import streamlit as st
import traceback

from tools.canslim_analysis_tool import CanslimReportGenerator
from tools.seasonality_analysis_tool import SeasonalityAnalysisTool
from tools.yfinance_tool import is_ticker_active
from load_cfg import DEMO_MODE, LLM_PROVIDER, get_llm

st.set_page_config(page_title="Stock Report Generator", layout="wide")

st.title("📊 Stock Report Generator")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** Live report generation is disabled. Sample reports will be shown instead.",
        icon="🔒"
    )

st.markdown(f"""
Enter a stock ticker to generate a report. Choose from:
- **CANSLIM Analysis**: A detailed evaluation based on William O'Neil's CANSLIM investing principles.
- **Comprehensive Report**: View a sample report. *(Note: This feature is for demonstration only. Contact the author for details on the full version.)*
- **Seasonality Analysis**: A statistical review of historical performance patterns tied to calendar events.

The currently configured LLM provider is: **{LLM_PROVIDER.upper()}**
""")

# --- State Management to preserve the report and ticker ---
if 'report' not in st.session_state:
    st.session_state.report = ""
if 'ticker' not in st.session_state:
    st.session_state.ticker = "META"  # Default ticker
if 'report_bytes' not in st.session_state:
    st.session_state.report_bytes = None
if 'seasonality_results' not in st.session_state:
    st.session_state.seasonality_results = None

def clear_report_state():
    """Clears the generated report and its bytes from the session state."""
    st.session_state.report = ""
    st.session_state.report_bytes = None
    st.session_state.seasonality_results = None

def generate_and_display_report(ticker_symbol, report_type):
    """Generates and displays the selected report, or a sample for the comprehensive one."""
    report_file = None
    try:
        if report_type == "Comprehensive Report":
            st.info("Displaying a sample Comprehensive Report. This feature is for demonstration purposes.")
            # Construct path to the sample report in the 'samples' directory at the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_file = os.path.join(project_root, "samples", "META_comprehensive_report_20250824.pdf")

            if not os.path.exists(report_file):
                st.error("Sample report file not found. Please ensure 'samples/META_comprehensive_report_20250824.pdf' exists.")
                clear_report_state()
                return

        elif report_type == "CANSLIM Analysis":
            if DEMO_MODE:
                st.info("Displaying a sample CANSLIM Analysis Report. This feature is for demonstration purposes.")
                # Construct path to the sample report in the 'samples' directory at the project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                report_file = os.path.join(project_root, "samples", "META_canslim_report_20250824.pdf")

                if not os.path.exists(report_file):
                    st.error("Sample report file not found. Please ensure 'samples/META_canslim_report_20250824.pdf' exists.")
                    clear_report_state()
                    return
            else: # Original logic for non-demo mode
                if not is_ticker_active(ticker_symbol):
                    st.error(f"Ticker {ticker_symbol} is not actively traded or could not be found.")
                    clear_report_state()
                    return

                spinner_text = f"Generating {report_type.lower()} for {ticker_symbol}... This may take some time."
                with st.spinner(spinner_text):
                    llm = get_llm()
                    generator = CanslimReportGenerator(llm)
                    report_file = generator.generate_report(ticker_symbol)
        else:
            st.error("Invalid report type selected.")
            clear_report_state()
            return

        if report_file:
            st.session_state.report = report_file
            with open(report_file, "rb") as f:
                st.session_state.report_bytes = f.read()
            st.success(f"Report '{os.path.basename(st.session_state.report)}' loaded successfully.")

    except Exception as e:
        st.error(f"An error occurred while generating the report for {ticker_symbol}: {e}")
        clear_report_state()
        traceback.print_exc()

def generate_seasonality_analysis(ticker_symbol, start_date, end_date):
    """Generates and stores seasonality analysis results."""
    if not is_ticker_active(ticker_symbol):
        st.error(f"Ticker {ticker_symbol} is not actively traded or could not be found.")
        st.session_state.seasonality_results = None
        return

    try:
        spinner_text = f"Generating seasonality analysis for {ticker_symbol}..."
        with st.spinner(spinner_text):
            tool = SeasonalityAnalysisTool(
                ticker=ticker_symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            results = tool.run_full_analysis()
            st.session_state.seasonality_results = results
            st.success(f"Seasonality analysis for {ticker_symbol} generated successfully.")
    except Exception as e:
        st.error(f"An error occurred during seasonality analysis for {ticker_symbol}: {e}")
        st.session_state.seasonality_results = None
        traceback.print_exc()

# --- UI Components ---
ticker_input = st.text_input(
    "Enter Stock Ticker:", 
    value=st.session_state.ticker, 
    on_change=clear_report_state
).upper()
st.session_state.ticker = ticker_input

tab1, tab2 = st.tabs(["📄 Reports", "📅 Seasonality Analysis"])

with tab1:
    st.header("Generate PDF Report")
    report_type = st.radio(
        "Select Report Type:",
        ("Comprehensive Report", "CANSLIM Analysis"),
        key="report_type_selection",
        on_change=clear_report_state
    )

    if st.button("Generate Report", key="generate_pdf_report"):
        generate_and_display_report(st.session_state.ticker, report_type)

    # --- Display & Download Section for PDF Reports ---
    if st.session_state.report_bytes:
        st.download_button(
            label="Download Report",
            data=st.session_state.report_bytes,
            file_name=os.path.basename(st.session_state.report),
            mime="application/pdf"
        )
        st.subheader("Generated Report Preview (PDF)")
        try:
            base64_pdf = base64.b64encode(st.session_state.report_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not display PDF preview: {e}")

with tab2:
    st.header("Perform Seasonality Analysis")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=datetime(2000, 1, 1), key="seasonality_start", on_change=clear_report_state)
    end_date = col2.date_input("End Date", value=datetime.now(), key="seasonality_end", on_change=clear_report_state)

    if st.button("Generate Seasonality Analysis", key="generate_seasonality", disabled=DEMO_MODE):
        generate_seasonality_analysis(st.session_state.ticker, start_date, end_date)

    if st.session_state.seasonality_results:
        results = st.session_state.seasonality_results
        st.subheader(f"Seasonality Analysis for {st.session_state.ticker}")

        # --- Display Charts and Data ---
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.markdown(f"### {name.replace('_', ' ').title()}")

                # Add specific note for Turn of the Month
                if name == "turn_of_month_effect":
                    st.caption("> **Note:** 'Turn of Month' days are defined as the period from the 28th of a given month through the 4th day of the next month. The statistics show the average daily performance during this period compared to all other days.")
                
                # Add specific note for Special Patterns
                if name == "special_patterns":
                    with st.expander("View Notes on Patterns"):
                        st.markdown("- **Santa Claus Rally:** Stocks often experience a rally from the last week of December into the first two trading days of January.")
                        st.markdown("- **January (After Losing Year):** Historically, small-cap stocks that were down in the previous year have often seen buying pressure in January, partly due to tax-loss selling in December.")
                        st.markdown("- **Sell in May:** This analysis compares the total return of the six-month period from November to April against the period from May to October.")
                        
                # Display bar chart for sector comparison (spread only)
                if name == "sector_comparison":
                    # The dataframe contains the stock's return, the benchmark's return, and the spread.
                    # Plotting all of them gives a better comparative view.
                    if not df.empty:
                        st.bar_chart(df)
                    else:
                        st.info("Sector comparison not available. No specific sector ETF found for this ticker, or data could not be loaded for the benchmark.")
                # Display bar chart for other analyses
                elif 'avg_return' in df.columns and 'median_return' in df.columns:
                    st.bar_chart(df[['avg_return', 'median_return']])

                # Put the detailed dataframe inside an expander
                with st.expander("View Data Table"):
                    # Rename for clarity before displaying the table
                    df_display = df.rename(columns={'std_dev_return': 'Std Dev (%)', 'periods_analyzed': 'Periods'})
                    st.dataframe(df_display)

# -*- coding: utf-8 -*-
"""
Streamlit application for interactive logistic regression modeling based on
Frank Harrell's "Regression Modeling Strategies".

This tool allows users to upload data, build a logistic regression model with
restricted cubic splines, perform rigorous bootstrap validation, and use the
model for prediction via an interactive nomogram-like calculator.

Author: Your Name (as Senior Biostatistician/Python Developer)
Date: 2023-10-27
Version: 2.1 (Definitive: Hardened calibration slope for intercept-only models)
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices, build_design_matrices, stateful_transform
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional
import re

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="RMS Predictive Modeler",
    page_icon="⚕️"
)

# --- Type Hinting ---
DataFrame = pd.DataFrame
ModelResults = Any  # statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
ValidationResults = Dict[str, Any]

# --- Patsy rcs() function implementation ---
@stateful_transform
class rcs:
    """
    Restricted Cubic Spline stateful transform for patsy.
    """
    def __init__(self, knots=4):
        self.knots = knots

    def memorize_chunk(self, x, knots):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) > 0:
            self.knots = np.quantile(x_clean, np.linspace(0, 1, self.knots))
        else:
            self.knots = []

    def memorize_finish(self):
        pass

    def transform(self, x, knots):
        if not hasattr(self, 'knots') or len(self.knots) == 0:
            return np.zeros((len(x), 1))
        k = len(self.knots)
        X = np.zeros((len(x), k - 1))
        X[:, 0] = x
        def d(knot_idx):
            numerator = (np.maximum(x - self.knots[knot_idx], 0)**3 - 
                         np.maximum(x - self.knots[k - 1], 0)**3)
            denominator = self.knots[k - 1] - self.knots[0]
            if denominator > 1e-8:
                return numerator / denominator
            else:
                return np.zeros_like(x)
        for j in range(k - 2):
            X[:, j + 1] = d(j) - d(k - 2)
        return X

# --- Core Functions ---
@st.cache_data
def load_data(uploaded_file: Any) -> Optional[DataFrame]:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    return None

def preprocess_data(df: DataFrame, missing_handler: str) -> DataFrame:
    df_processed = df.copy()
    if missing_handler == "Remove Rows with Missing Data":
        df_processed.dropna(inplace=True)
    elif missing_handler == "Mean/Mode Imputation":
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    return df_processed

def create_formula(dependent_var: str, independent_vars: List[str], spline_vars: List[str], knots: int = 4) -> str:
    if not independent_vars:
        return f"{dependent_var} ~ 1"
    terms = [f"rcs({var}, {knots})" if var in spline_vars else var for var in independent_vars]
    return f"{dependent_var} ~ {' + '.join(terms)}"

@st.cache_resource
def fit_model(_df: DataFrame, formula: str) -> Optional[Dict[str, Any]]:
    try:
        y, X = dmatrices(formula, data=_df, return_type='dataframe')
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0)
        # Store the design info for later use in predictions
        design_info = X.design_info
        return {"model": model, "formula": formula, "y": y, "X": X, "design_info": design_info}
    except Exception as e:
        st.error(f"Error fitting model: {e}")
        st.warning("This can happen if a variable has no variation after preprocessing, or if the model is not identifiable.")
        return None

@st.cache_data
def run_bootstrap_validation(formula: str, _df: DataFrame, n_bootstraps: int = 200) -> Optional[ValidationResults]:
    original_y, original_X = dmatrices(formula, data=_df, return_type='dataframe')
    original_model = sm.GLM(original_y, original_X, family=sm.families.Binomial()).fit(disp=0)
    original_preds = original_model.predict(original_X)
    original_linpred = original_X @ original_model.params
    apparent_c_index = roc_auc_score(original_y, original_preds)
    apparent_brier = brier_score_loss(original_y, original_preds)
    apparent_cal_large = original_preds.mean() / original_y.iloc[:, 0].mean()
    cal_slope_model_app = sm.GLM(original_y, sm.add_constant(original_linpred), family=sm.families.Binomial()).fit(method='bfgs')
    apparent_slope = cal_slope_model_app.params.iloc[1] if len(cal_slope_model_app.params) > 1 else 0.0
    optimisms = {'c_index': [], 'brier': [], 'cal_large': [], 'slope': []}
    boot_progress = st.progress(0, text="Running bootstrap validation...")
    all_boot_preds = np.zeros((n_bootstraps, len(_df)))
    failed_bootstraps = 0
    for i in range(n_bootstraps):
        try:
            boot_sample = _df.sample(n=len(_df), replace=True)
            y_boot, X_boot = dmatrices(formula, data=boot_sample, return_type='dataframe')
            boot_model = sm.GLM(y_boot, X_boot, family=sm.families.Binomial()).fit(disp=0)
            preds_boot = boot_model.predict(X_boot)
            linpred_boot = X_boot @ boot_model.params
            cal_slope_model_boot = sm.GLM(y_boot, sm.add_constant(linpred_boot), family=sm.families.Binomial()).fit(method='bfgs')
            slope_boot = cal_slope_model_boot.params.iloc[1] if len(cal_slope_model_boot.params) > 1 else 0.0
            c_boot = roc_auc_score(y_boot, preds_boot)
            brier_boot = brier_score_loss(y_boot, preds_boot)
            cal_large_boot = preds_boot.mean() / y_boot.iloc[:, 0].mean()
            preds_orig = boot_model.predict(original_X)
            linpred_orig = original_X @ boot_model.params
            cal_slope_model_orig = sm.GLM(original_y, sm.add_constant(linpred_orig), family=sm.families.Binomial()).fit(method='bfgs')
            slope_orig = cal_slope_model_orig.params.iloc[1] if len(cal_slope_model_orig.params) > 1 else 0.0
            c_orig = roc_auc_score(original_y, preds_orig)
            brier_orig = brier_score_loss(original_y, preds_orig)
            cal_large_orig = preds_orig.mean() / original_y.iloc[:, 0].mean()
            all_boot_preds[i, :] = preds_orig
            optimisms['c_index'].append(c_boot - c_orig)
            optimisms['brier'].append(brier_boot - brier_orig)
            optimisms['cal_large'].append(cal_large_boot - cal_large_orig)
            optimisms['slope'].append(slope_boot - slope_orig)
            boot_progress.progress((i + 1) / n_bootstraps, text=f"Bootstrap resample {i+1}/{n_bootstraps}")
        except Exception:
            failed_bootstraps += 1
            continue
    boot_progress.empty()
    if failed_bootstraps > 0:
        st.warning(f"{failed_bootstraps} of {n_bootstraps} bootstrap iterations failed and were skipped. This can happen with small datasets or unstable models.")
    if not optimisms['c_index']:
        st.error("All bootstrap iterations failed. Validation results are not available.")
        return None
    mean_optimism_c = np.mean(optimisms['c_index'])
    mean_optimism_brier = np.mean(optimisms['brier'])
    mean_optimism_cal = np.mean(optimisms['cal_large'])
    mean_optimism_slope = np.mean(optimisms['slope'])
    corrected_c = apparent_c_index - mean_optimism_c
    corrected_brier = apparent_brier + mean_optimism_brier
    corrected_cal_large = apparent_cal_large - mean_optimism_cal
    corrected_slope = apparent_slope - mean_optimism_slope
    avg_boot_preds = np.mean(all_boot_preds, axis=0)
    prob_true, prob_pred = calibration_curve(original_y, avg_boot_preds, n_bins=10, strategy='quantile')
    return {"apparent": {"C-Index (AUC)": apparent_c_index, "Brier Score": apparent_brier, "E/O Index": apparent_cal_large, "Calibration Slope": apparent_slope}, "corrected": {"C-Index (AUC)": corrected_c, "Brier Score": corrected_brier, "E/O Index": corrected_cal_large, "Calibration Slope": corrected_slope}, "calibration_curve": (prob_true, prob_pred)}

# --- UI Display Functions ---
def display_model_results(model_results: Dict[str, Any]):
    st.subheader("Model Formula")
    st.code(model_results['formula'], language='r')
    st.subheader("Coefficients and 95% Confidence Intervals")
    conf_int = model_results['model'].conf_int()
    params = model_results['model'].params
    coef_df = pd.DataFrame({'Coefficient': params, 'Lower CI': conf_int[0], 'Upper CI': conf_int[1]}).reset_index().rename(columns={'index': 'Term'})
    coef_df = coef_df[coef_df['Term'] != 'Intercept']
    if not coef_df.empty:
        fig = px.scatter(coef_df, x='Coefficient', y='Term', error_x=coef_df['Upper CI'] - coef_df['Coefficient'], error_x_minus=coef_df['Coefficient'] - coef_df['Lower CI'], title="Model Coefficients (Log-Odds Scale)")
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The model only contains an intercept. There are no coefficients to plot.")
    with st.expander("Full Model Summary (for experts)"):
        st.code(str(model_results['model'].summary()), language=None)

def display_validation_results(validation_results: ValidationResults):
    st.subheader("Optimism-Corrected Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Performance Metrics**")
        perf_df = pd.DataFrame({'Apparent': validation_results['apparent'], 'Optimism-Corrected': validation_results['corrected']}).round(3)
        st.dataframe(perf_df, use_container_width=True)
        st.caption("- **C-Index (AUC):** Discrimination. 0.5=random, 1.0=perfect. Higher is better.\n- **Brier Score:** Overall accuracy. 0=perfect. Lower is better.\n- **E/O Index:** Calibration-in-the-large. Ratio of Expected/Observed events. 1.0=perfect.\n- **Calibration Slope:** Measures prediction scaling. 1.0=perfect. <1 suggests overfitting (predictions are too extreme).")
    with col2:
        st.write("**Bootstrap Calibration Curve**")
        prob_true, prob_pred = validation_results['calibration_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibration Curve', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(color='black', dash='dash')))
        fig.update_layout(title="Model Reliability", xaxis_title="Predicted Probability", yaxis_title="Observed Proportion", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), legend=dict(x=0.05, y=0.95))
        st.plotly_chart(fig, use_container_width=True)

def display_diagnostics_plots(model_results: Dict[str, Any]):
    st.header("Model Diagnostics")
    model = model_results['model']
    X = model_results['X']
    y = model_results['y'].iloc[:, 0]
    st.subheader("Binned Residuals Plot")
    st.markdown("**Interpretation:** This plot helps check if the model is well-calibrated across the range of predictions. We group data points into bins based on their predicted probability and plot the average actual outcome minus the average predicted outcome (the residual).\n- **What to look for:** The points should scatter randomly around the horizontal line at zero.\n- **Red flags:** A systematic pattern (like a U-shape or a slope) suggests the model is not capturing the relationship correctly. Points with error bars not crossing the zero line indicate regions where the model's predictions are systematically wrong.")
    preds = model.predict(X)
    residuals = y - preds
    binned_df = pd.DataFrame({'preds': preds, 'resid': residuals})
    if binned_df['preds'].nunique() > 1:
        try:
            binned_df['bin'] = pd.qcut(binned_df['preds'], q=10, duplicates='drop')
            binned_summary = binned_df.groupby('bin').agg(mean_pred=('preds', 'mean'), mean_resid=('resid', 'mean'), n=('resid', 'size')).reset_index()
            binned_summary['se'] = np.sqrt(binned_summary['mean_pred'] * (1 - binned_summary['mean_pred']) / binned_summary['n'])
            fig_binned = go.Figure()
            fig_binned.add_trace(go.Scatter(x=binned_summary['mean_pred'], y=binned_summary['mean_resid'], mode='markers', name='Average Residual', error_y=dict(type='data', array=1.96 * binned_summary['se'], visible=True)))
            fig_binned.add_hline(y=0, line_width=2, line_dash="dash", line_color="grey")
            fig_binned.update_layout(title="Binned Residuals vs. Predicted Probability", xaxis_title="Average Predicted Probability in Bin", yaxis_title="Average Residual (Observed - Predicted)")
            st.plotly_chart(fig_binned, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate binned residuals plot. Error: {e}")
    else:
        st.info("Binned residuals plot cannot be generated for an intercept-only model.")
    st.subheader("Influence and Leverage Plot")
    st.markdown("**Interpretation:** This plot helps identify influential data points that might be overly affecting the model's coefficients.\n- **Leverage (X-axis):** Measures how unusual a point's predictor values are. Points far to the right are outliers in the predictor space. A common threshold for high leverage is marked by the vertical dashed line.\n- **Pearson Residuals (Y-axis):** Measures how poorly the model fits a point (standardized for GLMs). Points far from zero (e.g., > |2|) are potential outliers.\n- **Cook's Distance (Bubble Size):** Combines leverage and residual size to measure a point's total influence.\n- **What to look for:** Points with large bubbles, especially those with high leverage and large residuals, are highly influential. You may want to investigate these data points (identified by their index number) to ensure they are not data entry errors.")
    try:
        if len(model.params) > 1:
            influence = model.get_influence()
            influence_df = pd.DataFrame({'leverage': influence.hat_matrix_diag, 'pearson_resid': model.resid_pearson, 'cooks_d': influence.cooks_distance[0]}).reset_index().rename(columns={'index': 'Observation Index'})
            fig_influence = px.scatter(influence_df, x='leverage', y='pearson_resid', size='cooks_d', hover_name='Observation Index', hover_data={'leverage': ':.3f', 'pearson_resid': ':.3f', 'cooks_d': ':.3f'}, size_max=30, title="Leverage vs. Pearson Residuals")
            fig_influence.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
            fig_influence.add_hline(y=2, line_width=1, line_dash="dot", line_color="silver")
            fig_influence.add_hline(y=-2, line_width=1, line_dash="dot", line_color="silver")
            k = len(model.params)
            n = len(y)
            leverage_threshold = 2 * k / n
            fig_influence.add_vline(x=leverage_threshold, line_width=1, line_dash="dash", line_color="red")
            fig_influence.update_layout(xaxis_title="Leverage (Hat-value)", yaxis_title="Pearson Residuals")
            st.plotly_chart(fig_influence, use_container_width=True)
        else:
            st.info("Influence plot is not applicable for an intercept-only model.")
    except Exception as e:
        st.warning(f"Could not generate influence plot. Error: {e}")

def display_prediction_calculator(model_results: Dict[str, Any]):
    st.subheader("Interactive Prediction Calculator")
    st.markdown("Use the controls below to get a prediction for a new observation.")
    X = model_results['X']
    model = model_results['model']
    if len(model.params) == 1:
        st.info("The model is intercept-only. The prediction is the same for all inputs.")
        prob = model.predict(X.iloc[0:1])[0]
        st.metric(label="Predicted Probability", value=f"{prob:.1%}")
        return
    predictor_controls = {}
    processed_predictors = set()
    for col in X.columns:
        if col == 'Intercept': continue
        rcs_match = re.match(r"rcs\((.*),.*\)", col)
        if rcs_match:
            original_var = rcs_match.group(1).strip()
            if original_var in processed_predictors: continue
            min_val, max_val, mean_val = float(st.session_state.df_processed[original_var].min()), float(st.session_state.df_processed[original_var].max()), float(st.session_state.df_processed[original_var].mean())
            predictor_controls[original_var] = st.slider(f"**{original_var}**", min_val, max_val, mean_val)
            processed_predictors.add(original_var)
        elif '[' in col and ']' in col:
            original_var = col.split('[T.')[0].strip()
            if original_var in processed_predictors: continue
            options = [lvl.split('[T.')[1].strip(']') for lvl in X.columns if lvl.startswith(original_var)]
            base_level = list(set(st.session_state.df_processed[original_var].unique()) - set(options))[0]
            all_levels = [base_level] + options
            predictor_controls[original_var] = st.selectbox(f"**{original_var}**", options=all_levels, index=0)
            processed_predictors.add(original_var)
        else:
            if col in processed_predictors: continue
            min_val, max_val, mean_val = float(st.session_state.df_processed[col].min()), float(st.session_state.df_processed[col].max()), float(st.session_state.df_processed[col].mean())
            predictor_controls[col] = st.slider(f"**{col}**", min_val, max_val, mean_val)
            processed_predictors.add(col)
    if predictor_controls:
        new_data = pd.DataFrame([predictor_controls])
        try:
            design_info = model_results['design_info']
            new_X = build_design_matrices([design_info], new_data, return_type='dataframe')[0]
            prediction = model.get_prediction(new_X)
            pred_summary = prediction.summary_frame(alpha=0.05)
            prob, ci_lower, ci_upper = pred_summary['mean'].iloc[0], pred_summary['mean_ci_lower'].iloc[0], pred_summary['mean_ci_upper'].iloc[0]
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.metric(label="Predicted Probability", value=f"{prob:.1%}", help=f"95% Confidence Interval: {ci_lower:.1%} - {ci_upper:.1%}")
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, title={'text': "Probability (%)"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "royalblue"}}, domain={'x': [0, 1], 'y': [0, 1]}))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                st.write("**Predictor Contributions (Log-Odds)**")
                params = model.params
                contributions = params * new_X.iloc[0]
                contrib_data = {}
                for term, value in contributions.items():
                    rcs_match = re.match(r"rcs\((.*),.*\)", term)
                    if rcs_match:
                        original_var = rcs_match.group(1).strip()
                        contrib_data[original_var] = contrib_data.get(original_var, 0) + value
                    else:
                        contrib_data[term] = value
                contrib_df = pd.DataFrame(list(contrib_data.items()), columns=['Term', 'Contribution']).sort_values('Contribution', ascending=True)
                fig_contrib = px.bar(contrib_df, x='Contribution', y='Term', orientation='h', title="How Each Factor Influences the Outcome", labels={'Contribution': 'Change in Log-Odds'})
                fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                fig_contrib.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_contrib, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate prediction. Error: {e}")

# --- Main Application ---
def main():
    st.title("⚕️ RMS Predictive Modeler")
    st.markdown("A tool for building and validating logistic regression models based on the principles of **Frank Harrell's Regression Modeling Strategies**.")
    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader("Upload your data (CSV format)", type="csv")
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            st.success(f"Loaded `{uploaded_file.name}` with {df.shape[0]} rows and {df.shape[1]} columns.")
            st.header("2. Preprocessing")
            with st.expander("Advanced Options"):
                missing_handler = st.radio("Handle Missing Data", ("Mean/Mode Imputation", "Remove Rows with Missing Data"), help="Choose how to handle rows with missing values. Imputation is generally preferred to retain data.")
                st.session_state.missing_handler = missing_handler
            st.header("3. Model Specification")
            dep_var_options = [""] + list(df.columns)
            dependent_var = st.selectbox("Select Dependent Variable (Outcome)", dep_var_options, help="This must be a binary (0/1) or two-level categorical variable.")
            if dependent_var:
                if df[dependent_var].nunique() != 2:
                    st.error(f"Error: Column '{dependent_var}' is not binary. It has {df[dependent_var].nunique()} unique values.")
                    st.stop()
                if not pd.api.types.is_numeric_dtype(df[dependent_var]):
                    df[dependent_var] = pd.factorize(df[dependent_var])[0]
                    st.info(f"Categorical outcome '{dependent_var}' was automatically converted to 0/1.")
                indep_var_options = [col for col in df.columns if col != dependent_var]
                independent_vars = st.multiselect("Select Independent Variables (Predictors)", indep_var_options, default=indep_var_options[:min(len(indep_var_options), 3)])
                spline_vars = []
                if independent_vars:
                    numeric_candidates = [var for var in independent_vars if pd.api.types.is_numeric_dtype(df[var]) and df[var].nunique() > 5]
                    if numeric_candidates:
                        st.markdown("---")
                        spline_vars = st.multiselect("Apply Splines (RCS) to:", options=numeric_candidates, default=numeric_candidates, help="Select numeric variables to model with non-linear splines. Deselect to model them linearly.")
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    if not independent_vars:
                        st.warning("No independent variables selected. Running an intercept-only model.")
                    with st.spinner("Processing data and building model..."):
                        # Ensure dependent_var is always in the list for preprocessing
                        cols_to_process = [dependent_var] + independent_vars
                        df_processed = preprocess_data(df[list(set(cols_to_process))], st.session_state.missing_handler)
                        st.session_state.df_processed = df_processed
                        if len(df_processed) < 50:
                            st.warning(f"Warning: After preprocessing, only {len(df_processed)} rows remain. Model results may be unstable.")

                        formula = create_formula(dependent_var, independent_vars, spline_vars)
                        model_results = fit_model(df_processed, formula)
                        st.session_state.model_results = model_results
                        if model_results:
                            validation_results = run_bootstrap_validation(formula, df_processed)
                            st.session_state.validation_results = validation_results
                        else:
                            st.session_state.validation_results = None
                    st.success("Analysis complete! View results in the tabs.")
    if 'model_results' in st.session_state and st.session_state.model_results:
        model_results = st.session_state.model_results
        validation_results = st.session_state.get('validation_results')
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Results", "🛡️ Model Validation", "🔎 Model Diagnostics", "🧮 Prediction Calculator"])
        with tab1: display_model_results(model_results)
        with tab2:
            if validation_results: display_validation_results(validation_results)
            else: st.warning("Validation could not be completed.")
        with tab3: display_diagnostics_plots(model_results)
        with tab4: display_prediction_calculator(model_results)
    else:
        st.info("Welcome! Please upload a CSV file and select your variables in the sidebar to begin.")

if __name__ == "__main__":
    main()
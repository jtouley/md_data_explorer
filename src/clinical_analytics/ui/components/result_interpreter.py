"""
Result Interpreter Component

Provides plain-language interpretation of statistical results for clinicians.
"""

import streamlit as st


class ResultInterpreter:
    """
    Interprets statistical results in plain language for clinicians.

    Converts p-values, effect sizes, and statistical outputs into
    understandable clinical interpretations.
    """

    @staticmethod
    def interpret_p_value(p_value: float, alpha: float = 0.05) -> dict[str, any]:
        """
        Interpret a p-value in plain language.

        Args:
            p_value: P-value from statistical test
            alpha: Significance threshold (default 0.05)

        Returns:
            Dictionary with interpretation details
        """
        if p_value < 0.001:
            significance = "highly significant"
            emoji = "✅✅✅"
            interpretation = (
                "This result is **highly unlikely** to be due to chance (p<0.001). Very strong evidence of an effect."
            )
        elif p_value < 0.01:
            significance = "very significant"
            emoji = "✅✅"
            interpretation = (
                "This result is **very unlikely** to be due to chance (p<0.01). Strong evidence of an effect."
            )
        elif p_value < alpha:
            significance = "significant"
            emoji = "✅"
            interpretation = (
                f"This result is **unlikely** to be due to chance (p={p_value:.3f}). Statistically significant."
            )
        elif p_value < 0.10:
            significance = "marginally significant"
            emoji = "⚠️"
            interpretation = (
                f"This result shows a **trend** but doesn't reach traditional significance "
                f"(p={p_value:.3f}). Consider with caution."
            )
        else:
            significance = "not significant"
            emoji = "❌"
            interpretation = (
                f"This result **could easily be due to chance** (p={p_value:.3f}). No strong evidence of an effect."
            )

        return {
            "p_value": p_value,
            "significance": significance,
            "emoji": emoji,
            "interpretation": interpretation,
            "is_significant": p_value < alpha,
        }

    @staticmethod
    def interpret_odds_ratio(
        or_value: float, ci_lower: float, ci_upper: float, p_value: float, variable_name: str
    ) -> str:
        """
        Interpret an odds ratio in clinical terms.

        Args:
            or_value: Odds ratio point estimate
            ci_lower: Lower 95% CI
            ci_upper: Upper 95% CI
            p_value: P-value
            variable_name: Name of variable

        Returns:
            Plain-language interpretation
        """
        p_interp = ResultInterpreter.interpret_p_value(p_value)

        if not p_interp["is_significant"]:
            return f"""
**{variable_name}**: Not significantly associated with outcome (p={p_value:.3f})

The odds ratio is {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}),
but this could be due to chance.
"""

        if or_value > 1:
            if or_value >= 2:
                magnitude = "**strongly increases**"
            elif or_value >= 1.5:
                magnitude = "**moderately increases**"
            else:
                magnitude = "**slightly increases**"

            pct_increase = (or_value - 1) * 100

            interpretation = f"""
**{variable_name}**: {magnitude} the odds of the outcome {p_interp["emoji"]}

- **Odds Ratio**: {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: Having this characteristic increases the odds of the outcome by
  approximately **{pct_increase:.0f}%**
- **Statistical Significance**: {p_interp["interpretation"]}
"""

        elif or_value < 1:
            if or_value <= 0.5:
                magnitude = "**strongly decreases**"
            elif or_value <= 0.67:
                magnitude = "**moderately decreases**"
            else:
                magnitude = "**slightly decreases**"

            pct_decrease = (1 - or_value) * 100

            interpretation = f"""
**{variable_name}**: {magnitude} the odds of the outcome {p_interp["emoji"]}

- **Odds Ratio**: {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: Having this characteristic decreases the odds of the outcome by
  approximately **{pct_decrease:.0f}%**
- **Statistical Significance**: {p_interp["interpretation"]}
"""

        else:  # OR ≈ 1
            interpretation = f"""
**{variable_name}**: No meaningful association with outcome

- **Odds Ratio**: {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: This variable doesn't appear to affect the outcome
"""

        return interpretation

    @staticmethod
    def interpret_mean_difference(
        mean_diff: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        group1: str,
        group2: str,
        outcome_name: str,
        units: str | None = None,
    ) -> str:
        """
        Interpret a mean difference (t-test result).

        Args:
            mean_diff: Difference in means
            ci_lower: Lower 95% CI
            ci_upper: Upper 95% CI
            p_value: P-value
            group1: Name of first group
            group2: Name of second group
            outcome_name: Name of outcome variable
            units: Units of measurement (optional)

        Returns:
            Plain-language interpretation
        """
        p_interp = ResultInterpreter.interpret_p_value(p_value)
        units_str = f" {units}" if units else ""

        if not p_interp["is_significant"]:
            return f"""
**No significant difference** between {group1} and {group2} (p={p_value:.3f}) ❌

The difference in {outcome_name} is {mean_diff:.2f}{units_str}
(95% CI: {ci_lower:.2f} to {ci_upper:.2f}), but this could easily be due to chance.
"""

        direction = "higher" if mean_diff > 0 else "lower"
        abs_diff = abs(mean_diff)

        interpretation = f"""
**Significant difference** between groups {p_interp["emoji"]}

- **{group1}** has {direction} {outcome_name} than **{group2}**
- **Difference**: {abs_diff:.2f}{units_str} (95% CI: {abs(ci_lower):.2f} to {abs(ci_upper):.2f})
- **Statistical Significance**: {p_interp["interpretation"]}

**Clinical Interpretation**: On average, {group1} patients have {outcome_name} that is
{abs_diff:.2f}{units_str} {direction} than {group2} patients.
"""

        return interpretation

    @staticmethod
    def interpret_hazard_ratio(
        hr_value: float, ci_lower: float, ci_upper: float, p_value: float, variable_name: str
    ) -> str:
        """
        Interpret a hazard ratio in clinical terms.

        Args:
            hr_value: Hazard ratio point estimate
            ci_lower: Lower 95% CI
            ci_upper: Upper 95% CI
            p_value: P-value
            variable_name: Name of variable

        Returns:
            Plain-language interpretation
        """
        p_interp = ResultInterpreter.interpret_p_value(p_value)

        if not p_interp["is_significant"]:
            return f"""
**{variable_name}**: Not significantly associated with outcome (p={p_value:.3f})

The hazard ratio is {hr_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}),
but this could be due to chance.
"""

        if hr_value > 1:
            if hr_value >= 2:
                magnitude = "**strongly increases**"
            elif hr_value >= 1.5:
                magnitude = "**moderately increases**"
            else:
                magnitude = "**slightly increases**"

            pct_increase = (hr_value - 1) * 100

            interpretation = f"""
**{variable_name}**: {magnitude} the hazard (risk) of the event {p_interp["emoji"]}

- **Hazard Ratio**: {hr_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: Having this characteristic increases the hazard by approximately
  **{pct_increase:.0f}%** (faster time to event)
- **Statistical Significance**: {p_interp["interpretation"]}
"""

        elif hr_value < 1:
            if hr_value <= 0.5:
                magnitude = "**strongly decreases**"
            elif hr_value <= 0.67:
                magnitude = "**moderately decreases**"
            else:
                magnitude = "**slightly decreases**"

            pct_decrease = (1 - hr_value) * 100

            interpretation = f"""
**{variable_name}**: {magnitude} the hazard (risk) of the event {p_interp["emoji"]}

- **Hazard Ratio**: {hr_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: Having this characteristic decreases the hazard by approximately
  **{pct_decrease:.0f}%** (slower time to event)
- **Statistical Significance**: {p_interp["interpretation"]}
"""

        else:  # HR ≈ 1
            interpretation = f"""
**{variable_name}**: No meaningful association with event timing

- **Hazard Ratio**: {hr_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})
- **Interpretation**: This variable doesn't appear to affect time to event
"""

        return interpretation

    @staticmethod
    def interpret_correlation(correlation: float, p_value: float, var1: str, var2: str) -> str:
        """
        Interpret a correlation coefficient.

        Args:
            correlation: Correlation coefficient (-1 to 1)
            p_value: P-value
            var1: First variable name
            var2: Second variable name

        Returns:
            Plain-language interpretation
        """
        p_interp = ResultInterpreter.interpret_p_value(p_value)

        # Interpret magnitude
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "**strong**"
        elif abs_corr >= 0.5:
            strength = "**moderate**"
        elif abs_corr >= 0.3:
            strength = "**weak**"
        else:
            strength = "**very weak**"

        # Interpret direction
        if correlation > 0:
            direction = "**positive**"
            relationship = f"As {var1} increases, {var2} tends to increase"
        else:
            direction = "**negative**"
            relationship = f"As {var1} increases, {var2} tends to decrease"

        if not p_interp["is_significant"]:
            return f"""
**No significant correlation** between {var1} and {var2} (p={p_value:.3f}) ❌

The correlation is {correlation:.3f}, but this could be due to chance.
"""

        interpretation = f"""
**{strength.replace("**", "")} {direction.replace("**", "")} correlation** {p_interp["emoji"]}

- **Correlation coefficient**: {correlation:.3f}
- **Strength**: {strength} relationship
- **Direction**: {direction}
- **Interpretation**: {relationship}
- **Statistical Significance**: {p_interp["interpretation"]}
"""

        return interpretation

    @staticmethod
    def render_result_card(
        title: str,
        statistic: float,
        p_value: float,
        interpretation: str,
        additional_info: dict[str, any] | None = None,
    ):
        """
        Render a styled result card with interpretation.

        Args:
            title: Result title
            statistic: Main statistic value
            p_value: P-value
            interpretation: Plain-language interpretation
            additional_info: Optional additional metrics to display
        """
        p_interp = ResultInterpreter.interpret_p_value(p_value)

        with st.container():
            st.markdown(f"### {title}")

            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Statistic", f"{statistic:.3f}")
            with col2:
                st.metric("P-value", f"{p_value:.4f}")
            with col3:
                st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

            # Additional info
            if additional_info:
                cols = st.columns(len(additional_info))
                for col, (key, value) in zip(cols, additional_info.items()):
                    with col:
                        if isinstance(value, float):
                            st.metric(key, f"{value:.3f}")
                        else:
                            st.metric(key, value)

            # Interpretation
            st.markdown("#### 📖 Interpretation")
            st.markdown(interpretation)

            st.divider()

    @staticmethod
    def generate_methods_text(
        analysis_type: str,
        test_name: str,
        variables: dict[str, any],
        software: str = "Clinical Analytics Platform",
    ) -> str:
        """
        Generate methods section text for manuscripts.

        Args:
            analysis_type: Type of analysis performed
            test_name: Specific statistical test used
            variables: Dictionary of variables analyzed
            software: Software used

        Returns:
            Methods text ready for manuscript
        """
        methods_templates = {
            "descriptive": """
Descriptive statistics were calculated for all variables. Continuous variables are
presented as mean ± standard deviation or median (interquartile range) as appropriate.
Categorical variables are presented as frequencies and percentages.
All analyses were performed using {software}.
""",
            "group_comparison": """
{test_name} was used to compare {outcome} between {groups}. For continuous variables,
results are reported as mean difference with 95% confidence intervals.
For categorical variables, results are reported as proportions.
Statistical significance was defined as p<0.05 (two-tailed).
All analyses were performed using {software}.
""",
            "regression": """
{test_name} was performed to identify predictors of {outcome}. Results are reported as
odds ratios (OR) with 95% confidence intervals (CI). Variables with p<0.05 were considered
statistically significant. Model fit was assessed using pseudo-R² and likelihood ratio tests.
All analyses were performed using {software}.
""",
            "survival": """
{test_name} was used to analyze time to {event}. Survival curves were compared using the log-rank test.
Hazard ratios (HR) with 95% confidence intervals were calculated.
Censoring was handled using the Kaplan-Meier method.
Statistical significance was defined as p<0.05. All analyses were performed using {software}.
""",
            "correlation": """
Pearson correlation coefficients were calculated to assess relationships between continuous variables.
Correlation strength was interpreted as weak (|r|<0.3), moderate (0.3≤|r|<0.7), or strong (|r|≥0.7).
Statistical significance was defined as p<0.05 (two-tailed).
All analyses were performed using {software}.
""",
        }

        template = methods_templates.get(analysis_type, "")

        # Fill in template
        methods = template.format(
            test_name=test_name,
            software=software,
            outcome=variables.get("outcome", "[outcome]"),
            groups=variables.get("groups", "[groups]"),
            event=variables.get("event", "[event]"),
        )

        return methods.strip()

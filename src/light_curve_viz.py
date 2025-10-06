"""
Light Curve Visualization Generator
Creates synthetic transit light curves for exoplanet detections
"""

import numpy as np
import plotly.graph_objects as go


class LightCurveGenerator:
    """Generate and visualize exoplanet transit light curves."""

    def __init__(self):
        self.noise_level = 0.0001  # Realistic photometric noise

    def generate_transit_light_curve(self, period, depth, duration, num_transits=3):
        """
        Generate synthetic transit light curve.

        Args:
            period: Orbital period in days
            depth: Transit depth in ppm (parts per million)
            duration: Transit duration in hours
            num_transits: Number of transits to show

        Returns:
            time (array), flux (array)
        """
        # Time array covering multiple orbits
        total_time = period * (num_transits + 0.5)
        time = np.linspace(0, total_time, 2000)

        # Start with normalized flux (1.0 = no transit)
        flux = np.ones_like(time)

        # Convert duration from hours to days
        duration_days = duration / 24.0

        # Add transits
        for n in range(num_transits):
            transit_center = period * (n + 0.5)

            # Simple box model with smooth ingress/egress
            for i, t in enumerate(time):
                dt = abs(t - transit_center)

                if dt < duration_days / 2:
                    # In transit - use trapezoidal shape
                    ingress_egress = duration_days * 0.1  # 10% of duration for ingress/egress

                    if dt < duration_days / 2 - ingress_egress:
                        # Full transit depth
                        flux[i] -= depth / 1e6
                    else:
                        # Ingress/egress - linear ramp
                        ramp_position = (duration_days / 2 - dt) / ingress_egress
                        flux[i] -= (depth / 1e6) * ramp_position

        # Add realistic noise
        noise = np.random.normal(0, self.noise_level, len(flux))
        flux += noise

        return time, flux

    def create_light_curve_plot(self, planet_data, prediction_confidence=None):
        """
        Create interactive Plotly light curve visualization.

        Args:
            planet_data: Dict with planet properties
            prediction_confidence: Optional confidence score from ML model

        Returns:
            Plotly figure JSON
        """
        # Extract parameters
        period = planet_data.get('koi_period', 10)
        depth = planet_data.get('koi_depth', 500)  # ppm
        duration = planet_data.get('koi_duration', 3)  # hours
        name = planet_data.get('kepoi_name', 'Candidate')

        # Generate light curve
        time, flux = self.generate_transit_light_curve(period, depth, duration)

        # Convert flux to percentage
        flux_percent = (flux - 1) * 100

        # Create plot
        fig = go.Figure()

        # Add light curve
        fig.add_trace(go.Scatter(
            x=time,
            y=flux_percent,
            mode='lines',
            name='Brightness',
            line=dict(color='#3498db', width=1),
            hovertemplate='<b>Time</b>: %{x:.2f} days<br>' +
                          '<b>Brightness</b>: %{y:.4f}%<br>' +
                          '<extra></extra>'
        ))

        # Mark transit centers
        num_transits = 3
        for n in range(num_transits):
            transit_center = period * (n + 0.5)
            fig.add_vline(
                x=transit_center,
                line_dash="dash",
                line_color="red",
                opacity=0.3,
                annotation_text=f"Transit {n+1}",
                annotation_position="top"
            )

        # Update layout
        title_text = f"Transit Light Curve: {name}"
        if prediction_confidence:
            title_text += f" (Confidence: {prediction_confidence:.1%})"

        fig.update_layout(
            title=title_text,
            xaxis_title='Time (days)',
            yaxis_title='Brightness Change (%)',
            height=400,
            hovermode='x unified',
            template='plotly_white',
            showlegend=False
        )

        # Add annotations
        annotations_text = (
            f"Period: {period:.2f} days | "
            f"Depth: {depth:.0f} ppm | "
            f"Duration: {duration:.2f} hours"
        )

        fig.add_annotation(
            text=annotations_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor='center'
        )

        return fig.to_json()

    def create_phase_folded_plot(self, planet_data):
        """
        Create phase-folded light curve (all transits overlaid).
        Shows the 'average' transit signature.
        """
        period = planet_data.get('koi_period', 10)
        depth = planet_data.get('koi_depth', 500)
        duration = planet_data.get('koi_duration', 3)
        name = planet_data.get('kepoi_name', 'Candidate')

        # Generate single transit centered at phase 0
        duration_days = duration / 24.0
        phase = np.linspace(-0.1 * period, 0.1 * period, 500)
        flux = np.ones_like(phase)

        # Add transit shape
        ingress_egress = duration_days * 0.1
        for i, p in enumerate(phase):
            dt = abs(p)
            if dt < duration_days / 2:
                if dt < duration_days / 2 - ingress_egress:
                    flux[i] -= depth / 1e6
                else:
                    ramp_position = (duration_days / 2 - dt) / ingress_egress
                    flux[i] -= (depth / 1e6) * ramp_position

        # Add noise
        flux += np.random.normal(0, self.noise_level * 0.5, len(flux))
        flux_percent = (flux - 1) * 100

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=phase,
            y=flux_percent,
            mode='markers',
            name='Folded Data',
            marker=dict(size=3, color='#2ecc71', opacity=0.6),
            hovertemplate='<b>Phase</b>: %{x:.3f} days<br>' +
                          '<b>Brightness</b>: %{y:.4f}%<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            title=f"Phase-Folded Transit: {name}",
            xaxis_title='Orbital Phase (days from center)',
            yaxis_title='Brightness Change (%)',
            height=400,
            template='plotly_white',
            showlegend=False
        )

        return fig.to_json()

    def create_comparison_chart(self, planet_data):
        """
        Create visual comparison to known planets.
        """
        radius = planet_data.get('koi_prad', 1.0)
        temp = planet_data.get('koi_teq', 288)
        period = planet_data.get('koi_period', 365)

        # Known planets for comparison
        known_planets = {
            'Earth': {'radius': 1.0, 'temp': 288, 'period': 365, 'color': '#3498db'},
            'Mars': {'radius': 0.53, 'temp': 210, 'period': 687, 'color': '#e74c3c'},
            'Venus': {'radius': 0.95, 'temp': 737, 'period': 225, 'color': '#f39c12'},
            'Jupiter': {'radius': 11.2, 'temp': 165, 'period': 4333, 'color': '#95a5a6'},
            'Your Discovery': {'radius': radius, 'temp': temp, 'period': period, 'color': '#2ecc71'}
        }

        # Create subplots for size, temperature, and period
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Size Comparison', 'Temperature', 'Orbital Period'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )

        names = list(known_planets.keys())
        colors = [known_planets[n]['color'] for n in names]

        # Size comparison
        sizes = [known_planets[n]['radius'] for n in names]
        fig.add_trace(
            go.Bar(x=names, y=sizes, marker_color=colors, showlegend=False),
            row=1, col=1
        )
        fig.update_yaxes(title_text="Earth Radii", row=1, col=1)

        # Temperature comparison
        temps = [known_planets[n]['temp'] for n in names]
        fig.add_trace(
            go.Bar(x=names, y=temps, marker_color=colors, showlegend=False),
            row=1, col=2
        )
        fig.update_yaxes(title_text="Kelvin", row=1, col=2)

        # Period comparison (log scale)
        periods = [known_planets[n]['period'] for n in names]
        fig.add_trace(
            go.Bar(x=names, y=periods, marker_color=colors, showlegend=False),
            row=1, col=3
        )
        fig.update_yaxes(title_text="Days", type="log", row=1, col=3)

        fig.update_layout(
            height=400,
            title_text="How Does Your Discovery Compare?",
            template='plotly_white'
        )

        return fig.to_json()


def generate_discovery_story(planet_data, habitability_result, prediction_confidence):
    """
    Generate NASA-style press release description.
    """
    name = planet_data.get('kepoi_name', 'Candidate Planet')
    radius = planet_data.get('koi_prad', 1.0)
    temp = planet_data.get('koi_teq', 288)
    period = planet_data.get('koi_period', 365)
    steff = planet_data.get('koi_steff', 5778)

    # Build story
    story = f"**EXOPLANET DISCOVERY ALERT**\n\n"
    story += f"Scientists have identified **{name}**, a "

    # Size description
    if radius < 0.8:
        story += "sub-Earth-sized world "
    elif radius < 1.25:
        story += "roughly Earth-sized planet "
    elif radius < 2.0:
        story += "super-Earth "
    elif radius < 4.0:
        story += "mini-Neptune "
    else:
        story += "gas giant "

    # Temperature description
    if temp < 200:
        story += "in a frozen orbit "
    elif temp < 273:
        story += "in a cold but potentially habitable orbit "
    elif temp < 373:
        story += "in the habitable 'Goldilocks zone' "
    elif temp < 500:
        story += "in a warm orbit "
    else:
        story += "in a scorching orbit "

    # Star description
    if 5200 <= steff <= 6000:
        story += f"around a Sun-like star. "
    elif 3700 <= steff < 5200:
        story += f"around a cooler, orange dwarf star. "
    elif steff >= 6000:
        story += f"around a hot, bright star. "
    else:
        story += f"around a dim red dwarf star. "

    # Period
    if period < 1:
        story += f"This world completes an orbit in less than one Earth day! "
    elif period < 10:
        story += f"A year on this planet lasts just {period:.1f} days. "
    elif period < 100:
        story += f"It orbits its star every {period:.0f} days. "
    else:
        story += f"Its orbital period is {period:.0f} days. "

    # ML confidence
    story += f"\n\nOur machine learning model detected this planet with **{prediction_confidence:.1%} confidence**, "
    story += f"analyzing subtle dips in starlight as the planet passes in front of its host star.\n\n"

    # Habitability assessment
    if habitability_result:
        hab_score = habitability_result.get('overall_score', 0)
        classification = habitability_result.get('classification', '')

        story += f"**Habitability Assessment:** {classification}\n"
        story += f"**Score:** {hab_score:.2f}/1.00\n\n"

        if hab_score >= 0.6:
            story += "This exciting discovery ranks among the most promising candidates for potential habitability! "
            story += "Further observations could reveal if this world has an atmosphere and possibly even liquid water.\n"
        elif hab_score >= 0.4:
            story += "While not a perfect Earth analog, this planet is an intriguing target for follow-up observations "
            story += "and could teach us about planetary formation and evolution.\n"
        else:
            story += "Though unlikely to harbor life, this discovery adds to our catalog of exoplanets "
            story += "and helps us understand the incredible diversity of planetary systems.\n"

    story += f"\n*Discovery made using NASA Kepler/TESS mission data and advanced AI analysis.*"

    return story


if __name__ == "__main__":
    # Test light curve generation
    generator = LightCurveGenerator()

    test_planet = {
        'kepoi_name': 'Kepler-442b',
        'koi_period': 112.3,
        'koi_depth': 350,
        'koi_duration': 5.2,
        'koi_prad': 1.34,
        'koi_teq': 233,
        'koi_steff': 4402
    }

    print("Generating light curve visualization...")
    plot_json = generator.create_light_curve_plot(test_planet, prediction_confidence=0.95)
    print("Light curve generated successfully!")

    print("\nGenerating discovery story...")
    story = generate_discovery_story(test_planet, None, 0.95)
    print(story)

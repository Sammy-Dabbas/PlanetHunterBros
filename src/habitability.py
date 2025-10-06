"""
Exoplanet Habitability Assessment System
Advanced scoring for potential life-supporting planets
"""

import numpy as np
import pandas as pd


class HabitabilityScorer:
    """
    Calculate habitability scores for exoplanets.

    Based on:
    - Earth Similarity Index (ESI)
    - Habitable Zone position
    - Planet characteristics
    - Host star properties
    """

    def __init__(self):
        # Reference values (Earth = 1.0)
        self.earth_radius = 1.0  # Earth radii
        self.earth_mass = 1.0    # Earth masses
        self.earth_temp = 288    # Kelvin
        self.earth_flux = 1.0    # Solar flux

    def calculate_esi(self, radius, temp, escape_velocity=None):
        """
        Earth Similarity Index (ESI)
        Range: 0-1, where 1.0 = exactly like Earth

        Formula from Schulze-Makuch et al. (2011)
        """
        # Radius component (0-1)
        if radius > 0:
            esi_radius = 1 - abs((radius - self.earth_radius) / (radius + self.earth_radius))
        else:
            esi_radius = 0

        # Temperature component (0-1)
        if temp > 0:
            esi_temp = 1 - abs((temp - self.earth_temp) / (temp + self.earth_temp))
        else:
            esi_temp = 0

        # Combined ESI (geometric mean)
        esi = (esi_radius * esi_temp) ** 0.5

        return min(max(esi, 0), 1)  # Clamp to [0, 1]

    def habitable_zone_position(self, temp, insol):
        """
        Determine if planet is in habitable zone.

        Conservative HZ: 273-373K (liquid water)
        Optimistic HZ: 200-400K (allows subsurface water)

        Returns: position_score (0-1) and zone name
        """
        if temp is None or temp <= 0:
            return 0, "Unknown"

        if 273 <= temp <= 373:
            # Conservative habitable zone
            center = 323  # Ideal temp
            distance = abs(temp - center) / 50
            return max(1 - distance, 0.5), "Conservative HZ"

        elif 200 <= temp <= 273:
            # Cold edge (subsurface oceans possible)
            return 0.4, "Cold HZ Edge"

        elif 373 <= temp <= 400:
            # Hot edge (early Earth conditions)
            return 0.4, "Hot HZ Edge"

        elif 150 <= temp <= 200:
            # Too cold but interesting
            return 0.2, "Cold"

        elif 400 <= temp <= 500:
            # Too hot but interesting
            return 0.2, "Hot"

        else:
            # Definitely not habitable
            return 0, "Extreme"

    def size_category(self, radius):
        """Categorize planet by size."""
        if radius is None or radius <= 0:
            return "Unknown", 0

        if 0.5 <= radius <= 1.5:
            return "Earth-like", 1.0
        elif 1.5 <= radius <= 2.0:
            return "Super-Earth", 0.8
        elif 2.0 <= radius <= 4.0:
            return "Mini-Neptune", 0.3
        elif 4.0 <= radius <= 10:
            return "Neptune-like", 0.1
        elif radius > 10:
            return "Jupiter-like", 0.0
        else:
            return "Sub-Earth", 0.6

    def stellar_habitability(self, steff, srad):
        """
        Assess host star suitability for life.

        Best: Sun-like (G-type) stars
        Good: K-type (orange dwarfs) - long-lived, stable
        Fair: F-type (hotter than Sun) - shorter-lived
        Poor: M-type (red dwarfs) - tidal locking issues
        """
        if steff is None or steff <= 0:
            return 0.5, "Unknown"

        if 5200 <= steff <= 6000:
            # G-type (Sun-like)
            return 1.0, "Sun-like (G-type)"
        elif 3700 <= steff <= 5200:
            # K-type (orange dwarf)
            return 0.9, "Orange Dwarf (K-type)"
        elif 6000 <= steff <= 7500:
            # F-type (hotter than Sun)
            return 0.7, "Hot Star (F-type)"
        elif 2400 <= steff <= 3700:
            # M-type (red dwarf)
            return 0.5, "Red Dwarf (M-type)"
        else:
            return 0.3, "Extreme Star"

    def orbital_stability(self, period, eccen):
        """
        Assess orbital stability (important for climate).

        Circular orbits (low eccentricity) = stable climate
        Eccentric orbits = extreme seasons
        """
        if eccen is None:
            return 0.7, "Unknown"

        if eccen < 0.1:
            return 1.0, "Circular (Stable)"
        elif 0.1 <= eccen < 0.3:
            return 0.7, "Low Eccentricity"
        elif 0.3 <= eccen < 0.5:
            return 0.4, "Moderate Eccentricity"
        else:
            return 0.2, "High Eccentricity (Unstable)"

    def calculate_habitability_score(self, planet_data):
        """
        Comprehensive habitability assessment.

        Input: Dict with planet properties
        Output: Dict with scores and assessment
        """
        # Extract properties (handle missing values)
        radius = planet_data.get('koi_prad', None)
        temp = planet_data.get('koi_teq', None)
        insol = planet_data.get('koi_insol', None)
        period = planet_data.get('koi_period', None)
        eccen = planet_data.get('koi_eccen', 0)
        steff = planet_data.get('koi_steff', None)
        srad = planet_data.get('koi_srad', None)

        # Calculate individual scores
        esi = self.calculate_esi(radius if radius else 1, temp if temp else 288)
        hz_score, hz_zone = self.habitable_zone_position(temp, insol)
        size_cat, size_score = self.size_category(radius)
        star_score, star_type = self.stellar_habitability(steff, srad)
        orbit_score, orbit_type = self.orbital_stability(period, eccen)

        # Weighted overall score
        overall_score = (
            esi * 0.35 +              # Earth similarity (35%)
            hz_score * 0.30 +         # Habitable zone (30%)
            size_score * 0.15 +       # Planet size (15%)
            star_score * 0.10 +       # Star type (10%)
            orbit_score * 0.10        # Orbital stability (10%)
        )

        # Classify habitability
        if overall_score >= 0.8:
            classification = "🌟 Highly Habitable"
            description = "Excellent candidate for life!"
        elif overall_score >= 0.6:
            classification = "🌍 Potentially Habitable"
            description = "Good prospects for habitability"
        elif overall_score >= 0.4:
            classification = "🔬 Interesting Target"
            description = "Worth further study"
        elif overall_score >= 0.2:
            classification = "⚠️ Marginal"
            description = "Low but non-zero habitability"
        else:
            classification = "❌ Not Habitable"
            description = "Unlikely to support life"

        return {
            'overall_score': overall_score,
            'classification': classification,
            'description': description,
            'components': {
                'earth_similarity_index': esi,
                'habitable_zone_score': hz_score,
                'habitable_zone': hz_zone,
                'size_score': size_score,
                'size_category': size_cat,
                'stellar_score': star_score,
                'stellar_type': star_type,
                'orbital_score': orbit_score,
                'orbital_type': orbit_type
            },
            'key_properties': {
                'radius': f"{radius:.2f} Earth radii" if radius else "Unknown",
                'temperature': f"{temp:.0f} K" if temp else "Unknown",
                'orbital_period': f"{period:.1f} days" if period else "Unknown",
                'star_temperature': f"{steff:.0f} K" if steff else "Unknown"
            }
        }

    def compare_to_known_planets(self, score):
        """Compare habitability to known planets."""
        if score >= 0.95:
            return "More habitable than Mars, similar to Earth!"
        elif score >= 0.8:
            return "As habitable as early Earth"
        elif score >= 0.6:
            return "Similar to Mars or Europa (subsurface life possible)"
        elif score >= 0.4:
            return "Similar to Venus (extreme but interesting)"
        else:
            return "More like Jupiter or Mercury (unlikely for life)"

    def generate_discovery_description(self, planet_data, hab_score):
        """
        Generate a press-release style description.
        Creative storytelling for discoveries!
        """
        name = planet_data.get('kepoi_name', 'Unknown')
        radius = planet_data.get('koi_prad', 1)
        temp = planet_data.get('koi_teq', 288)
        period = planet_data.get('koi_period', 365)

        # Size comparison
        if radius < 0.8:
            size_desc = "smaller than Earth"
        elif radius > 1.2:
            size_desc = f"{radius:.1f}x larger than Earth"
        else:
            size_desc = "Earth-sized"

        # Temperature comparison
        if temp < 273:
            temp_desc = "frozen world"
        elif temp > 373:
            temp_desc = "scorching hot planet"
        else:
            temp_desc = "temperate world"

        # Year comparison
        if period < 10:
            year_desc = f"year lasts only {period:.1f} days"
        elif period > 365:
            year_desc = f"year spans {period/365:.1f} Earth years"
        else:
            year_desc = f"year is {period:.0f} days long"

        # Build description
        description = f"🌍 {name} is a {size_desc} {temp_desc} where a {year_desc}. "

        classification = hab_score['classification']
        if "Highly Habitable" in classification:
            description += "This remarkable planet sits in the perfect zone for liquid water and could potentially harbor life as we know it! 🌟"
        elif "Potentially Habitable" in classification:
            description += "With conditions favorable for water and a stable climate, this planet is an exciting target in the search for extraterrestrial life! 🔭"
        elif "Interesting" in classification:
            description += "While challenging, this world offers unique conditions that could support extremophile life forms. 🔬"
        else:
            description += "Though inhospitable by Earth standards, this planet teaches us about the diversity of worlds in our galaxy. 🪐"

        return description


def calculate_dataset_habitability(df):
    """Calculate habitability for entire dataset."""
    scorer = HabitabilityScorer()
    results = []

    for idx, row in df.iterrows():
        hab_score = scorer.calculate_habitability_score(row.to_dict())
        results.append({
            'index': idx,
            'name': row.get('kepoi_name', f'Planet_{idx}'),
            'overall_score': hab_score['overall_score'],
            'classification': hab_score['classification'],
            'esi': hab_score['components']['earth_similarity_index'],
            'hz_score': hab_score['components']['habitable_zone_score'],
            'hz_zone': hab_score['components']['habitable_zone']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with example planet
    scorer = HabitabilityScorer()

    # Earth-like planet example
    example_planet = {
        'kepoi_name': 'Kepler-442b',
        'koi_prad': 1.34,      # Slightly larger than Earth
        'koi_teq': 233,        # Cool but in HZ
        'koi_period': 112.3,   # 112 day year
        'koi_eccen': 0.04,     # Nearly circular
        'koi_steff': 4402,     # K-type star
        'koi_srad': 0.6,
        'koi_insol': 0.7
    }

    result = scorer.calculate_habitability_score(example_planet)

    print("="*60)
    print("HABITABILITY ASSESSMENT")
    print("="*60)
    print(f"\nPlanet: {example_planet['kepoi_name']}")
    print(f"\nOverall Score: {result['overall_score']:.2f}/1.00")
    print(f"Classification: {result['classification']}")
    print(f"Assessment: {result['description']}")

    print(f"\nComponents:")
    for key, value in result['components'].items():
        print(f"  {key}: {value}")

    print(f"\nKey Properties:")
    for key, value in result['key_properties'].items():
        print(f"  {key}: {value}")

    print(f"\nComparison: {scorer.compare_to_known_planets(result['overall_score'])}")

    print(f"\nDiscovery Description:")
    print(scorer.generate_discovery_description(example_planet, result))
    print("="*60)

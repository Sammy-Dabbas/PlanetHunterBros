"""
Planet Characterization System
Predict specific planet properties beyond just detection
"""

import numpy as np


class PlanetCharacterizer:
    """Characterize exoplanet properties from observational data."""

    def classify_planet_size(self, radius):
        """
        Classify planet by size category.

        Categories based on NASA's exoplanet classification:
        - Sub-Earth: Smaller than Earth
        - Earth-sized: Similar to Earth
        - Super-Earth: Larger rocky planets
        - Mini-Neptune: Small gas planets
        - Neptune-like: Ice giants
        - Jupiter-like: Gas giants
        """
        if radius is None or radius <= 0:
            return {
                'category': 'Unknown',
                'description': 'Insufficient data',
                'emoji': '❓',
                'color': '#999999'
            }

        if radius < 0.8:
            return {
                'category': 'Sub-Earth',
                'description': 'Smaller than Earth (like Mars)',
                'emoji': '🪨',
                'color': '#c44536',
                'examples': ['Mars (0.53 R⊕)', 'Mercury (0.38 R⊕)']
            }
        elif radius < 1.25:
            return {
                'category': 'Earth-sized',
                'description': 'Similar size to Earth',
                'emoji': '🌍',
                'color': '#3498db',
                'examples': ['Earth (1.00 R⊕)', 'Venus (0.95 R⊕)']
            }
        elif radius < 2.0:
            return {
                'category': 'Super-Earth',
                'description': 'Larger rocky planet',
                'emoji': '🌎',
                'color': '#2ecc71',
                'examples': ['Kepler-452b (1.6 R⊕)', 'LHS 1140b (1.4 R⊕)']
            }
        elif radius < 4.0:
            return {
                'category': 'Mini-Neptune',
                'description': 'Small gas/ice planet',
                'emoji': '🔵',
                'color': '#3498db',
                'examples': ['Kepler-11b (1.97 R⊕)', 'GJ 1214b (2.7 R⊕)']
            }
        elif radius < 10:
            return {
                'category': 'Neptune-like',
                'description': 'Ice giant',
                'emoji': '💠',
                'color': '#5dade2',
                'examples': ['Neptune (3.88 R⊕)', 'Uranus (4.01 R⊕)']
            }
        else:
            return {
                'category': 'Jupiter-like',
                'description': 'Gas giant',
                'emoji': '🪐',
                'color': '#e67e22',
                'examples': ['Jupiter (11.2 R⊕)', 'Saturn (9.45 R⊕)']
            }

    def classify_temperature(self, temp):
        """
        Classify planet temperature zone.
        """
        if temp is None or temp <= 0:
            return {
                'category': 'Unknown',
                'description': 'Temperature not available',
                'emoji': '❓',
                'color': '#999999'
            }

        if temp < 150:
            return {
                'category': 'Frozen',
                'description': 'Extremely cold, frozen world',
                'emoji': '❄️',
                'color': '#aed6f1',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Colder than Neptune'
            }
        elif temp < 200:
            return {
                'category': 'Very Cold',
                'description': 'Ice and frozen gases likely',
                'emoji': '🧊',
                'color': '#85c1e9',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Like Pluto or outer solar system'
            }
        elif temp < 273:
            return {
                'category': 'Cold',
                'description': 'Below freezing, but subsurface oceans possible',
                'emoji': '🌨️',
                'color': '#5dade2',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Like Mars or Europa'
            }
        elif temp < 373:
            return {
                'category': 'Habitable Zone',
                'description': 'Liquid water possible!',
                'emoji': '💧',
                'color': '#2ecc71',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Like Earth (0-100°C)'
            }
        elif temp < 500:
            return {
                'category': 'Hot',
                'description': 'Too hot for liquid water',
                'emoji': '🔥',
                'color': '#f39c12',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Like Venus'
            }
        elif temp < 1000:
            return {
                'category': 'Very Hot',
                'description': 'Extreme temperatures, possible lava',
                'emoji': '🌋',
                'color': '#e67e22',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Hotter than Venus'
            }
        else:
            return {
                'category': 'Scorching',
                'description': 'Extreme heat, molten surface',
                'emoji': '☄️',
                'color': '#c0392b',
                'celsius': f'{temp - 273:.0f}°C',
                'comparison': 'Ultra-hot Jupiter'
            }

    def classify_star_type(self, steff):
        """
        Classify host star type (spectral class).

        Based on temperature:
        - M: Red dwarf (2400-3700 K)
        - K: Orange dwarf (3700-5200 K)
        - G: Yellow dwarf/Sun-like (5200-6000 K)
        - F: White star (6000-7500 K)
        - A, B, O: Hotter stars (>7500 K)
        """
        if steff is None or steff <= 0:
            return {
                'category': 'Unknown',
                'description': 'Star type unknown',
                'emoji': '⭐',
                'color': '#999999'
            }

        if steff < 2400:
            return {
                'category': 'Brown Dwarf',
                'description': 'Failed star, very dim',
                'spectral_class': 'L/T',
                'emoji': '🟤',
                'color': '#5d4037',
                'habitability_note': 'Poor for life (too dim, flares)'
            }
        elif steff < 3700:
            return {
                'category': 'Red Dwarf',
                'description': 'Cool, long-lived star',
                'spectral_class': 'M',
                'emoji': '🔴',
                'color': '#e74c3c',
                'habitability_note': 'Tidal locking concerns, but very common'
            }
        elif steff < 5200:
            return {
                'category': 'Orange Dwarf',
                'description': 'Stable, long-lived star',
                'spectral_class': 'K',
                'emoji': '🟠',
                'color': '#f39c12',
                'habitability_note': 'Excellent for life! Stable and long-lived'
            }
        elif steff < 6000:
            return {
                'category': 'Sun-like Star',
                'description': 'Yellow dwarf, like our Sun',
                'spectral_class': 'G',
                'emoji': '🟡',
                'color': '#f1c40f',
                'habitability_note': 'Ideal for life (like Earth!)'
            }
        elif steff < 7500:
            return {
                'category': 'Hot Star',
                'description': 'White/blue-white star',
                'spectral_class': 'F',
                'emoji': '⚪',
                'color': '#ecf0f1',
                'habitability_note': 'Shorter lifespan, but possible'
            }
        else:
            return {
                'category': 'Very Hot Star',
                'description': 'Blue/massive star',
                'spectral_class': 'A/B/O',
                'emoji': '🔵',
                'color': '#3498db',
                'habitability_note': 'Too short-lived for complex life'
            }

    def estimate_composition(self, radius, mass=None):
        """
        Estimate planet composition based on mass-radius relationship.

        If mass unknown, estimate from radius using empirical relations.
        """
        if radius is None or radius <= 0:
            return {
                'composition': 'Unknown',
                'description': 'Insufficient data',
                'emoji': '❓'
            }

        # If no mass, estimate from radius (empirical relations)
        if mass is None:
            if radius < 1.5:
                # Likely rocky (Earth-like density)
                mass_estimate = radius ** 3.7  # Empirical for rocky planets
                composition = 'Rocky'
            elif radius < 4.0:
                # Transition zone (rocky core + volatiles)
                mass_estimate = radius ** 2.5
                composition = 'Mixed'
            else:
                # Likely gas (low density)
                mass_estimate = radius ** 1.3
                composition = 'Gaseous'
        else:
            mass_estimate = mass

        # Calculate density (relative to Earth)
        if mass_estimate and radius:
            density_rel = mass_estimate / (radius ** 3)

            if density_rel > 0.8:
                return {
                    'composition': 'Rocky/Iron',
                    'description': 'Dense, likely iron-rich core',
                    'emoji': '🪨',
                    'density': f'{density_rel:.2f} Earth densities',
                    'examples': ['Mercury', 'Earth', 'Mars']
                }
            elif density_rel > 0.3:
                return {
                    'composition': 'Rocky',
                    'description': 'Silicate/rock composition',
                    'emoji': '🌎',
                    'density': f'{density_rel:.2f} Earth densities',
                    'examples': ['Earth', 'Venus']
                }
            elif density_rel > 0.1:
                return {
                    'composition': 'Mixed (Ice/Rock)',
                    'description': 'Water ice and rock',
                    'emoji': '🧊',
                    'density': f'{density_rel:.2f} Earth densities',
                    'examples': ['Uranus', 'Neptune', 'Europa']
                }
            else:
                return {
                    'composition': 'Gas/Ice',
                    'description': 'Hydrogen/helium dominated',
                    'emoji': '💨',
                    'density': f'{density_rel:.2f} Earth densities',
                    'examples': ['Jupiter', 'Saturn']
                }

        return {
            'composition': 'Unknown',
            'description': 'Insufficient mass data',
            'emoji': '❓'
        }

    def characterize_planet(self, planet_data):
        """
        Full planet characterization from observational data.

        Returns comprehensive classification of:
        - Size category
        - Temperature zone
        - Star type
        - Estimated composition
        """
        radius = planet_data.get('koi_prad', None)
        temp = planet_data.get('koi_teq', None)
        steff = planet_data.get('koi_steff', None)
        period = planet_data.get('koi_period', None)

        result = {
            'size': self.classify_planet_size(radius),
            'temperature': self.classify_temperature(temp),
            'star': self.classify_star_type(steff),
            'composition': self.estimate_composition(radius),
            'summary': self._generate_summary(radius, temp, steff, period)
        }

        return result

    def _generate_summary(self, radius, temp, steff, period):
        """Generate human-readable summary of planet characteristics."""
        size_info = self.classify_planet_size(radius)
        temp_info = self.classify_temperature(temp)
        star_info = self.classify_star_type(steff)

        summary = f"A {size_info['category']} "

        if temp:
            summary += f"in the {temp_info['category']} zone "

        if steff:
            summary += f"orbiting a {star_info['category']} "

        if period:
            if period < 1:
                summary += f"with an ultra-short {period:.2f} day orbit"
            elif period < 10:
                summary += f"with a {period:.1f} day orbit"
            elif period < 365:
                summary += f"with a {period:.0f} day year"
            else:
                summary += f"with a {period/365:.1f} Earth-year orbit"

        return summary


if __name__ == "__main__":
    # Test characterization
    characterizer = PlanetCharacterizer()

    # Earth-like planet
    earth_like = {
        'koi_prad': 1.0,
        'koi_teq': 288,
        'koi_steff': 5778,
        'koi_period': 365
    }

    # Hot Jupiter
    hot_jupiter = {
        'koi_prad': 11.0,
        'koi_teq': 1500,
        'koi_steff': 6200,
        'koi_period': 3.5
    }

    print("="*60)
    print("PLANET CHARACTERIZATION TEST")
    print("="*60)

    for name, planet in [("Earth-like", earth_like), ("Hot Jupiter", hot_jupiter)]:
        print(f"\n{name} Planet:")
        result = characterizer.characterize_planet(planet)

        print(f"  Size: {result['size']['emoji']} {result['size']['category']}")
        print(f"  Temp: {result['temperature']['emoji']} {result['temperature']['category']}")
        print(f"  Star: {result['star']['emoji']} {result['star']['category']}")
        print(f"  Composition: {result['composition']['emoji']} {result['composition']['composition']}")
        print(f"\n  Summary: {result['summary']}")

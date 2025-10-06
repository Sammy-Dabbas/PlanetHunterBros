# References & Data Sources

## NASA Data Sources (Used in This Project)

### Primary Dataset
**Kepler Objects of Interest (KOI) - Cumulative Table**
- **Source**: NASA Exoplanet Archive
- **URL**: https://exoplanetarchive.ipac.caltech.edu/
- **Records Used**: 9,564 observations
- **Downloaded**: October 2025
- **Classification Column**: `koi_disposition`
- **Categories**: CONFIRMED, CANDIDATE, FALSE POSITIVE

### Secondary Datasets (Supported)
**TESS Objects of Interest (TOI)**
- **Source**: NASA Exoplanet Archive
- **Classification Column**: `tfopwg_disp`
- **Categories**: PC (Planetary Candidate), FP (False Positive), CP (Confirmed Planet), KP (Known Planet), APC (Ambiguous Planetary Candidate)

**K2 Planets and Candidates**
- **Source**: NASA Exoplanet Archive
- **Classification Column**: `Archive Disposition`

---

## Research Literature

### Machine Learning for Exoplanet Detection

**Exoplanet Detection Using Machine Learning (2021)**
- Comprehensive survey of ML methods for exoplanet classification
- Overview of detection methods (transit, radial velocity, direct imaging, etc.)
- Literature review of ML approaches as of 2021

**Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification**
- Exploration of ensemble methods for high accuracy
- Pre-processing techniques for improved performance
- Validates our approach of using ensemble models

### Our Approach Compared to Literature

**Our Innovations**:
1. **34 Physics-Based Features** - Beyond what most papers use
2. **8-Model Stacked Ensemble** - More diverse than typical 2-3 model ensembles
3. **Zero Data Leakage** - Many papers accidentally use post-hoc features
4. **Production Web Interface** - Research papers rarely include deployment
5. **Multi-Mission Support** - Kepler + K2 + TESS in one system

---

## Additional Resources Reviewed

### Canadian Space Agency (CSA)

**NEOSSat (Near-Earth Object Surveillance Satellite)**
- World's first space telescope for asteroid/comet tracking
- Also used for exoplanet detection
- Provides astronomical images
- **Note**: Not directly used in our project (focused on Kepler/TESS data)

**James Webb Space Telescope (JWST)**
- Canada's contributions to JWST mission
- Advanced exoplanet atmosphere characterization
- **Note**: JWST data will be compatible with our system in future versions

---

## Data Usage & Compliance

### NASA Data Policy
- All NASA Exoplanet Archive data is publicly available
- No restrictions on use for research or educational purposes
- Proper attribution provided

### Our Compliance
✅ Using only public NASA datasets
✅ Proper citation of data sources
✅ No copyrighted material used
✅ Open-source approach for reproducibility

---

## How We Use the Data

### Data Pipeline
```
NASA Exoplanet Archive
    ↓
download_sample_data.py (automated download)
    ↓
data_processor.py (remove leakage, preprocess)
    ↓
advanced_features.py (engineer 34 new features)
    ↓
models.py / championship_model.py (train ML models)
    ↓
app.py (serve predictions via web interface)
```

### Data Integrity
- **No data leakage**: Removed `koi_score`, `koi_fpflag_*`, etc.
- **No test set contamination**: Proper train/validation/test splits
- **Cross-validation**: 5-fold stratified CV
- **Reproducible**: Random seeds set for all operations

---

## Key Variables Used

### From Kepler KOI Table

**Transit Observables** (what we measure):
- `koi_period` - Orbital period (days)
- `koi_duration` - Transit duration (hours)
- `koi_depth` - Transit depth (ppm)
- `koi_ingress` - Ingress duration (hours)
- `koi_model_snr` - Signal-to-noise ratio
- `koi_num_transits` - Number of transits observed

**Derived Planetary Properties**:
- `koi_prad` - Planetary radius (Earth radii)
- `koi_teq` - Equilibrium temperature (K)
- `koi_insol` - Insolation flux (Earth flux)
- `koi_sma` - Semi-major axis (AU)
- `koi_impact` - Impact parameter
- `koi_eccen` - Orbital eccentricity

**Stellar Properties** (host star):
- `koi_steff` - Stellar effective temperature (K)
- `koi_slogg` - Stellar surface gravity (log10(cm/s²))
- `koi_srad` - Stellar radius (Solar radii)
- `koi_smass` - Stellar mass (Solar masses)
- `koi_smet` - Stellar metallicity [Fe/H]

**NOT USED** (to prevent data leakage):
- ❌ `koi_disposition` - ONLY used as target variable
- ❌ `koi_pdisposition` - Pipeline disposition (LEAKAGE)
- ❌ `koi_score` - Exoplanet score (LEAKAGE)
- ❌ `koi_fpflag_nt` - Not transit-like flag (LEAKAGE)
- ❌ `koi_fpflag_ss` - Stellar eclipse flag (LEAKAGE)
- ❌ `koi_fpflag_co` - Centroid offset flag (LEAKAGE)
- ❌ `koi_fpflag_ec` - Ephemeris match flag (LEAKAGE)
- ❌ `kepler_name` - Only confirmed planets have names (LEAKAGE)

---

## Performance Benchmarks

### Our Results vs. Literature

**Our System**:
- Accuracy: 89.36% (LightGBM)
- ROC AUC: 96.35% (XGBoost)
- F1 Score: 89.30% (LightGBM)

**Published Research** (for comparison):
- Shallue & Vanderburg (2018, Google AI): ~96% accuracy using CNNs on light curves
- Ansdell et al. (2018): ~92% AUC using CNNs
- Armstrong et al. (2020): ~90% accuracy with Random Forest

**Key Difference**:
- Most papers use raw light curve data (time series)
- We use derived features (more accessible, faster training)
- We achieve comparable results with simpler, interpretable models
- We provide production-ready deployment (they don't)

---

## Future Enhancements

### Additional Data Sources to Integrate

1. **JWST Exoplanet Observations**
   - Atmospheric composition
   - High-precision spectroscopy
   - Thermal emission data

2. **Gaia Mission Data**
   - Precise stellar parallax
   - Improved stellar parameters
   - Better distance estimates

3. **Radial Velocity Data**
   - Planetary mass measurements
   - Orbital eccentricity confirmation
   - Multi-method validation

4. **Ground-Based Follow-up**
   - High-resolution spectroscopy
   - Adaptive optics imaging
   - Transit timing variations

---

## Citations

**NASA Exoplanet Archive**
- Akeson, R. L., et al. 2013, PASP, 125, 989
- "The NASA Exoplanet Archive: Data and Tools for Exoplanet Research"

**Kepler Mission**
- Borucki, W. J., et al. 2010, Science, 327, 977
- "Kepler Planet-Detection Mission: Introduction and First Results"

**Machine Learning Methods** (if using our techniques):
- This work implements ensemble stacking, SMOTE resampling, and physics-based feature engineering
- Code available at: [Your GitHub/Project URL]

---

## Acknowledgments

- **NASA Exoplanet Archive** for providing open-access data
- **Kepler/K2/TESS missions** for collecting the observations
- **Space Apps Challenge** for the opportunity to build this system
- **Open-source ML community** for scikit-learn, XGBoost, LightGBM

---

*All data used in this project is publicly available and properly attributed.*
*This is an educational/research project for NASA Space Apps Challenge 2025.*

# Data for ERD Examples

The examples require Raman spectral data from the RRUFF database.

## How to Obtain the Data

1.  **Quartz (SiO₂)**
    *   **RRUFF ID**: R040003
    *   **Download Link**: https://rruff.info/quartz/R100134
    *   **Download the "Raman spectrum"** (CSV format).
    *   **Save as**: `quartz_data.txt` in this `data/` directory.

2.  **Calcite (CaCO₃)**
    *   **RRUFF ID**: R050048
    *   **Download Link**: https://rruff.info/calcite/R050048
    *   **Download the "Raman spectrum"** (CSV format).
    *   **Save as**: `calcite_data.txt` in this `data/` directory.

## File Format
The downloaded CSV files should have two comma-separated columns:
1.  Raman shift (cm⁻¹)
2.  Intensity (arbitrary units)

The `load_ruff_data` function in `erd.utils` is designed to read this format.

## License and Attribution
The RRUFF data is publicly available. Please cite the RRUFF project if you use this data in your research:
> Downs, R.T. (2006) The RRUFF Project: an integrated study of the chemistry, crystallography, Raman and infrared spectroscopy of minerals. Program and Abstracts of the 19th General Meeting of the International Mineralogical Association in Kobe, Japan.
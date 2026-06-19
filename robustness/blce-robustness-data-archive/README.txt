This README.txt file was generated on 2026-01-21 by Benjamin Black

GENERAL INFORMATION

1. Title of Dataset: Bern Landscape Change Explorer Landscape Robustness Results.    

2. Author Information
	A. Principal Investigator Contact Information
		Name: Benjamin Black
		Institution:  Institute for Spatial and Landscape Development, Swiss Federal Institute of Technology (ETH) 
		Address:  HIL H 52.1, Stefano-Franscini-Platz 5 CH-8093 Zürich, Switzerland
		Email: bblack@ethz.ch

	B. Additional researcher Contact Information
		Name: Manuel Kurmann
		Institution:  Institute for Spatial and Landscape Development, Swiss Federal Institute of Technology (ETH) 
		Address:  HIL H 52.1, Stefano-Franscini-Platz 5 CH-8093 Zürich, Switzerland
		Email: mankurma@ethz.ch

  C. Additional researcher Contact Information
		Name: Isabel Nicholson Thomas
    Institution:  Institute for Spatial and Landscape Development, Swiss Federal Institute of Technology (ETH)
    Address:  HIL H 52.1, Stefano-Franscini-Platz 5 CH-8093 Zürich, Switzerland
		Email: inthomas@ethz.ch


3. Date of data collection: 2025-06-01 > 2026-01-1 

4. Geographic location of data: Switzerland 


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: Creative Commons Attribution 4.0 International 

2. Recommended citation for this dataset: Black, B., Kurmann, M., & Nicholson Thomas, I. (2026). Bern Landscape Change Explorer Landscape Robustness Results (Version 1.0.0) [Data set]. ETH Zurich. https://doi.org/10.5281/zenodo.18327228

DESCRIPTION/REPRODUCIBILITY: 
- This dataset contains rasters of landscape robustness for the Bern Landscape Change Explorer project.
- All rasters are projected in the CH1903+ / LV95 coordinate reference system and share the same extent and resolution (1 hectare).
- For detailed descriptions of the data generation process, raster value interpretation and suggested colour scheme, please refer to the accompanying metadata files:
-  BeschreibungMetadatenRobustheit_DE.pdf
-  BeschreibungMetadatenRobustheit_ENG.pdf

Two raster-based indicators are provided for each topic for the robustness evaluation: performance and
stability. The performance indicator is stored in files with the prefix Mean_sum_of_change_ and
describes the mean cumulative change value across all scenarios and future points in time. The
stability indicator is stored in files with the prefix Undesirable_deviation_sum_of_change_ and
describes the extent of undesirable negative deviations across all scenarios. Both indicators are
available for biodiversity (BD), ecosystem services (ES), and the combination of biodiversity and
ecosystem services (BD-ES). The robustness classes on the platform are derived from the joint
classification of these two indicator grids and are therefore not included as a separate grid layer.

Included files:
BeschreibungMetadatenRobustheit_DE.pdf
BeschreibungMetadatenRobustheit_ENG.pdf
Mean_sum_of_change_BD-ES.tif
Mean_sum_of_change_BD.tif
Mean_sum_of_change_ES.tif
Undesirable_deviation_sum_of_change_BD-ES.tif
Undesirable_deviation_sum_of_change_BD.tif
Undesirable_deviation_sum_of_change_ES.tif

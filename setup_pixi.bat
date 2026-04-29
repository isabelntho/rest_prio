@echo off
echo Installing packages into existing pixi workspace...

pixi add python=3.11

pixi add numpy pandas matplotlib seaborn scipy scikit-learn
pixi add rasterio geopandas pyogrio shapely fiona
pixi add ipywidgets ipykernel jupyter nbformat
pixi add openpyxl plotly

pixi add pymoo
IF %ERRORLEVEL% NEQ 0 (
    echo pymoo not found on conda-forge, trying PyPI...
    pixi add --pypi pymoo
)

echo.
echo Done! Run 'pixi shell' to activate the environment.
echo Then test with: python -c "import numpy; import pandas; print('OK')"

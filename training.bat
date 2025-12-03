:: Options:
:: pd_variant=[im, ilp, hm]

set pd_variant=%1

del /F /Q Output\T\PetriNet\*

python training.py %pd_variant%





set use_baseline=%1
set pd_variant=%2
set clustering_technique=%3
set n_iteration=%4


del /F /Q Output\OA\Diagnoses\*
del /F /Q Output\OA\PetriNets\*
del /F /Q Output\OA\Timing\*

python online_adaptation.py %use_baseline% %pd_variant% %clustering_technique% %n_iteration%



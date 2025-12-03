:: Options:
:: anomaly_type=[V, W]; one anomaly type at a time
:: pd_variant=[im, ilp, hm]
:: n_clusters=<integer>
:: n_simulation_traces=<integer>

set pd_variant=hm ilp im
set clustering_technique=optics dpgmm dbscan
set n_reps=3
set iterations=1 2 3 4
set use_baseline=0

for /D %%p IN ("Results\Method\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /l %%a in (1, 1, %n_reps%) do (

	mkdir Results\Method\%%a
	
	for %%x in (%pd_variant%) do (
	
		mkdir Results\Method\%%a\%%x\ITERATION_0
		mkdir Results\Method\%%a\%%x\ITERATION_0\PetriNets
		mkdir Results\Method\%%a\%%x\ITERATION_0\EventLogs
	
		copy Data\Output\Method\ITERATION_0\L_TR.xes Results\Method\%%a\%%x\ITERATION_0\EventLogs
		ren Results\Method\%%a\%%x\ITERATION_0\EventLogs\L_TR.xes L_TR_0.xes
		copy Data\Output\Method\ITERATION_0\L_TST_N.xes Results\Method\%%a\%%x\ITERATION_0\EventLogs
		ren Results\Method\%%a\%%x\ITERATION_0\EventLogs\L_TST_N.xes L_TST_N_0.xes
		copy Data\Output\Method\ITERATION_0\L_TST_A.xes Results\Method\%%a\%%x\ITERATION_0\EventLogs
		
		copy Data\Output\Method\ITERATION_0\L_TR.xes Input\T\Data
	
		call training %%x
		
		copy Output\T\PetriNet\PN.pnml Results\Method\%%a\%%x\ITERATION_0\PetriNets
		ren Results\Method\%%a\%%x\ITERATION_0\PetriNets\PN.pnml PN_0.pnml
		
		for %%c in (%clustering_technique%) do (
		
			mkdir Results\Method\%%a\%%x\%%c
		
			for %%y in (%iterations%) do (
				mkdir Results\Method\%%a\%%x\%%c\ITERATION_%%y
				mkdir Results\Method\%%a\%%x\%%c\ITERATION_%%y\PetriNets
				mkdir Results\Method\%%a\%%x\%%c\ITERATION_%%y\EventLogs
				mkdir Results\Method\%%a\%%x\%%c\ITERATION_%%y\Diagnoses
				mkdir Results\Method\%%a\%%x\%%c\ITERATION_%%y\Timing
			)

			:: === Copy test event logs consistently ===
			for %%i in (0 1 2 3 4) do (
				for /l %%j in (%%i,1,4) do (
					if %%j geq %%i (
						if exist Data\Output\Method\ITERATION_%%i\L_TST_N.xes (
							copy Data\Output\Method\ITERATION_%%i\L_TST_N.xes Results\Method\%%a\%%x\%%c\ITERATION_%%j\EventLogs >nul
							ren Results\Method\%%a\%%x\%%c\ITERATION_%%j\EventLogs\L_TST_N.xes L_TST_N_%%i.xes
						)
						if exist Data\Output\Method\ITERATION_%%i\L_TST_A.xes (
							copy Data\Output\Method\ITERATION_%%i\L_TST_A.xes Results\Method\%%a\%%x\%%c\ITERATION_%%j\EventLogs >nul
						)
						if exist Data\Output\Method\ITERATION_%%i\L_TST.xes (
							copy Data\Output\Method\ITERATION_%%i\L_TST.xes Results\Method\%%a\%%x\%%c\ITERATION_%%j\EventLogs >nul
							ren Results\Method\%%a\%%x\%%c\ITERATION_%%j\EventLogs\L_TST.xes L_TST_%%i.xes
						)
					)
				)
			)

			

			:: === Prepare input Petri nets for online adaptation ===
			del /F /Q Input\OA\PetriNets\*
			copy Results\Method\%%a\%%x\ITERATION_0\PetriNets\PN_0.pnml Input\OA\PetriNets >nul
			
			for %%y in (%iterations%) do (
				
				del /F /Q Input\OA\Data\*
				copy Data\Output\Method\ITERATION_%%y\L_TR.xes Input\OA\Data
			
				call online_adaptation %use_baseline% %%x %%y %%c
				
				copy Output\OA\PetriNets\* Input\OA\PetriNets
				
				copy Output\OA\PetriNets\* Results\Method\%%a\%%x\%%c\ITERATION_%%y\PetriNets
				copy Output\OA\Diagnoses\* Results\Method\%%a\%%x\%%c\ITERATION_%%y\Diagnoses
				copy Output\OA\Timing\* Results\Method\%%a\%%x\%%c\ITERATION_%%y\Timing
				
			)
		)
	)
)





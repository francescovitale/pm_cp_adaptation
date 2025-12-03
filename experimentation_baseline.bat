:: Options:
:: anomaly_type=[V, W]; one anomaly type at a time
:: pd_variant=[im, ilp, hm]
:: n_clusters=<integer>
:: n_simulation_traces=<integer>

set pd_variant=im ilp hm
set n_reps=3
set iterations=1 2 3 4
set use_baseline=1

for /D %%p IN ("Results\Baseline\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /l %%a in (1, 1, %n_reps%) do (

	mkdir Results\Baseline\%%a
	
	for %%x in (%pd_variant%) do (
	
		mkdir Results\Baseline\%%a\%%x\ITERATION_0
		mkdir Results\Baseline\%%a\%%x\ITERATION_0\PetriNet
		mkdir Results\Baseline\%%a\%%x\ITERATION_0\EventLogs
	
		copy Data\Output\Baseline\ITERATION_0\L_TR.xes Results\Baseline\%%a\%%x\ITERATION_0\EventLogs
		ren Results\Baseline\%%a\%%x\ITERATION_0\EventLogs\L_TR.xes L_TR_0.xes
		copy Data\Output\Baseline\ITERATION_0\L_TST_N.xes Results\Baseline\%%a\%%x\ITERATION_0\EventLogs
		ren Results\Baseline\%%a\%%x\ITERATION_0\EventLogs\L_TST_N.xes L_TST_N_0.xes
		copy Data\Output\Baseline\ITERATION_0\L_TST_A.xes Results\Baseline\%%a\%%x\ITERATION_0\EventLogs
		
		copy Data\Output\Baseline\ITERATION_0\L_TR.xes Input\T\Data
	
		call training %%x
		
		copy Output\T\PetriNet\PN.pnml Results\Baseline\%%a\%%x\ITERATION_0\PetriNet
		ren Results\Baseline\%%a\%%x\ITERATION_0\PetriNet\PN.pnml PN_0.pnml
		
		mkdir Results\Baseline\%%a\%%x
		
		for %%y in (%iterations%) do (
			mkdir Results\Baseline\%%a\%%x\ITERATION_%%y
			mkdir Results\Baseline\%%a\%%x\ITERATION_%%y\PetriNet
			mkdir Results\Baseline\%%a\%%x\ITERATION_%%y\EventLogs
			mkdir Results\Baseline\%%a\%%x\ITERATION_%%y\Timing
		)

		:: === Copy test event logs consistently ===
		for %%i in (0 1 2 3 4) do (
			for /l %%j in (%%i,1,4) do (
				if exist Data\Output\Baseline\ITERATION_%%i\L_TST_A.xes (
					copy Data\Output\Baseline\ITERATION_%%i\L_TST_A.xes Results\Baseline\%%a\%%x\ITERATION_%%j\EventLogs >nul
				)
				if %%j geq %%i (
					if exist Data\Output\Baseline\ITERATION_%%i\L_TST_N.xes (
						copy Data\Output\Baseline\ITERATION_%%i\L_TST_N.xes Results\Baseline\%%a\%%x\ITERATION_%%j\EventLogs >nul
						ren Results\Baseline\%%a\%%x\ITERATION_%%j\EventLogs\L_TST_N.xes L_TST_N_%%i.xes
					)
					if exist Data\Output\Baseline\ITERATION_%%i\L_TST.xes (
						copy Data\Output\Baseline\ITERATION_%%i\L_TST.xes Results\Baseline\%%a\%%x\ITERATION_%%j\EventLogs >nul
						ren Results\Baseline\%%a\%%x\ITERATION_%%j\EventLogs\L_TST.xes L_TST_%%i.xes
					)
				)
			)
		)

		:: === Prepare input Petri nets for online adaptation ===
		del /F /Q Input\OA\PetriNets\*
		copy Results\Baseline\%%a\%%x\ITERATION_0\PetriNet\PN_0.pnml Input\OA\PetriNets >nul
			
		for %%y in (%iterations%) do (
			del /F /Q Input\OA\Data\*
			copy Data\Output\Baseline\ITERATION_%%y\L_TR.xes Input\OA\Data
			
			copy Results\Baseline\%%a\%%x\ITERATION_0\PetriNet\* Input\OA\PetriNets
			
			call online_adaptation %use_baseline% %%x %%y
				
			del /F /Q Input\OA\PetriNets\*
			copy Output\OA\PetriNets\* Input\OA\PetriNets
			copy Output\OA\PetriNets\* Results\Baseline\%%a\%%x\ITERATION_%%y\PetriNet
			copy Output\OA\Timing\* Results\Baseline\%%a\%%x\ITERATION_%%y\Timing
				
		)	
	)
)





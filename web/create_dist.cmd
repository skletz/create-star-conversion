:: creates a distribution folder with all needed files to distribute the game
@echo off
:: clean directory
echo ################## Cleaning up... #######################
if exist dist rd /s /q dist 
mkdir dist
mkdir dist\resources\lib\phaser
mkdir dist\js
echo ################## Copying files... #####################
:: copy assts/css/needed files
xcopy assets dist\assets\ /s /e /Y
xcopy css dist\css\ /s /e /Y
copy resources\lib\phaser\phaser.min.js dist\resources\lib\phaser
copy resources\lib\phaser\phaser.map dist\resources\lib\phaser

echo ################## Modifying index.html ##################
:: modify index.html (replace js files by game.min.js) 
:: important enable delayed expansion so writeline can be set in loop (use !var! to check)
setlocal enabledelayedexpansion
set writeline=true
for /f "tokens=*" %%a in (index.html) do (	

		:: check if html annotation for following js files is found
		if %%a == ^<^!--^ START_APP_JS^ --^> (
			:: append to new dist\index.html
			echo ^<script^ src="resources/lib/phaser/phaser.min.js"^>^</script^> >> dist\index.html
			echo ^<script^ src="js/game.min.js"^>^</script^>  >> dist\index.html			
			set writeline=false
		)
		
		:: append to new dist\index.html (check for doctype and add '!' -> removed somehow due to delayed expansions)
		if !writeline! == true (
			if %%a == ^<DOCTYPE^ html^> (
				echo ^<^^!DOCTYPE^ html^> >> dist\index.html
			)	else	 (
				echo %%a >> dist\index.html
			)
		)
		if %%a == ^<^!--^ END_APP_JS^ --^> (
			set writeline=true
		)
)

echo ################## Minifying js files ####################
:: Packs and minifies all js files into dist\js\game.min.js
java -jar "tools\closure\compiler.jar" ^
       --js js\generated\config.js ^
       --js js\generated\Utils\utils.js ^
       --js js\generated\Utils\Timer.js ^
	   --js js\generated\Entities\Character.js ^
	   --js js\generated\States\Title.js ^
	   --js js\generated\States\Menu.js ^
	   --js js\generated\States\Game.js ^
	   --js js\generated\States\End.js ^
	   --js js\generated\app.js ^
	   --js_output_file dist\js\game.min.js
echo ################## Finished! ##############################

@echo off
:: Applies the Android network security config patch.
:: Run this from the mobile\ directory AFTER: npx cap add android

set DEST=..\android\app\src\main\res\xml
if not exist "%DEST%" mkdir "%DEST%"
copy /Y "res\xml\network_security_config.xml" "%DEST%\network_security_config.xml"
echo Copied network_security_config.xml to %DEST%

echo.
echo Next step: open android\app\src\main\AndroidManifest.xml
echo Add to the ^<application^> tag:
echo     android:networkSecurityConfig="@xml/network_security_config"
echo.
pause

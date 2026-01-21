@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo 用法:
  echo   my.bat git "commit message"   ^> 执行 git 提交与推送
  echo   my.bat update                 ^> 扫描 doc 并更新配置
  echo   my.bat server                 ^> 启动本地服务器
  exit /b 1
)

set ACTION=%~1
shift

if /i "%ACTION%"=="git" (
  set MSG=%~1
  if "!MSG!"=="" set MSG=update

  @REM rem 先自动执行 update
  @REM call :do_update
  @REM if errorlevel 1 (
  @REM   echo 更新脚本执行失败，终止 git 提交
  @REM   exit /b 1
  @REM )

  git add .
  git commit -m "!MSG!"
  if errorlevel 1 (
    echo Git 提交失败
    exit /b 1
  )
  git push
  exit /b
)

if /i "%ACTION%"=="update" (
  python ".\update.py"
  if errorlevel 1 (
    echo 更新脚本执行失败
    exit /b 1
  )
  exit /b
)

if /i "%ACTION%"=="server" (

  rem 先自动执行 update
  call :do_update
  if errorlevel 1 (
    echo 更新脚本执行失败，终止 git 提交
    exit /b 1
  )
  teedoc serve
  exit /b
)

echo 未知动作: %ACTION%
exit /b 1

:do_update
python ".\update.py"
if errorlevel 1 exit /b 1
exit /b 0
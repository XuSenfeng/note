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
  python "e:\note\test.py"
  exit /b
)

if /i "%ACTION%"=="server" (
  teedoc serve
  exit /b
)

echo 未知动作: %ACTION%
exit /b 1
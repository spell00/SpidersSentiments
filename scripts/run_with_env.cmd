@echo off
set TWITTER_AUTH_COOKIE=%1
set LANGSMITH_API_KEY=%2
shift
shift
"%~dp0..\\.conda\\python.exe" %*

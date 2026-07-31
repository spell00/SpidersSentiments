param(
    [string]$TwitterAuthCookie,
    [string]$LangsmithApiKey,
    [string]$PythonExe,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$PythonArgs
)

if ($TwitterAuthCookie) {
    $env:TWITTER_AUTH_COOKIE = $TwitterAuthCookie
}
if ($LangsmithApiKey) {
    $env:LANGSMITH_API_KEY = $LangsmithApiKey
}

& $PythonExe @PythonArgs

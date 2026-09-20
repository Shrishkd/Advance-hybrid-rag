# Start Ollama with models on D:, not C:.
#
# WHY THIS SCRIPT EXISTS
# A newly spawned process inherits the environment its PARENT had. The
# OLLAMA_MODELS User variable is set, but a shell started before that (or a
# tray app launched at the previous login) still carries the old environment
# and will happily recreate an empty C:\Users\<you>\.ollama\models, leaving
# `ollama list` blank while ~4 GB of models sit on D: untouched.
#
# Setting it explicitly in this process before launching removes the guesswork.
# After a login where the tray app picks up the User variable, this is no
# longer needed.
$env:OLLAMA_MODELS = 'D:\ollama-models'
Start-Process -FilePath 'ollama.exe' -ArgumentList 'serve' -WindowStyle Hidden
Start-Sleep -Seconds 5
Write-Host "OLLAMA_MODELS = $env:OLLAMA_MODELS"
ollama list

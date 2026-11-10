# F1 predictor

Bot Telegram che stima le probabilita' di vittoria dei piloti di F1 (dati dal 2014).

Comandi: /classificagara · /classificaqualifiche · /penalita · /aiuto
Notifiche automatiche: solo venerdi', sabato, domenica (30 min prima di ogni sessione, dopo le qualifiche, esito gara).

Secrets GitHub richiesti: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID.
Prima configurazione: Actions -> operazioni-manuali -> azione "bootstrap".
Facoltativo: azione "storico-libere" (ripetila finche' non dice "Storico libere completo"), poi di nuovo "bootstrap".
Quote (facoltative): odds.csv con colonne season,round,code,odds.
Test locale dei calcoli: python -m tests.smoke_test && python -m tests.logic_test

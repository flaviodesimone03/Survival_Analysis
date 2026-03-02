# Repository Tesi — Analisi di Sopravvivenza e Modelli Predittivi

Questa repository contiene tutti i file utilizzati per la ricerca di tesi, inclusi:
- i codici per creare i modelli impiegati nello studio;
- il dataset finale utilizzato per eseguire le analisi e validare i modelli.

## Struttura della repository

- `data/`  
  Contiene il dataset finale utilizzato nello studio (versione pronta per le analisi e la modellazione).

- `kaplan-meier/`  
  Script e risorse per la stima Kaplan–Meier e la produzione dei relativi grafici/risultati.

- `log-rank test/`  
  Script per l’esecuzione dei test log-rank (confronto tra curve di sopravvivenza per gruppi).

- `cox/`  
  Codice relativo al modello di regressione di Cox, inclusi eventuali step di fitting e valutazione.

- `bayesian-network/`  
  Codice per la rete bayesiana, utile per modellare dipendenze tra variabili cliniche e per interrogazioni probabilistiche condizionate.

## Obiettivo

L’obiettivo di questa repository è rendere tracciabili e riproducibili:
- il workflow di analisi (KM, log-rank, Cox);
- la modellazione probabilistica tramite rete bayesiana;
- l’utilizzo del dataset finale su cui sono stati prodotti risultati e modelli della tesi.

## Note operative

- Gli script possono essere in Python per tutti i modelli tranne la rete bayesiana che è scritto in R.
- I requisiti software/dipendenze dipendono dagli script presenti.

## Autore
Flavio de Simone

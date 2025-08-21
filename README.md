# SIR Epidemic Simulator (GUI)

Un semplice progetto Python che simula un modello epidemiologico **SIR** su una griglia 2D,
con un'interfaccia grafica per:
- visualizzare la **matrice** (sani/infetti/recuperati) al tempo *t*,
- **slider del tempo** per navigare avanti/indietro nella simulazione,
- **slider densità** per impostare la percentuale di popolazione iniziale,
- **slider contagiosità** (β) per impostare la probabilità di infezione per contatto.

Il recupero è modellato con un **tempo di infettività** (γ⁻¹) fissato in passi discreti.

## Requisiti
- Python 3.9+
- pacchetti: `numpy`, `matplotlib`

( Tkinter è nella libreria standard su Windows/macOS/Linux. )

```
pip install -r requirements.txt
```

## Avvio
```
python main.py
```

## Note sul modello
- Griglia N×N; stati:
  - 0 = vuoto (cella priva di individuo),
  - 1 = S (suscettibile),
  - 2..(2+T_inf-1) = I con **timer** di infezione,
  - 3 = R (rimossi/recuperati, immuni).
- Aggiornamento sincrono a passi discreti (von Neumann o Moore nei parametri, di default **Moore** 8-vicini).
- La probabilità che un suscettibile con *n* vicini infetti si ammali in un passo è:

  **q = 1 − (1 − β)ⁿ**

  (processi indipendenti per ciascun vicino).

Cambiare **densità** o **β** pulisce la storia memorizzata per garantire coerenza del *time scrubber*.# epidemic_models

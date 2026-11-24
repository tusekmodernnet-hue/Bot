# Prerejestracja hipotez — krok 3

Data rejestracji: 2026-11-24. Ten plik powstał **przed** pierwszym testem którejkolwiek strategii. Commit z tym
plikiem jest dowodem, że definicje i siatki parametrów nie zostały dopasowane do wyników. Każda późniejsza
zmiana wymaga nowego wpisu w sekcji „Zmiany” z uzasadnieniem i liczy się jako nowe próby w rejestrze.

## Wspólne zasady

- **Dane:** okres badawczy 2020-01-01 – 2025-05-31. Holdout (od 2025-06-01) pozostaje zablokowany.
- **Uniwersum:** point-in-time top 50 kontraktów USDT według średniego dziennego obrotu z 30 dni. Kontrakt musi
  mieć co najmniej 90 dni notowań i co najmniej 20 mln USDT dziennego obrotu (`UniverseRules()`).
- **Decyzje:** strategie dzienne decydują na zamknięciu dnia UTC i wykonują transakcję na otwarciu kolejnej
  świecy. Strategie godzinowe decydują na zamknięciu godziny.
- **Koszty:** prowizja taker 0,05% za stronę. Slippage wynosi 2 pb dla BTC i ETH oraz 10 pb dla pozostałych
  kontraktów. Funding jest rzeczywisty. Test warunków skrajnych liczy koszty ×2.
- **Wielkość pozycji:** każda pozycja ma docelową zmienność 15% rocznie / √(liczba pozycji), liczoną z
  odchylenia dziennych zwrotów z 30 dni. Portfel ma limit dźwigni brutto 2×, maksymalnie 1× na pozycję
  w BTC i ETH oraz 0,25× na pozycję w pozostałych kontraktach.
- **Benchmarki:** `BENCH/buy_and_hold` (1× BTC, rebalansowane dziennie) oraz `BENCH/trend_sma200`.
- **Ocena:** walk-forward z 24 miesiącami treningu i 3 miesiącami testu, przesuwany co 3 miesiące. W oknie
  treningowym wybierany jest wariant o najwyższym Sharpe, a wynik out-of-sample to sklejone okna testowe.
  Dla każdej rodziny liczę PBO (CSCV, 16 bloków) na wszystkich wariantach oraz Deflated Sharpe z liczbą prób
  z rejestru.

## Rodziny hipotez i siatki parametrów

### H1 — trend czasowy (time-series momentum)
- Sygnał: znak zwrotu z ostatnich L dni. Pozycja long przy zwrocie > 0, short (lub brak pozycji) przy ≤ 0.
- L ∈ {20, 40, 60, 90, 120} dni; tryb ∈ {long/short, tylko long}. **10 wariantów.**
- Rynki: BTC i ETH razem oraz osobno całe uniwersum (walidacja krzyżowa).

### H2 — momentum i odwrócenie przekrojowe
- H2a, momentum: ranking zwrotu z L dni z pominięciem ostatniego dnia, L ∈ {7, 14, 30, 60, 90}. Long górny
  kwintyl, short dolny kwintyl, rebalans co 7 dni. Neutralność ∈ {dolarowa, beta do BTC}. **10 wariantów.**
- H2b, krótkoterminowe odwrócenie: ranking zwrotu z L ∈ {1, 3} dni, odwrotnie do momentum (long przegrani,
  short zwycięzcy), rebalans dzienny, neutralność dolarowa. **2 warianty.**

### H3 — tłok w pozycjach (funding i open interest)
- H3a, przekrojowo: ranking średniego fundingu z W dni, W ∈ {1, 3, 7, 14}. Long dolny kwintyl (najtańszy
  long), short górny kwintyl, rebalans dzienny, neutralność dolarowa. **4 warianty.**
- H3b, BTC i ETH w czasie: short, gdy z-score fundingu z 30 dni > Z, a OI wzrósł o więcej niż 10% w 3 dni.
  Long przy warunkach lustrzanych. Wyjście po H dniach. Z ∈ {1,5; 2,0}, H ∈ {3, 7}. **4 warianty.**
  OI z metryk 5-minutowych z opóźnieniem 5 minut.

### H4 — carry (premia za funding)
- Long spot i short perpetual na BTC i ETH, gdy średni funding z 3 dni w ujęciu rocznym > T. Wyjście, gdy
  spadnie poniżej T/2. T ∈ {5%, 10%, 20%}. **3 warianty.** Koszty liczone na dwóch nogach. Wymaga danych spot.

### H5 — reżim zmienności
- Nakładka na H1–H3, nie samodzielna strategia. Ekspozycja skalowana odwrotnie do zmienności z 30 dni
  albo stała. Porównanie wariantu z nakładką i bez niej. **2 warianty na każdą rodzinę bazową.**

### H6 — fałszywe wybicia (świece godzinowe)
- Cena przebija maksimum (minimum) z N godzin, a w ciągu M świec zamyka się z powrotem wewnątrz zakresu.
  Wtedy wchodzę w przeciwnym kierunku. Stop za ekstremum wybicia, wyjście na środku zakresu albo po 24 h.
- N ∈ {24, 72, 168}, M ∈ {1, 3}. **6 wariantów.** Rynki: BTC, ETH i top 20 uniwersum.

### H7 — filtry zewnętrzne
- Bramka nakładana na najlepszą rodzinę z H1–H6: Fear & Greed (skrajne wartości < 20 lub > 80), MVRV
  (powyżej 3,0) i VIX (powyżej 30). Każdy filtr działa osobno, z opóźnieniem 1 dnia. **3 warianty.**

### ML — meta-model
- Gradient boosting (LightGBM) łączący sygnały H1–H7 oraz zmienność i funding jako cechy. Etykieta to znak
  zwrotu z kolejnych 5 dni. Walk-forward z purgingiem 5 dni i embargo 5 dni.
- Model zostaje tylko wtedy, gdy out-of-sample pobije równoważoną kombinację sygnałów (risk parity).
  **Uruchamiany dopiero po H1–H7.**

## Kryteria akceptacji

Zgodnie z dokumentem z kroku 1: Sharpe OOS ≥ 1,0, DSR ≥ 0,95, PBO ≤ 0,25, maksymalne obsunięcie ≤ 25%, wynik
dodatni przy kosztach ×2, stabilność między reżimami i lepszy wynik niż oba benchmarki.

## Zmiany

- **2026-11-24, przed pierwszym testem: szczegóły wykonania, które nie są strojone.**
  - Strategie dzienne mają pasmo rebalansu 2% kapitału, żeby nie płacić za mikrokorekty wynikające z dryfu wag.
  - Strategie przekrojowe wymagają co najmniej 10 kontraktów z sygnałem. Przy mniejszej liczbie pozycji nie ma.
  - Strategie dzienne symuluję na świecach dziennych: decyzja na zamknięciu dnia UTC, wykonanie na otwarciu
    następnego.
  - Walk-forward pomija koszt przełączenia między wariantami przy zmianie wyboru. Ten koszt jest mały
    w porównaniu z dziennym rebalansem, ale go odnotowuję.
  - Reżimy rynku wyznacza BTC: hossa przy zwrocie z 90 dni > +20%, bessa przy < −20%, w pozostałych
    przypadkach konsolidacja. Wysoka zmienność to zrealizowana zmienność 30-dniowa BTC > 80% rocznie.
- **2026-11-24, przed testami H3b–H7: doprecyzowanie bez strojenia.**
  - H4: wielkość 1× brutto na parę spot/perp, dzielona równo między BTC i ETH. Prowizja spot jest wyższa niż
    na futures, więc noga spot ma +5 pb poślizgu (łącznie ok. 0,12% za stronę).
  - H5: wariant „bez nakładki” to cała siatka rodziny bazowej z wyłączonym celowaniem zmienności portfela
    (stała ekspozycja brutto 1×). Rodziny bazowe: H1 (BTC+ETH), H2a i H3a.
  - H7: bramka zeruje ekspozycję w dniu aktywnym. Nakładam ją na każdy z 10 wariantów H1 (BTC+ETH), czyli
    rodziny o najlepszym wyniku OOS z H1–H6, i ponownie przeprowadzam walk-forward. To 30 prób.

- **2026-11-24, przed testem H6 na top 20: szczegóły wykonania, bez zmiany siatki.**
  - Wynik H6 na BTC+ETH jest już znany (Sharpe OOS −1,11). Ten test wykonuję, bo był prerejestrowany. Jego 6 prób
    dopisuję do rodziny H6 (razem 12), więc DSR liczy obie części.
  - Uniwersum: point-in-time top 20 według średniego obrotu z 30 dni (`UniverseRules(top_n=20)`), z kontraktami
    zdjętymi z giełdy. Przynależność z zamknięcia dnia d pozwala otwierać transakcje od dnia d+1. Otwarta transakcja
    biegnie do swojego wyjścia, także gdy kontrakt wypadnie z top 20.
  - Limity pozycji jak we wspólnych zasadach: 1× na BTC i ETH, 0,25× na pozostałe kontrakty. Limit brutto 2×:
    nowe wejścia są proporcjonalnie zmniejszane w chwili otwarcia, otwartych transakcji nie tnę.
- **2026-11-24, przed testem meta-modelu ML: pełna specyfikacja, bez strojenia.** Wyniki H1–H7 są znane; model
  ma stałe hiperparametry i jedną wersję, więc nie ma czego dopasować do tych wyników.
  - Rynek: BTC i ETH, jeden model dla obu kontraktów. Decyzja na zamknięciu dnia, wykonanie na otwarciu następnego.
  - Cechy, wszystkie znane na zamknięciu dnia t: zwroty z L ∈ {1, 3, 20, 40, 60, 90, 120} dni (H1, H2b);
    percentyl zwrotu z 30 dni z pominięciem ostatniego dnia w przekroju uniwersum top 50 (H2a); suma fundingu
    z W ∈ {1, 3, 7, 14} dni i jego z-score z 30 dni (H3, H4); percentyl 14-dniowego fundingu w przekroju (H3a);
    zmiana OI z 3 dni z opóźnieniem 5 minut (H3b); zmienność zrealizowana z 30 dni (H5); suma kierunków wejść H6
    (N = 24, M = 1) z godzin dnia t (H6); Fear & Greed, MVRV i VIX z opóźnieniem 1 dnia (H7).
  - Etykieta: 1, gdy zwrot od otwarcia t+1 do otwarcia t+6 jest dodatni, w przeciwnym razie 0.
  - Model: LightGBM (binary) ze stałymi parametrami: 200 drzew, learning rate 0,05, 15 liści, minimum 50
    obserwacji w liściu, bagging 0,8 co iterację, feature fraction 0,8, seed 0.
  - Walk-forward: te same okna co rodziny (24 miesiące treningu, 3 miesiące testu, pierwszy test od 01.2022).
    Purging: w treningu zostają tylko dni, których etykieta jest znana przed początkiem testu (t + 6 ≤ start).
    Embargo: dodatkowe 5 dni przerwy przed testem.
  - Pozycja: long przy p > 0,5, w przeciwnym razie short, z tym samym rozmiarem co H1 (cel zmienności 15%,
    limit 1× na pozycję i 2× brutto) i pasmem rebalansu 2%.
  - Punkt odniesienia, czyli równoważona kombinacja sygnałów: znak średniej z sześciu kierunków o równej wadze:
    H1 (średnia znaków zwrotów z 20, 40, 60, 90 i 120 dni), H2a (+1 w górnym kwintylu momentum, −1 w dolnym),
    H2b (minus znak zwrotu z 3 dni), H3a (−1 w górnym kwintylu fundingu, +1 w dolnym), H3b (minus znak z-score
    fundingu, gdy |z| > 1,5) i H6 (znak sumy wejść z dnia). Rozmiar pozycji jak w modelu. H4 i H7 nie dają
    kierunku dla BTC i ETH, więc są tylko cechami modelu.
  - Rodzina „ML”, 2 próby. DSR liczę z liczbą wszystkich prób w rejestrze, bo model korzysta z całego programu
    badań. Model zostaje tylko wtedy, gdy spełni kryteria akceptacji i pobije punkt odniesienia out-of-sample.

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
  - Doprecyzowanie przed uruchomieniem: cel zmienności portfela potrzebuje 60 dni własnych pozycji strategii.
    Dlatego model i punkt odniesienia dostają sygnały od początku pierwszego okna testowego, przez pierwsze 60 dni
    są płaskie, a wynik OOS obu liczę od 61. dnia. Obie strategie mają identyczne okno oceny.

## Runda 2 (prerejestracja 2026-11-25, przed jakimkolwiek testem tej rundy)

Runda 1 (111 prób) nie dała kandydata. Właściciel wybrał jedną, ostatnią rundę 2 z najwyżej trzema hipotezami.
Dwie pierwsze wynikają z obserwacji rundy 1, więc są obciążone wiedzą o tych danych. Dlatego kryterium DSR
w tej rundzie liczę z liczbą **wszystkich** prób w rejestrze (cały program badań), a nie tylko danej rodziny.
Holdout pozostaje zablokowany. Jeśli runda 2 nie da kandydata, kończę szukanie strategii kierunkowych na tych
danych. Wspólne zasady (koszty, rozmiar pozycji, walk-forward 24/3, PBO, reżimy, koszty ×2) bez zmian.

### R2A — carry powyżej stopy wolnej od ryzyka
- Uzasadnienie: carry ma sens tylko wtedy, gdy daje więcej niż bony skarbowe przy tym samym kapitale. H4 tego
  nie sprawdzał i w OOS zarobił mniej niż bony.
- Long spot i short perpetual na BTC i ETH (konstrukcja, wielkość i koszty jak w H4), gdy średni funding z 3 dni
  w ujęciu rocznym przewyższa rentowność 3-miesięcznych bonów USA (FRED DTB3, opóźnienie 1 dnia) o więcej niż M.
  Wyjście, gdy nadwyżka spadnie poniżej M/2. M ∈ {2,5%; 5%; 10%}. **3 warianty.**
- Ocena na zwrocie ponad stopę wolną od ryzyka: od dziennego zwrotu odejmuję rentowność bonów naliczoną na
  kapitale zaangażowanym w nogę spot (wolny kapitał leży w bonach). Wszystkie kryteria akceptacji, łącznie
  z walk-forward i PBO, liczę na tej nadwyżce. Założenie: noga spot służy jako zabezpieczenie shorta perpetual.

### R2B — funding przekrojowo z niskim obrotem
- Uzasadnienie: H3a otrzymał funding równy 134% kapitału, ale koszty obrotu (93× rocznie) zjadły wynik.
- Ranking średniego fundingu z W dni, long dolny kwintyl, short górny kwintyl, neutralność dolarowa, rebalans
  co R dni (zamiast codziennie), wyjścia wymuszone przez uniwersum natychmiast. W ∈ {7, 14}, R ∈ {7, 14}.
  **4 warianty.**

### R2C — premia za ryzyko zmienności z rynku opcji
- Uzasadnienie: gdy zmienność implikowana jest wysoka względem zrealizowanej, rynek płaci za przejęcie ryzyka,
  co w literaturze dla akcji zapowiada wyższe zwroty (Bollerslev, Tauchen, Zhou 2009). Dla kryptowalut nie testowałem.
- VRP = DVOL z Deribit (30-dniowa zmienność implikowana, zamknięcie dnia, opóźnienie 1 dnia jak dla danych spoza
  giełdy wykonania) minus zmienność zrealizowana z 30 dni z dziennych zwrotów perpetual, oba w ujęciu rocznym.
  Sygnał: z-score VRP z 90 dni. Long, gdy z > Z; w trybie long/short także short, gdy z < −Z; w pozostałych
  dniach brak pozycji. Rozmiar jak w H1, rebalans dzienny z pasmem 2%. Z ∈ {0; 1}, tryb ∈ {tylko long,
  long/short}. **4 warianty.** Rynki: BTC i ETH.
- DVOL istnieje od 2021-03-24, więc okna walk-forward liczę od 2021-07-01: pierwszy test od 2023-07-01, OOS
  około 23 miesięcy. Benchmarki porównuję w tym samym oknie.

Razem 11 prób (rodziny R2A, R2B, R2C).

## Runda 3 (prerejestracja 2026-11-25, przed jakimkolwiek testem tej rundy)

Właściciel 2026-11-25 cofnął decyzję o zamknięciu i polecił szukać dalej: scalping, day trading, swing, SMC/ICT
i ich kombinacje. „Do skutku” nie zmienia progów: każda runda jest prerejestrowana, wszystkie próby trafiają
do rejestru, a holdout (od 2025-06-01) otwieram jeden raz, tylko dla kandydata, który przejdzie wszystkie kryteria.

### Wspólne zasady rundy 3
- Rynki: BTCUSDT i ETHUSDT perpetual. Koszty: prowizja taker 0,05% i poślizg 2 pb na stronę, funding rzeczywisty.
- Świece: 15m z archiwum, 1h z archiwum, 4h agregowane z 1h (granice 00, 04, …, 20 UTC), dzień UTC.
- Wykonanie: decyzja na zamknięciu świecy, wejście na otwarciu następnej. Stop działa w trakcie świecy (po cenie
  stopu albo po gorszym otwarciu przy luce). Cel i wyjście czasowe sprawdzam na zamknięciu świecy, wyjście
  następuje na otwarciu kolejnej. Jedna transakcja naraz na kontrakt; sygnały w trakcie transakcji są ignorowane,
  chyba że wariant odwraca pozycję (R3D).
- Rozmiar: ryzyko 0,5% kapitału do stopu, najwyżej 1× na kontrakt (razem najwyżej 2×). Strategie bez naturalnego
  stopu mają stop ochronny 3× ATR(14): z interwału 1h dla strategii na 15m, z własnego interwału dla 1h i 4h.
- ATR(14): średni true range 14 ostatnich świec, łącznie z bieżącą. Cel „2R” = cena decyzji ± 2 × (cena decyzji − stop).
- Sesje w UTC: Azja 00:00–06:00, killzone Londynu 07:00–10:00, killzone Nowego Jorku 12:00–15:00. Otwarcie giełdy
  w Nowym Jorku: 09:30 czasu nowojorskiego (z uwzględnieniem zmiany czasu).
- Ocena jak w rundach 1–2: walk-forward 24/3 na dziennych zwrotach, PBO (CSCV, 16 bloków), koszty ×2, reżimy,
  benchmarki. Kryterium stabilności doprecyzowuję: Sharpe OOS ≥ 0 w hossie, bessie i konsolidacji.
- Korekta na wielokrotne testowanie (DSR): N = wszystkie próby w rejestrze z wszystkich rund (bez benchmarków).
  Wariancję Sharpe liczę na próbach rundy 3, czyli w bieżącej przestrzeni poszukiwań. W rundzie 2 liczyłem ją na
  całym rejestrze, ale kilka wariantów carry o dziennym Sharpe około 0,3 zawyża ją tak, że oczekiwany najlepszy
  Sharpe przypadkowej strategii wynosi 4,7 rocznie, czego nie spełni żadna strategia kierunkowa. To przeczy
  założeniu DSR o wspólnym rozkładzie prób. Dla przejrzystości podaję też DSR z wariancją całego rejestru.
  Zmiana nie zmienia żadnego werdyktu rund 1–2: wszystkie odrzucone hipotezy odpadały także na innych kryteriach.

### R3A — sweep maksimum lub minimum poprzedniego dnia (PDH/PDL), 15m
- Short: high świecy > PDH, a ta świeca albo jedna z M−1 kolejnych zamyka się poniżej PDH. Stop: najwyższe high od
  pierwszego przebicia. Long lustrzanie przy PDL. Najwyżej jedna transakcja na stronę dziennie.
- Cel: 2R albo środek zakresu poprzedniego dnia (sygnał pomijam, gdy cel leży po złej stronie ceny). Wyjście
  najpóźniej na zamknięciu dnia UTC.
- Filtr czasu: cały dzień albo tylko gdy świeca powrotu zamyka się w killzone Londynu lub Nowego Jorku.
- M ∈ {1, 4}, cel ∈ {2R, środek dnia poprzedniego}, filtr ∈ {cały dzień, killzones}. **8 wariantów.**

### R3B — sweep zakresu azjatyckiego w killzone Londynu (Judas swing), 15m
- Zakres azjatycki: high i low świec 00:00–05:45 UTC. Między 07:00 a 10:00 UTC przebicie high zakresu i zamknięcie
  z powrotem poniżej w ciągu M świec → short; lustrzanie przy low → long. Stop za ekstremum sweepu.
- Cel: przeciwna krawędź zakresu albo 2R. Wyjście najpóźniej o 16:00 UTC. Jedna transakcja dziennie na kontrakt.
- M ∈ {1, 2}, cel ∈ {przeciwna krawędź, 2R}. **4 warianty.**

### R3C — fair value gap z retestem
- Byczy FVG na świecy t: low[t] > high[t−2], a korpus świecy t−1 przekracza 1× ATR(14) (przemieszczenie). Luka
  jest ważna przez 24 świece. Pierwsza świeca, której low wchodzi w lukę (low ≤ low[t]), a zamknięcie zostaje
  powyżej high[t−2] → long. Stop: niższe z high[t−2] i low świecy retestu. Cel 2R, wyjście czasowe po 24 świecach.
  Niedźwiedzi FVG lustrzanie.
- Filtr: brak albo zgodność z trendem dziennym (long tylko, gdy zamknięcie poprzedniego dnia > SMA(50) dziennej,
  short tylko, gdy < SMA(50)).
- Interwał ∈ {15m, 1h}, filtr ∈ {brak, trend dzienny}. **4 warianty.**

### R3D — struktura rynku: BOS i CHoCH (swing)
- Swing high na świecy i: high[i] jest maksimum świec i−k..i+k; znany dopiero na zamknięciu świecy i+k. Swing low
  lustrzanie.
- Zamknięcie powyżej ostatniego potwierdzonego swing high → long; poniżej ostatniego swing low → short. Przeciwny
  sygnał odwraca pozycję. Stop: ostatni potwierdzony swing po przeciwnej stronie.
- k ∈ {3, 5}, interwał ∈ {1h, 4h}. **4 warianty.**

### R3E — wybicie zakresu otwarcia (ORB), 15m
- Zakres otwarcia: high i low pierwszych N minut sesji. Pierwsze zamknięcie powyżej zakresu → long, poniżej →
  short, jedna transakcja na sesję. Stop: przeciwna krawędź zakresu. Wyjście: koniec dnia UTC (sesja UTC) albo
  16:00 czasu nowojorskiego (sesja NY).
- Sesja ∈ {dzień UTC od 00:00, giełda NY od 09:30}, N ∈ {15, 30, 60}. **6 wariantów.**

### R3F — momentum śróddzienne, 15m
- O godzinie H UTC: znak zwrotu od otwarcia dnia UTC do tej chwili → pozycja w tym kierunku do końca dnia UTC.
  Stop ochronny jak we wspólnych zasadach. H ∈ {12, 16, 20}. **3 warianty.**

### R3G — powrót do VWAP, 15m
- VWAP liczony od 00:00 UTC z ceny typowej (high+low+close)/3. Odchylenie d = close/VWAP − 1, z = d / odchylenie
  standardowe d z poprzednich 30 dni. Między 02:00 a 22:00 UTC: z > K → short, z < −K → long. Wyjście, gdy
  zamknięcie przetnie VWAP, najpóźniej na koniec dnia UTC. Stop ochronny jak we wspólnych zasadach.
- K ∈ {2, 3}. **2 warianty.**

### R3H — korekta w trendzie (swing), 4h
- Trend: zamknięcie poprzedniego dnia > SMA(S) dziennej → tylko long; < SMA(S) → tylko short. Long, gdy RSI(2)
  na 4h < T; short, gdy RSI(2) > 100 − T. Wyjście, gdy RSI(2) > 70 (short: < 30) albo po 30 świecach 4h.
  Stop ochronny jak we wspólnych zasadach.
- T ∈ {5, 10}, S ∈ {50, 200}. **4 warianty.**

Razem 35 prób (rodziny R3A–R3H). Scalping prerejestruję osobno, gdy będzie gotowa symulacja wypełnień zleceń
limit (scalping na zleceniach rynkowych przy koszcie ok. 0,14% za pełną transakcję nie ma sensu ekonomicznego).

### Doprecyzowania definicji rundy 3 (2026-11-25, przed uruchomieniem testów)
Ustalone przy pisaniu kodu, przed jakimkolwiek wynikiem. Nie zmieniają siatki parametrów ani liczby prób.
- Sesja NY (R3E): tylko dni robocze (pn–pt), 09:30–16:00 czasu nowojorskiego z uwzględnieniem zmiany czasu.
  Święta giełdowe w USA nie są wyłączone (brak kalendarza świąt). Brak wejścia na ostatniej świecy sesji.
- Limity transakcji w R3A (jedna na stronę dziennie) i R3B (jedna dziennie) liczę na faktycznie otwartych
  transakcjach. Sygnał pominięty, bo trwa inna transakcja albo cel leży po złej stronie ceny, nie zużywa limitu.
  Gdy na jednej świecy sweep występuje po obu stronach, sygnał pomijam jako niejednoznaczny.
- R3D: BOS/CHoCH to pierwsze zamknięcie za potwierdzonym swingiem, który nie został jeszcze przebity. Każdy
  poziom daje sygnał najwyżej raz; ponowne zamknięcie za tym samym poziomem po wyjściu ze stopu nie jest nowym BOS.
- RSI(2) liczony metodą Wildera. Trend dzienny (R3C, R3H) z dziennych świec archiwum.
- Decyzje z 1h i 4h umieszczam na ostatniej świecy 15m danej świecy 1h/4h; stop ustalony na jej zamknięciu
  obowiązuje na wszystkich świecach 15m następnej świecy 1h/4h. Po stopie strategia jest płaska do następnego
  sygnału. Każda decyzja jest wykonywana (bez progu rebalansowania).
- Przed uruchomieniem zrobiłem test techniczny (bez zapisu do rejestru) na danych z 2021 r., żeby sprawdzić,
  czy każda decyzja strategii trafia do silnika. Wykazał błąd: silnik blokował nowe wejście tuż po stopie, gdy
  miało tę samą wielkość co zatrzymana pozycja. Dodałem do silnika opcję `reenter_after_stop` (domyślnie wyłączoną,
  więc rundy 1–2 się nie zmieniają) i używam jej w rundzie 3. Po teście nie zmieniłem żadnej definicji ani
  parametru. Wyniki tego testu (2021, bez kosztów ×2 i bez walk-forward) nie są wynikami rundy.

### R3X — połączenie rodzin rundy 3 (dopisane 2026-11-25 w trakcie obliczeń rundy 3, przed obejrzeniem jakiegokolwiek wyniku)
- Portfel: średnia z równymi wagami dziennych zwrotów OOS ośmiu rodzin R3A–R3H, każda z wariantem wybranym przez
  jej własny walk-forward. Bez dodatkowych parametrów i bez wyboru rodzin po wynikach. **1 próba.**
- Kryteria jak dla rodzin, z wyjątkiem PBO, którego nie da się policzyć dla jednej strategii: podaję PBO składników.

## Runda 4 — scalping na zleceniach limit (prerejestracja 2026-11-25, przed jakimkolwiek testem tej rundy)

Scalping na zleceniach rynkowych kosztuje ok. 0,14% za pełną transakcję, więc testuję go tylko z wejściem
zleceniem limit (prowizja maker 0,02%). Dane: świece 5m (sygnały) i 1m (symulacja wypełnień) z archiwum Binance,
BTCUSDT i ETHUSDT perpetual, 2020-01 do 2025-05. Holdout bez zmian.

### Symulacja wypełnień (konserwatywna)
- Zlecenie limit wystawiam na zamknięciu świecy 5m z sygnałem; jest aktywne przez podany czas na świecach 1m.
- Kupno limit wypełnia się tylko wtedy, gdy low świecy 1m jest **niżej** niż cena zlecenia (cena przeszła przez
  poziom, samo dotknięcie nie wystarcza); sprzedaż lustrzanie. Wypełnienie po cenie zlecenia, prowizja maker
  0,02%, bez poślizgu. Zlecenia, które w chwili wystawienia byłyby od razu wykonalne, nie są wystawiane.
- Take profit: zlecenie limit (maker, ta sama reguła przejścia ceny), aktywne od świecy 1m po wejściu.
- Stop loss: stop rynkowy, uruchomiony, gdy low ≤ stop (long); wypełnienie po stopie albo po gorszym otwarciu
  przy luce, plus poślizg 2 pb i prowizja taker 0,05%. Wyjście czasowe: na otwarciu następnej świecy 1m, taker
  0,05% + 2 pb.
- Gdy w jednej świecy 1m mieszczą się i take profit, i stop, zakładam stop. W świecy wejścia stop może zadziałać,
  take profit nie.
- Rozmiar: ryzyko 0,5% kapitału do stopu, najwyżej 1× na kontrakt. Jedna pozycja albo jeden zestaw zleceń na
  kontrakt naraz. Funding rzeczywisty za pozycje trzymane w chwili rozliczenia.
- Koszty ×2: maker 0,04%, taker 0,10%, poślizg 4 pb.

### S1 — powrót do VWAP z wejściem limit (5m)
- VWAP od 00:00 UTC z ceny typowej świec 5m. d = close/VWAP − 1; σ = odchylenie standardowe d z poprzednich
  30 dni (świece 5m).
- Na każdym zamknięciu 5m między 02:00 a 22:00 UTC, bez pozycji: kupno limit po VWAP·(1 − K·σ) i sprzedaż limit
  po VWAP·(1 + K·σ), ważne 5 minut. Po wypełnieniu jednej strony druga jest anulowana.
- Take profit: VWAP z chwili wystawienia. Stop: w tej samej odległości od wejścia co take profit, po przeciwnej
  stronie (1:1). Wyjście czasowe po 60 minutach.
- Filtr trendu: brak albo tylko w kierunku trendu 1h (long, gdy ostatnie zamknięte 1h > EMA(50) z 1h; short,
  gdy poniżej).
- K ∈ {2, 3}, filtr ∈ {brak, trend 1h}. **4 warianty.**

### S2 — wejście limit za krawędzią zakresu azjatyckiego w killzone Londynu (Judas swing)
- Zakres azjatycki: high i low świec 00:00–06:00 UTC. A = ATR(14) z świec 1h, znany o 07:00 UTC.
- O 07:00 UTC: sprzedaż limit po high zakresu + x·A i kupno limit po low zakresu − x·A, ważne do 10:00 UTC.
  Najwyżej jedna transakcja dziennie na kontrakt; po wypełnieniu jednej strony druga jest anulowana.
- Take profit: środek zakresu azjatyckiego. Stop: 1·A za ceną wejścia. Wyjście najpóźniej o 16:00 UTC.
- x ∈ {0, 0,5}. **2 warianty.**

### Ocena rundy 4
- Jak w rundzie 3 (walk-forward 24/3, PBO, koszty ×2, reżimy z Sharpe ≥ 0, benchmarki, DD ≤ 25%, Sharpe OOS ≥ 1).
- DSR: N = wszystkie próby w rejestrze; wariancja Sharpe z prób rund 3 i 4 (przestrzeń strategii śróddziennych).
- Kandydat musi dodatkowo przejść paper trading na żywym rynku, bo symulacja na świecach 1m nie oddaje kolejki
  zleceń ani pełnego ryzyka niekorzystnej selekcji wypełnień.

## Runda 5 — przepływ zleceń, likwidacje i otwarcie sesji w Nowym Jorku (prerejestracja 2026-11-25, przed testem)

Właściciel polecił szukać dalej. Rundy 1–4 (164 próby) nie dały kandydata. Runda 5 wykorzystuje informacje, których
dotąd nie testowałem: wolumen kupującego agresora ze świec Binance (`taker_buy_volume`) i open interest co 5 minut
(dostępny 5 minut po znaczniku czasu, BTC od 09.2020, ETH od 12.2021). Dwie rodziny (R5C, R5D) wynikają z diagnostyki
rundy 3 bez kosztów na danych badawczych, więc oznaczam je jako **hipotezy inspirowane danymi**. Holdout tych
obserwacji nie widział i rozstrzyga o każdym kandydacie.

### Wspólne zasady rundy 5
- Rynki BTCUSDT i ETHUSDT perpetual, dane badawcze 2020-01 do 2025-05. Koszty: taker 0,05% + poślizg 2 pb na stronę,
  maker 0,02% (wypełnienia limit według reguł rundy 4), funding rzeczywisty. Ryzyko 0,5% kapitału do stopu, najwyżej
  1× na kontrakt. Decyzja na zamknięciu świecy, wejście rynkowe na otwarciu następnej (jak w rundzie 3).
- Przepływ agresora świecy: I = (2·taker_buy_volume − volume) / volume ∈ [−1, 1]; dla kilku świec sumy w liczniku
  i mianowniku.
- Stop ochronny 3× ATR(14) ze świec 1h, znany na zamknięciu świecy decyzji (jak w rundzie 3).
- Ocena jak w rundzie 3: walk-forward 24/3, PBO, koszty ×2, reżimy (Sharpe ≥ 0 w hossie, bessie i konsolidacji),
  obsunięcie ≤ 25%, Sharpe OOS ≥ 1, lepszy niż benchmarki w tym samym oknie.
- **Zmiana metody DSR (od tej rundy):** SR0 = √(1/T)·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))], gdzie T to liczba dni,
  na których szacuję Sharpe prób (cały okres badawczy), a N to wszystkie próby w rejestrze poza benchmarkami.
  Uzasadnienie: w rundach 3–4 wariancja Sharpe między próbami (zawyżona przez warianty z Sharpe −10, zdominowane przez
  koszty) dawała oczekiwany najlepszy wynik przypadkowej strategii 6 rocznie, czego nie da się osiągnąć. DSR modeluje
  szczęście strategii bez przewagi, a jego wariancja to wariancja estymatora Sharpe, ok. 1/T (Lo 2002). Próg zostaje
  0,95; dla N ≈ 174 oznacza to Sharpe OOS ok. 2 rocznie. Podaję też DSR z wariancją między próbami. Zmiana nie wpływa
  na werdykty rund 1–4: żadna rodzina nie spełniła tam progu Sharpe OOS ≥ 1.

### R5A — kontynuacja przepływu agresora, świece 1h
- F_L = przepływ agresora z ostatnich L świec 1h; z = (F_L − średnia) / odchylenie standardowe F_L z poprzednich
  30 dni (bez bieżącej świecy).
- Bez pozycji: z > 2 → long, z < −2 → short. Wyjście po 4 świecach 1h albo na stopie ochronnym.
- L ∈ {1, 4}. **2 warianty.** Uzasadnienie: poinformowani uczestnicy częściej używają zleceń rynkowych, więc
  nierównowaga agresywnych zleceń może zapowiadać krótkotrwałą kontynuację.

### R5B — odbicie po wymuszonych likwidacjach, świece 15m + OI
- Zwrot świecy r = close/open − 1; σ = odchylenie standardowe zwrotów 15m z poprzednich 30 dni. ΔOI = zmiana OI między
  ostatnimi odczytami dostępnymi na zamknięciu tej i poprzedniej świecy.
- Long (likwidacja longów): r < −k·σ, ΔOI < −1% i I < −0,2. Short (likwidacja shortów): r > k·σ, ΔOI < −1% i I > 0,2.
- Wyjście po 4 godzinach albo na stopie ochronnym. k ∈ {3, 4}. **2 warianty.** Uzasadnienie: wymuszone zamknięcia
  pozycji nie niosą informacji, więc ich wpływ na cenę powinien częściowo się cofnąć.

### R5C — sweep z absorpcją agresora (ICT + przepływ zleceń), 15m, inspirowana danymi
- Sweep jak w R3A z M = 4, bez filtra killzones, cel: środek zakresu poprzedniego dnia, najwyżej jedna transakcja na
  stronę dziennie, wyjście najpóźniej na koniec dnia UTC.
- Dodatkowy warunek: przepływ agresora na świecach sweepu (od pierwszego przebicia do świecy powrotu) po stronie
  przebicia. Short po sweepie PDH tylko, gdy I > θ (agresywni kupujący przebili poziom, ale cena wróciła pod niego);
  long po sweepie PDL tylko, gdy I < −θ.
- θ ∈ {0, 0,1}. **2 warianty.** Inspiracja: R3A (M = 4, bez killzones) miała bez kosztów Sharpe ok. 1,0.

### R5D — wybicie zakresu otwarcia sesji NY z tańszym wykonaniem i filtrem zmienności, inspirowana danymi
- Sesja i zakres jak w R3E (dni robocze, od 09:30 czasu nowojorskiego, pierwsze N minut), jedna transakcja na sesję,
  stop na przeciwnej krawędzi zakresu, wyjście o 16:00 czasu nowojorskiego.
- Filtr: sesję handluję tylko wtedy, gdy ATR(14) ze świec 1h podzielony przez cenę, znany o 09:30, jest nie niższy niż
  mediana z poprzednich 60 sesji (większy oczekiwany ruch przy stałym koszcie w punktach bazowych).
- Wykonanie: rynkowe (jak R3E) albo limit: po pierwszym zamknięciu świecy 15m za zakresem zlecenie limit na krawędzi
  zakresu (kupno na górnej, sprzedaż na dolnej), ważne do 12:00 czasu nowojorskiego, symulowane na świecach 1m
  według reguł rundy 4; stop i wyjście o 16:00 po cenie rynkowej.
- N ∈ {15, 30}, wykonanie ∈ {rynkowe, limit}. **4 warianty.** Inspiracja: ORB sesji NY w R3E miał bez kosztów
  Sharpe 1,5–1,9, a po kosztach 0,0–0,3.

Razem 10 prób (N w programie wzrośnie do ok. 174). Jeśli runda 5 nie da kandydata, kończę reguły na świecach z
archiwum Binance i rekomenduję nowe źródło danych (płatna księga zleceń) albo inny rynek.

### Doprecyzowania rundy 5 przed uruchomieniem (2026-11-25, przed jakimkolwiek wynikiem)
- Open interest: rekordy z zerowym OI w archiwum Binance (463 dla BTC, 198 dla ETH, m.in. 7–8.03.2022 dla obu
  kontraktów) to puste odczyty, nie stan rynku. Pomijam je, jak pominąłby je system działający na żywo, i używam
  ostatniego niezerowego odczytu dostępnego na zamknięciu świecy.
- T w nowej metodzie DSR = 1978 dni (01.01.2020–31.05.2025): na tylu dniach liczony jest Sharpe każdej próby w rejestrze.
- PBO: podział CSCV, w którym żaden wariant nie ma zwrotów w części in-sample, pomijam, bo nie ma w nim zwycięzcy do
  oceny (R5B handluje rzadko). W rundach 1–4 taki podział nie wystąpił (obliczenie by się przerwało), więc ich wyniki
  się nie zmieniają.
- R5A liczy sygnał na świecach 1h, a silnik wykonuje decyzje na świecach 15m (jak R3D na 1h): wejście na otwarciu
  następnej godziny, stop sprawdzany na świecach 15m.
- R5D z limitem: zlecenie aktywne od pierwszej świecy 1m po zamknięciu świecy 15m z sygnałem do świecy 11:59 czasu
  nowojorskiego włącznie; wyjście czasowe na otwarciu świecy 16:00.
- Przed rundą 5 rejestr ma 164 próby (bez benchmarków), po niej 174.

### Erratum (2026-11-25, po rundzie 5)
- Uzasadnienie zmiany metody DSR było nieprecyzyjne: progu Sharpe OOS ≥ 1 nie spełniła żadna rodzina rund 3–4, ale
  w rundach 1–2 spełniły go H4 (3,69) i R2A (3,04). Werdykty się nie zmieniają, bo obie odpadają na innych
  kryteriach niezależnie od metody DSR: H4 na PBO 0,96, obie na Sharpe w bessie (−3,89 i −3,05).

## Runda 6 — baza kalendarzowa kontraktów kwartalnych wobec fundingu, neutralnie rynkowo (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Właściciel wybrał kierunek rundy 6 na karcie decyzji 26.11.2026. Kontrakt kwartalny wycenia z góry koszt dźwigni do dnia dostawy
(bazę), a perpetual płaci stopę zmienną (funding). Jeśli baza systematycznie różni się od późniejszego fundingu, para
long perpetual / short kwartalny (albo odwrotnie) zarabia bez ekspozycji na kierunek rynku. W rundach 1–2 jedyny
stabilny zwrot po kosztach dało carry (H4, R2A), ale oba odpadły w bessie; bazy kwartalnej nie testowałem. Przed tym
zapisem sprawdziłem tylko kompletność danych, daty notowania kontraktów i spread kontraktów kwartalnych (do modelu
kosztów). Statystyk bazy nie oglądałem. Poziom fundingu BTC i ETH znam z rund 1–2. Kod rundy (commit d447658)
powstał przed tym zapisem i był uruchamiany tylko na danych syntetycznych w testach.

### Wspólne zasady rundy 6
- Rynek prób: Binance USDⓈ-M, BTCUSDT i ETHUSDT perpetual oraz ich kontrakty kwartalne (od BTCUSDT_210326 i
  ETHUSDT_210326), świece 1h z archiwum, dane badawcze do 31.05.2025. Dostawa jest o 08:00 UTC w dniu z symbolu.
  Rekordy archiwum po dostawie (np. BTCUSDT_230929 do 14.11.2023) nie są transakcjami, więc je odcinam.
- Decyzja raz dziennie na zamknięciu świecy 23:00 UTC (t = 00:00 UTC), wykonanie obu nóg na otwarciu następnej świecy.
- Kontrakt frontowy w t: spośród kontraktów notowanych co najmniej 24 h przed t (od pierwszej świecy z wolumenem), ze
  świecą w t i z co najmniej 7 dniami do dostawy, ten z najbliższą dostawą. Trzymany kontrakt roluję na pierwszej
  decyzji, w której przestaje spełniać warunek (6 dni i 8 godzin przed dostawą). Do 08.2023 Binance wystawiał następny
  kontrakt 2–10 dni przed dostawą bieżącego. Gdy żaden kontrakt nie spełnia warunku, moneta jest bez pozycji: według
  harmonogramu 26 dni na monetę w latach 2021–2023, z tego 18 w okresie OOS.
- Baza roczna b = (F/P − 1) · 365/D, gdzie F i P to zamknięcia świecy decyzji, a D to dni od t do dostawy.
- Funding f = suma stawek perpetuala rozliczonych w ciągu 30 dni przed t i znanych w t (rozliczenie + 5 min ≤ t),
  pomnożona przez 365/30. Suma, a nie średnia, działa tak samo przy rozliczeniu co 8 h (Binance) i co godzinę
  (Deribit).
- Stan na monetę: +1 = short baza (long perpetual, short kwartalny), −1 = long baza, 0 = bez pozycji. Każda noga ma
  0,5× kapitału: przy obu monetach łącznie 1× long i 1× short, czyli dźwignia 2:1, jak limit dla klienta detalicznego
  w UE. Wagi ustawiam tylko przy wejściu, wyjściu, zmianie kierunku i rollu. Między nimi silnik trzyma liczbę
  kontraktów, więc obie nogi dryfują z ceną, a zabezpieczenie zostaje moneta za monetę.
- Koszty: taker 0,05% od każdej transakcji, poślizg 2 pb na perpetualu i 3 pb na kontrakcie kwartalnym, funding
  rzeczywisty na nodze perpetual. Roll to zamknięcie starego i otwarcie nowego kontraktu po pełnym koszcie.
  Uzasadnienie 3 pb: w próbie 4 dni z lat 2023–2024 (bookTicker Binance) mediana połowy spreadu kontraktów
  kwartalnych wynosiła 0,02–0,67 pb, a o pełnych godzinach najwyżej 1,35 pb. Szczyt księgi miał jednak tylko
  500–1500 USD, więc większe zlecenie zejdzie głębiej w księgę.
- Każda decyzja musi mieć świecę wykonania na obu nogach. W danych nie ma luk (sprawdzone), a runner przerywa, gdyby
  jedna noga nie mogła zostać wykonana.
- Miara: zwrot ponad bony 3M (FRED DTB3, opóźnienie 1 dnia) od kapitału związanego jako depozyt. Depozyt to 50%
  nominału brutto, najwyżej cały kapitał; przy obu monetach w pozycji jest to cały kapitał. Reszta leży w bonach, jak
  w R2A. Wszystkie kryteria liczę na tej mierze. Sharpe bez odejmowania bonów podaję tylko informacyjnie.
- Ocena jak w rundzie 5:
  - walk-forward 24/3 z pierwszym oknem testowym 01.2022, więc OOS to 2022-01..2025-05; pierwszy wybór w R6B
    ma ok. 11 miesięcy danych;
  - PBO, koszty ×2, reżimy (Sharpe ≥ 0 w hossie, bessie i konsolidacji), obsunięcie ≤ 25%;
  - Sharpe OOS ≥ 1 i wyższy niż benchmarki w tym samym oknie;
  - DSR ≥ 0,95 z wariancją zerową 1/T (T = 1978 dni) i N = wszystkie próby programu (177). Wymaga to Sharpe OOS
    ok. 2,1 rocznie.
- R6A ma jeden wariant, więc nie ma w niej wyboru: PBO nie ma zastosowania i ten warunek uznaję za spełniony. Wpływ
  wielu prób ujmuje DSR liczony dla całego programu.
- Warunek dodatkowy kandydata: dodatni łączny zwrot ponad bony w OOS przy kosztach bazowych na Deribit.
  - Te same reguły: BTC i ETH, perpetual i kontrakty kwartalne (dostawa 08:00 UTC w ostatni piątek marca, czerwca,
    września i grudnia), świece 1h, funding godzinowy.
  - W każdym oknie walk-forward używam wariantu wybranego na Binance, bez strojenia.
  - Kontrakty Deribit są odwrotne (nominał w USD, rozliczenie w monecie); P&L traktuję liniowo w USD, co jest
    przybliżeniem.
  - Dane Deribit (sprawdzone pod kątem kompletności): świece 1h bez brakujących godzin, znaczniki czasu zgodne z
    Binance, funding co godzinę od 11.2020, 19 kontraktów kwartalnych na monetę. W 0–15% godzin kontraktu kwartalnego
    nie było transakcji, głównie przy dalekich terminach; cena takiej świecy to ostatnia transakcja.
  - Ten przebieg jest potwierdzeniem, nie nową próbą: zapisuję go w rejestrze jako zdarzenie i nie wliczam do N.

### R6A — premia terminowa bezwarunkowo
- Zawsze short baza: long perpetual, short kontrakt frontowy. **1 wariant.**
- Uzasadnienie: jeśli kontrakty kwartalne są systematycznie droższe niż późniejszy funding (popyt na ekspozycję
  bez zmiennego fundingu, ograniczony kapitał arbitrażystów), sprzedaż bazy zarabia tę premię.

### R6B — spread bazy i fundingu
- s = b − f. Bez pozycji: s > θ → short baza, s < −θ → long baza.
- Wyjście, gdy s osiąga przeciwny znak (s ≤ 0 przy short bazie, s ≥ 0 przy long bazie), albo przy rollu. Po wyjściu
  reguła wejścia działa od razu na tej samej decyzji, także na nowym kontrakcie frontowym.
- θ ∈ {2%, 5%} rocznie. **2 warianty.**
- Uzasadnienie: baza wycenia oczekiwany funding. Gdy odbiega od niedawnego fundingu bardziej niż koszty, różnica
  powinna się zamykać przez zbieżność bazy albo przez funding.

Razem 3 próby, więc N w programie wzrośnie ze 174 do 177. Jeśli runda 6 nie da kandydata, kończę badania bazy
kalendarzowej na tych danych bez strojenia progów i okien. Właściciel polecił szukać dalej, także inspiracji w
otwartych repozytoriach na GitHubie, więc przejdę do kolejnej rundy z własną prerejestracją i z tą samą korektą na
wszystkie próby programu.

## Runda 7 — carry na wielu monetach, arbitraż statystyczny i short nowych tokenów (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Runda 6 nie dała kandydata, a program liczy 177 prób. Właściciel polecił szukać dalej, także inspiracji w otwartych
repozytoriach na GitHubie. Przejrzałem nazwy i opisy najpopularniejszych repozytoriów ze strategiami na kryptowaluty,
bez ich wyników i bez kodu. Dominują trzy typy:
- arbitraż fundingu w pozycji neutralnej rynkowo, np. aoki-h-jp/funding-rate-arbitrage, IrakliXYZ/ARBOT i
  50shadesofgwei/funding-rate-arbitrage;
- arbitraż statystyczny na resztach modelu czynnikowego, np. repozytoria oparte na pracy Avellanedy i Lee;
- strategie freqtrade na wskaźnikach technicznych, np. freqtrade/freqtrade-strategies i NostalgiaForInfinity.

Reguły w stylu freqtrade odpowiadają rodzinom testowanym w rundach 1, 3 i 4. NostalgiaForInfinity dokupuje przy
spadkach, czego zasady programu zabraniają. Dlatego testuję dwa pierwsze typy i jedną hipotezę zdarzeniową.

Przed tym zapisem sprawdziłem tylko liczbę nowych kontraktów i ich obroty w pierwszych dniach (do filtra płynności)
oraz dostępność par spot w archiwum. Kompletność pobranych danych sprawdzę przed uruchomieniem, bez oglądania zwrotów.
Zwrotów altcoinów, ich fundingu ani wyników nowych kontraktów nie oglądałem. Funding BTC i ETH znam z rund 1–6, a
wyniki H2b, H3a, R2A i R2B z rund 1–2. R7B nie potrzebuje nowych danych, więc uruchomię ją pierwszą; R7A i R7C po
pobraniu dziennych świec spot i fundingu nowych kontraktów.

### Wspólne zasady rundy 7
- Dane badawcze do 31.05.2025, holdout zablokowany. Świece dzienne z archiwum Binance (perpetual USDⓈ-M i spot).
  Decyzja zapada na zamknięciu dnia UTC, a wykonanie na otwarciu następnego dnia.
- Koszty: prowizja taker 0,05% za stronę. Poślizg na perpetualu wynosi 2 pb dla BTC i ETH oraz 10 pb dla pozostałych.
  Noga spot ma dodatkowe 5 pb za wyższą prowizję spot (0,1%): 7 pb dla BTC i ETH, 15 pb dla pozostałych. Funding
  jest rzeczywisty. Test warunków skrajnych liczy koszty ×2.
- Funding jest znany w chwili t, gdy rozliczenie + 5 min ≤ t, jak w rundzie 6.
- Brak świecy wymusza wyjście (silnik, poślizg 2%). Kontrakty wycofane zostają w danych do ostatniej transakcji.
- Miara: zwrot ponad bony 3M (FRED DTB3, opóźnienie 1 dnia) od kapitału związanego. W R7A jest to kapitał w nodze
  spot, jak w R2A. W R7B i R7C jest to depozyt równy 50% nominału brutto, najwyżej cały kapitał, jak w R6.
- Ocena jak w rundach 5–6:
  - walk-forward 24/3 od 01.2020, więc OOS to 01.2022–05.2025;
  - PBO (CSCV, 16 bloków), koszty ×2, obsunięcie ≤ 25%;
  - Sharpe ≥ 0 w hossie, bessie i konsolidacji; reżim bez żadnej pozycji (zerowa zmienność zwrotu) liczę jako
    Sharpe 0, czyli warunek spełniony;
  - Sharpe OOS ≥ 1 i wyższy niż benchmarki w tym samym oknie;
  - DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 187), co wymaga Sharpe OOS
    ok. 2,1 rocznie.
- Giełda: dane Binance są najlepszym publicznym przybliżeniem, ale Binance nie obsługuje już większości klientów z
  Polski. Kandydat musiałby przejść paper trading na giełdzie dostępnej z Polski, gdzie funding i obroty mogą się
  różnić.

### R7A — carry spot–perpetual na wielu monetach
- Uzasadnienie: najczęstszy typ otwartych strategii to zbieranie fundingu w parze long spot / short perpetual. R2A
  dało nadwyżkę nad bonami tylko na BTC i ETH i tylko epizodycznie. H3a otrzymał funding równy 134% kapitału, ale
  bez zabezpieczenia ceny. R7A łączy oba wnioski, więc jest hipotezą inspirowaną danymi.
- Rynek: monety z universe point-in-time top 50, które mają parę spot USDT na Binance.
  - Para spot ma ten sam symbol albo symbol bez przedrostka wielkości kontraktu (1000PEPEUSDT i PEPEUSDT,
    mnożnik 1000).
  - Para jest dostępna w dniu t, gdy obie świece istnieją, spot ma co najmniej 30 dni notowań, a mediana
    P_perp / (mnożnik · P_spot) z ostatnich 7 dni mieści się w ±10%. Ten warunek chroni przed błędnym
    dopasowaniem nazw.
- Sygnał: e = funding z 3 dni przed t w ujęciu rocznym (suma stawek × 365/3) minus rentowność bonów.
- Wejście, gdy e > M i moneta jest w universe. Wyjście, gdy e < M/2, para przestaje być dostępna albo kontrakt
  przestaje spełniać warunki wieku, obrotu i płynności universe (bez warunku top 50).
- K miejsc. Każda para to +1/K kapitału w spot i −1/K w perpetual, więc przy pełnych miejscach 1× long spot i
  1× short perpetual. Wolne miejsca wypełniam monetami o najwyższym e, a trzymanej pary nie wymieniam na lepszą.
  Na każdej decyzji najpierw wyjścia, potem wejścia.
- Rebalans do wag docelowych z pasmem 2% kapitału, jak w R2A.
- M ∈ {10%, 20%} rocznie, K ∈ {4, 8}. **4 warianty.**

### R7B — powrót reszt do średniej (arbitraż statystyczny, Avellaneda i Lee 2010)
- Uzasadnienie: to drugi najczęstszy typ otwartych strategii. H2b handlował surowymi zwrotami, bez usunięcia
  ekspozycji na rynek i bez warunku szybkiego powrotu do średniej. R7B handluje odchyleniem reszty od modelu z BTC
  tylko wtedy, gdy reszta szybko wraca do średniej.
- Rynek: point-in-time top 20 universe bez BTC, który służy jako zabezpieczenie.
- Model w t z 60 dziennych zwrotów logarytmicznych:
  - regresja r_i = α + β·r_BTC + ε; X to skumulowana suma reszt w oknie;
  - regresja AR(1) X_{k+1} = a + b·X_k + ζ; model jest ważny, gdy 0 < b < e^(−1/30), czyli czas powrotu jest
    krótszy niż 30 dni;
  - m = a/(1 − b), σ_eq = √(Var(ζ)/(1 − b²)) z wariancją próbkową, s = (X_60 − m)/σ_eq. Reszty MNK sumują się
    do zera, więc X_60 = 0 i s = −m/σ_eq, jak u autorów.
  - Uwaga z testów na danych syntetycznych: przy oknie 60 dni estymator b jest obciążony w dół, więc filtr
    przepuszcza także błądzenie losowe w większości okien. Zostawiam go w wersji autorów.
- Bez pozycji: s < −s_in otwiera long monety i short β·BTC, a s > s_in otwiera short monety i long β·BTC. Long
  zamykam, gdy s > −0,5, a short, gdy s < 0,75. Pozycja zamyka się też, gdy moneta wypada z universe albo model
  przestaje być ważny.
- Rozmiar: 0,1× kapitału na monetę i zabezpieczenie BTC równe β·0,1, z β liczonym na bieżąco i obciętym do [0, 3].
  Najwyżej 10 pozycji; przy nadmiarze sygnałów pierwszeństwo ma większe |s|. Rebalans z pasmem 2% kapitału.
- s_in ∈ {1,25; 2,0}. **2 warianty.**

### R7C — short nowych tokenów (podaż po debiucie), neutralnie dolarowo
- Uzasadnienie: nowe tokeny debiutują z małą podażą w obiegu i wysoką wyceną w pełni rozwodnioną, a odblokowania
  dla zespołu i inwestorów tworzą stałą podaż po debiucie. Hipoteza jest obciążona wiedzą publiczną o latach
  2024–2025 (głośne spadki nowych tokenów), więc dane badawcze ją faworyzują. Rozstrzygnąć może dopiero holdout.
- Zdarzenie: pierwszy dzień z wolumenem (D0) kontraktu USDT perpetual, którego token jest nowy na Binance: para spot
  USDT tego tokenu nie istnieje albo nie miała notowań wcześniej niż 30 dni przed D0.
- Wyłączenia:
  - kontrakty indeksowe: BTCDOMUSDT, DEFIUSDT, FOOTBALLUSDT, BLUEBIRDUSDT;
  - stablecoiny: USDCUSDT, BUSDUSDT, TUSDUSDT, FDUSDUSDT, USDPUSDT;
  - znane zmiany nazwy lub denominacji istniejących tokenów: POLUSDT (MATIC), SUSDT (FTM), RENDERUSDT (RNDR),
    KAIAUSDT (KLAY), BEAMXUSDT (MC), VICUSDT (TOMO), GUSDT (GAL), EGLDUSDT (ERD), AAVEUSDT (LEND),
    1000LUNCUSDT (LUNA), 1000XECUSDT i XECUSDT (BCHA), 1000BTTCUSDT (BTT), HIFIUSDT (MFT), STRAXUSDT (STRAT).
    Zmiana nazwy, której nie znam, zostaje w próbie jako szum.
- Wejście: decyzja na zamknięciu dnia D0 + d, gdy średni obrót z dni D0+1 … D0+d wynosi co najmniej 20 mln USDT.
  Short otwieram na otwarciu następnego dnia. Wyjście po H dniach w pozycji albo przy wycofaniu kontraktu.
- Warunek wieku universe (90 dni) tu nie obowiązuje, bo chodzi właśnie o nowe kontrakty.
- Rozmiar: −0,1× kapitału na kontrakt, najwyżej 10 kontraktów jednocześnie; zdarzenia przy pełnych miejscach pomijam,
  a przy kilku zdarzeniach tego samego dnia pierwszeństwo ma większy obrót.
  Long BTC perpetual równy sumie shortów daje neutralność dolarową. Rebalans do wag docelowych z pasmem 2% kapitału:
  short, który rośnie, zmniejszam do wagi docelowej, więc ryzyko po stracie maleje.
- d ∈ {1, 7} dni, H ∈ {30, 90} dni. **4 warianty.**

Razem 10 prób, więc N w programie wzrośnie ze 177 do 187. Jeśli runda 7 nie da kandydata, przejdę do kolejnej rundy z
nową prerejestracją, bez strojenia progów i okien tej rundy i z tą samą korektą na wszystkie próby programu.

## Runda 8 — mechanizm podaży nowych kontraktów: koszyk altcoinów, nowe kontrakty starych tokenów i wiek tokena (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Runda 7 nie dała kandydata, a program liczy 187 prób. Najsilniejszy sygnał spoza carry dał short nowych tokenów z
longiem BTC (R7C): Sharpe OOS 1,01, ale obsunięcie −60%, DSR 0,38 i zysk skupiony w I kwartale 2025. Taka para łączy
dwa źródła wyniku: słabość nowych tokenów i słabość altcoinów wobec BTC w latach 2022–2025. Runda 8 rozdziela te
źródła i sprawdza mechanizm podaży na zdarzeniach i okresach, których zwrotów nie oglądałem. Wszystkie trzy rodziny
wynikają z wyników R7C, więc są hipotezami inspirowanymi danymi, a DSR liczę z liczbą wszystkich prób programu.

Przed tym zapisem sprawdziłem tylko liczebność zdarzeń i koszyków, bez zwrotów:
- 213 nowych kontraktów USDT perpetual na tokeny, których para spot USDT na Binance miała notowania wcześniej niż 30
  dni przed debiutem kontraktu (2020: 50, 2021: 41, 2022: 15, 2023: 58, 2024: 37, 2025: 12). We wszystkich ceny
  perpetual i spot z dni D0 i D0+1 zgadzają się w granicach ±10%, a 147 spełnia filtr płynności przy d = 1.
- W płynnym universe bez limitu liczby kontraktów (wiek kontraktu ≥ 90 dni, średni obrót ≥ 20 mln USDT) tokenów w
  wieku do 365 dni jest średnio od 2 (2020) do 41 (2025), a starszych niż 730 dni od 6 do 97. Co najmniej 3 nazwy w
  obu koszykach ma 84% tygodni przy progu 365 dni i 67% przy progu 180 dni.

Wyniki R7C i ich atrybucję po teście znam z rundy 7. Hipotezy tej rundy są też obciążone wiedzą publiczną o spadkach
nowych tokenów, a moja wiedza o rynku sięga połowy 2026 roku, więc obejmuje część holdoutu. Dla tych hipotez holdout
nie jest w pełni ślepy, a w pełni niezależne będą dopiero dane z handlu papierowego.

### Wspólne zasady rundy 8
- Jak w rundzie 7: dane badawcze do 31.05.2025, holdout zablokowany, świece dzienne perpetual USDⓈ-M z archiwum
  Binance, decyzja na zamknięciu dnia UTC i wykonanie na otwarciu następnego dnia.
- Koszty jak w rundzie 7: taker 0,05% za stronę, poślizg 2 pb dla BTC i ETH i 10 pb dla pozostałych, rzeczywisty
  funding (znany 5 min po rozliczeniu), test warunków skrajnych z kosztami ×2. Brak świecy wymusza wyjście (poślizg
  2%), a kontrakty wycofane zostają w danych.
- Rebalans do wag docelowych z pasmem 20% wagi docelowej każdego kontraktu. Dla shortu −0,1 to te same 2% kapitału co
  w R7C, a mniejsze pozycje koszyka mają pasmo proporcjonalne. Wejścia i wyjścia zawsze się wykonują. Silnik dostał do
  tego nowy parametr, przetestowany przed rundą; wcześniejsze wyniki go nie używają.
- Miara: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał, jak w R7B i R7C.
- Ocena jak w rundach 5–7: walk-forward 24/3 od 01.2020 (OOS 01.2022–05.2025), PBO (CSCV, 16 bloków), koszty ×2,
  obsunięcie ≤ 25%, Sharpe ≥ 0 w hossie, bessie i konsolidacji (reżim bez pozycji liczę jako Sharpe 0), Sharpe OOS
  ≥ 1 i wyższy niż benchmarki, DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 197), co
  wymaga Sharpe OOS ok. 2,1 rocznie.
- Koszyk altcoinów, zabezpieczenie w R8A i R8B: point-in-time top 20 universe bez BTC (jak w R7B), kontraktów
  indeksowych i stablecoinów, w równych wagach, bez kontraktów shortowanych tego dnia i bez kontraktów bez świecy.
  Nominał koszyka równa się sumie shortów, więc pozycja jest neutralna dolarowo. Gdy koszyk jest pusty, nie trzymam
  też shortów.
- Runner odmawia przebiegu rodziny, której kontrakty nie mają danych o fundingu.

### R8A — short nowych tokenów z koszykiem altcoinów zamiast BTC
- Uzasadnienie: jeśli przewaga R7C wynika z podaży nowych tokenów, przetrwa zabezpieczenie altcoinami. Jeśli wynikała
  ze słabości altcoinów wobec BTC, zniknie.
- Zdarzenia, wejście, wyjście, rozmiar i pierwszeństwo jak w R7C: te same 257 zdarzeń, filtr 20 mln USDT, −0,1
  kapitału na short, najwyżej 10 jednocześnie. Jedyna zmiana to long koszyka altcoinów zamiast longu BTC.
- d ∈ {1, 7} dni, H ∈ {30, 90} dni. **4 warianty.**

### R8B — short nowego kontraktu na token, który już był na Binance (Miller 1977)
- Uzasadnienie: gdy krótka sprzedaż jest utrudniona, cena odzwierciedla opinie optymistów (Miller 1977). Nowy kontrakt
  perpetual na Binance ułatwia shortowanie, a jego zapowiedź przyciąga kupujących, więc po debiucie kontraktu token
  powinien słabnąć względem rynku. Efekt osłabia to, że część tokenów miała już kontrakty na innych giełdach albo
  margin na Binance.
- Zdarzenie: pierwszy dzień z wolumenem (D0) kontraktu USDT perpetual, którego token miał parę spot USDT na Binance z
  notowaniami wcześniej niż 30 dni przed D0, czyli zdarzenia, które R7C wyłączyła jako stare tokeny. Ceny muszą się
  zgadzać: mediana P_perp / (mnożnik · P_spot) z dni D0 i D0+1 mieści się w ±10%. Wyłączenia jak w R7C (indeksy,
  stablecoiny, zmiany nazwy).
- Wejście, wyjście, rozmiar, pierwszeństwo i zabezpieczenie jak w R8A.
- d ∈ {1, 7} dni, H ∈ {30, 90} dni. **4 warianty.**

### R8C — wiek tokena: short młodych, long starych
- Uzasadnienie: odblokowania dla zespołu i inwestorów zaczynają się zwykle po 6–12 miesiącach i trwają 2–4 lata, więc
  podaż młodych tokenów rośnie także po pierwszych tygodniach od debiutu. Portfel przekrojowy trzyma naraz więcej
  tokenów niż R7C, a obejmuje inny okres życia tokena (od 90 dni zamiast pierwszych 30–90 dni).
- Rynek: płynne universe point-in-time bez limitu liczby kontraktów (wiek kontraktu ≥ 90 dni, średni obrót z 30 dni
  ≥ 20 mln USDT), z kontraktami wycofanymi, bez BTC, indeksów, stablecoinów i znanych zmian nazwy (ich wiek liczony od
  nowej nazwy byłby błędny).
- Wiek tokena: dni od pierwszego notowania na Binance, czyli wcześniejszej z dat: pierwszy dzień kontraktu perpetual
  albo pierwszy dzień pary spot dopasowanej po nazwie.
- Skład koszyków ustalam na zamknięciu każdej niedzieli i trzymam do następnej: młode to wiek ≤ A dni, stare to wiek
  powyżej 730 dni.
- Wagi: młode −0,1 na kontrakt, razem najwyżej −0,5 (przy ponad 5 nazwach równe wagi sumujące się do −0,5), stare w
  równych wagach o łącznym nominale równym shortom. Kontrakt bez świecy w danym dniu ma wagę 0, a reszta jego koszyka
  przejmuje nominał. Gdy którykolwiek koszyk ma mniej niż 3 nazwy ze świecą, nie mam pozycji.
- A ∈ {180, 365} dni. **2 warianty.**

Razem 10 prób, więc N w programie wzrośnie ze 187 do 197. Jeśli runda 8 nie da kandydata, przejdę do kolejnej rundy z
nową prerejestracją, bez strojenia progów i okien tej rundy i z tą samą korektą na wszystkie próby programu.

## Runda 9 — kontrola ryzyka shortów nowych kontraktów: wspólna pula zdarzeń, mniejsze pozycje i stop (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Runda 8 nie dała kandydata, a program liczy 197 prób. Short nowych kontraktów powtórzył się na zdarzeniach, których
zwrotów wcześniej nie oglądałem (R8B: Sharpe OOS 0,57, dodatni w każdym roku OOS, obsunięcie −27%), ale koszyk
altcoinów zamiast BTC nie zmniejszył obsunięcia R7C (R8A: −61%). Ryzyko leży więc w samych shortach. Runda 9
sprawdza, czy da się je ograniczyć bez utraty przewagi: jedna pula zdarzeń R7C i R8B, dwa razy mniejsze pozycje i stop
na zamknięciu dnia. To hipoteza inspirowana danymi: d = 1 i H = 30 wybrałem po wynikach R7C (walk-forward wybrał ten
wariant w 13 z 14 okien) i R8A (efekt tylko w pierwszych 30 dniach), a stop po obsunięciach R7C i R8A. DSR liczę z
liczbą wszystkich prób programu.

Przed tym zapisem sprawdziłem tylko liczebność zdarzeń i zajętość miejsc, bez zwrotów:
- Pula ma 470 zdarzeń: 257 z R7C i 213 z R8B, bez wspólnych kontraktów (2020: 75, 2021: 57, 2022: 22, 2023: 95,
  2024: 127, 2025 do maja: 94). Wszystkie kontrakty mają dane o fundingu.
- 383 zdarzenia spełniają filtr płynności przy d = 1. Bez stopu i przy limicie 20 shortów weszłoby 366, średnio 5,4
  shorta naraz (2025: 15,3), a limit byłby pełny w 2,5% dni.

Zastrzeżenia jak w rundzie 8: wyniki R7C i R8 znam, hipoteza jest obciążona wiedzą publiczną o spadkach nowych
tokenów, a moja wiedza o rynku obejmuje część holdoutu. W pełni niezależne będą dopiero dane z handlu papierowego.

### Wspólne zasady rundy 9
- Jak w rundzie 8: dane badawcze do 31.05.2025, holdout zablokowany, świece dzienne perpetual USDⓈ-M z archiwum
  Binance, decyzja na zamknięciu dnia UTC i wykonanie na otwarciu następnego dnia. Taker 0,05% za stronę, poślizg 2 pb
  dla BTC i ETH i 10 pb dla pozostałych, rzeczywisty funding, test z kosztami ×2. Brak świecy wymusza wyjście
  (poślizg 2%), a kontrakty wycofane zostają w danych.
- Rebalans z pasmem 20% wagi docelowej każdego kontraktu (dla shortu −0,05 to 1% kapitału). Wejścia i wyjścia zawsze
  się wykonują.
- Miara: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał.
- Ocena jak w rundach 5–8: walk-forward 24/3 od 01.2020 (OOS 01.2022–05.2025), PBO (CSCV, 16 bloków), koszty ×2,
  obsunięcie ≤ 25%, Sharpe ≥ 0 w hossie, bessie i konsolidacji (reżim bez pozycji liczę jako Sharpe 0), Sharpe OOS
  ≥ 1 i wyższy niż benchmarki, DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 200), co
  wymaga Sharpe OOS ok. 2,1 rocznie.
- Koszyk altcoinów jak w R8A i R8B: point-in-time top 20 bez BTC, kontraktów indeksowych i stablecoinów, w równych
  wagach, bez kontraktów shortowanych tego dnia i bez kontraktów bez świecy, o nominale równym sumie shortów. Gdy
  koszyk jest pusty, nie trzymam też shortów.
- Runner odmawia przebiegu, gdy któryś kontrakt nie ma danych o fundingu.

### R9A — wspólna pula shortów nowych kontraktów ze stopem
- Zdarzenia: suma zdarzeń R7C (nowe tokeny) i R8B (nowe kontrakty tokenów z historią na spocie Binance) z ich
  definicjami, wyłączeniami i kontrolą zgodności cen, czyli prawie każdy nowy kontrakt USDT perpetual po 01.01.2020.
- Wejście: na zamknięciu dnia D0 + 1, gdy obrót z tego dnia wynosi co najmniej 20 mln USDT, z wykonaniem na otwarciu
  dnia D0 + 2.
- Rozmiar: −0,05 kapitału na short, najwyżej 20 naraz, więc łączny short wynosi najwyżej −1,0 kapitału, jak w R7C.
  Wśród zdarzeń z tego samego dnia pierwszeństwo ma większy obrót, a zdarzenie bez wolnego miejsca przepada.
- Wyjście: po 30 dniach w pozycji, gdy kontrakt przestaje być notowany, albo na stopie. Stop: gdy zamknięcie dnia w
  pozycji wynosi co najmniej (1 + k) × cena otwarcia, po której short wszedł, short zamykam na następnym otwarciu.
  Zdarzenia nie shortuję ponownie, a jego miejsce się zwalnia i może je zająć zdarzenie z tego samego zamknięcia.
  Stop działa tylko na zamknięciu dnia, więc ruch powyżej poziomu w trakcie dnia go nie uruchamia, a wyjście może
  wypaść dużo powyżej poziomu.
- k ∈ {bez stopu, 0,5, 1,0}. **3 warianty.**
- Raportuję też liczbę stopów i stosunek ceny wyjścia do ceny wejścia przy stopie (tylko diagnostyka).

Razem 3 próby, więc N w programie wzrośnie ze 197 do 200. Jeśli R9A da Sharpe OOS ≥ 1 i obsunięcie do 25%, ale nie
przejdzie DSR, nie nazwę jej kandydatem: przedstawię właścicielowi decyzję, czy potwierdzić ją na holdoucie, czy na
danych z handlu papierowego. W przeciwnym razie przejdę do kolejnej rundy z nową prerejestracją, bez strojenia progów
i okien tej rundy i z tą samą korektą na wszystkie próby programu.

## Test końcowy R9A na holdoucie (prerejestracja 2026-11-26, przed jakimkolwiek otwarciem holdoutu; tylko za zgodą właściciela)

R9A nie jest kandydatem: spełnia pięć z siedmiu kryteriów, ale ma DSR 0,59 i PBO 0,51. Zasada z rundy 3 otwiera
holdout tylko dla kandydata, więc ten test jest wyjątkiem. Odbędzie się wyłącznie po wyraźnej zgodzie właściciela,
której treść zapisuję w rejestrze razem z wynikiem. Bez tej zgody holdout pozostaje zamknięty. Kryteria ustalam teraz,
przed decyzją właściciela i przed jakimkolwiek wglądem w dane holdoutu.

- Strategia: R9A ze stopem przy k = 1,0, czyli short zamykany, gdy zamknięcie dnia wynosi co najmniej dwukrotność ceny
  wejścia. Wybrałem ten wariant na danych badawczych: walk-forward wybrał go w 8 z 14 okien, a na całym okresie
  2020 – 05.2025 ma najwyższy Sharpe (1,56) i najmniejsze obsunięcie (−22%) z trzech wariantów. Pozostałe reguły bez
  zmian: zdarzenia R7C i R8B z ich definicjami, filtr 20 mln USDT, −0,05 kapitału na short, najwyżej 20 naraz,
  30 dni, koszyk top 20 bez BTC, pasmo rebalansu 20% wagi, koszty i funding jak w rundzie 9.
- Dane: wszystkie zapisane dane do 23.11.2026, czyli świece dzienne perpetual i spot, funding i bony 3M. Zdarzenia
  obejmują też kontrakty notowane po raz pierwszy od 01.06.2025. Koszyk to point-in-time top 20 ze wszystkich
  kontraktów. Na danych badawczych daje dokładnie ten sam koszyk co w R9A (0 różnic, sprawdzone przed tym zapisem bez
  danych holdoutu).
- Przebieg: jedna ciągła symulacja od 01.2020 do 23.11.2026, więc pozycje otwarte w maju 2025 przechodzą na holdout
  jak w prawdziwym handlu. Ocena obejmuje tylko dni od 01.06.2025.
- Miara: dzienny zwrot ponad bony od depozytu 50% nominału brutto, jak w rundzie 9.
- Kryteria zaliczenia, wszystkie naraz: Sharpe na holdoucie ≥ 1,0, maksymalne obsunięcie na holdoucie nie gorsze niż
  −25% i Sharpe przy kosztach ×2 > 0.
- Raportuję też, bez wpływu na werdykt: zwrot i Sharpe w każdym półroczu, PSR wobec Sharpe 0, liczbę zdarzeń, shortów
  i stopów, udział 10 najlepszych dni, wynik shortów z obu zbiorów zdarzeń oraz BTC buy and hold i SMA200 w tym samym
  oknie. Sprawdzam też, czy część symulacji do 31.05.2025 odtwarza zarejestrowany Sharpe tego wariantu.
- Skutki: zaliczenie pozwala przejść do handlu papierowego (krok 4), nadal bez prawdziwych pieniędzy. Niezaliczenie
  kończy rodzinę shortów nowych kontraktów. W obu przypadkach holdout przestaje być ślepym testem dla tej rodziny,
  a wynik trafia do rejestru i dokumentu bez względu na to, jaki będzie. Testu nie powtarzam.
- Wiarygodność: moja wiedza o rynku sięga połowy 2026 roku i obejmuje część holdoutu, a wybór hipotezy mógł być nią
  obciążony. Dobry wynik będzie więc słabszym dowodem niż zły. W pełni niezależne będą dopiero dane z handlu
  papierowego.
- Kod: `cqs.research.final_r9a`. Uruchomienie wymaga zatwierdzonego kodu, `CQS_ALLOW_HOLDOUT=1`, cytatu zgody
  właściciela i braku wcześniejszego testu końcowego R9A w rejestrze.

## Runda 10 — short po gwałtownym wzroście płynnych tokenów (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Runda 9 dała pierwszy trop (R9A: Sharpe OOS 1,30, obsunięcie −24,5%, bez DSR i PBO), a program liczy 200 prób. Po
teście sprawdziłem, bez nowych prób, że zwrot R9A nie wynika z ekspozycji na rynek: beta do BTC wynosi 0,00, do
koszyka altcoinów 0,03, a alfa ok. 44% rocznie ma t = 2,4 (błędy Neweya-Westa). Jedno z wyjaśnień tej przewagi to
przecena aktywów o profilu loterii przy ograniczonej krótkiej sprzedaży (Miller 1977; efekt MAX w akcjach: Bali, Cakici
i Whitelaw 2011; w kryptowalutach wyniki badań są mieszane). Jeśli to wyjaśnienie jest trafne, podobny efekt powinien
wystąpić po gwałtownych wzrostach płynnych tokenów, które nie są nowe. Runda 10 sprawdza to na zdarzeniach rozłącznych
z R9A, bo wymaga wieku kontraktu co najmniej 90 dni. Po takich wzrostach longi często płacą wysoki funding, więc short
może też na nim zarabiać; tego przed testem nie sprawdzałem. Konstrukcja portfela (rozmiar, limit, stop, koszyk)
pochodzi z R9A, więc to hipoteza inspirowana danymi, a DSR liczę z liczbą wszystkich prób programu.

Przed tym zapisem sprawdziłem tylko liczebność zdarzeń i zajętość miejsc bez stopu, bez zwrotów po zdarzeniu:
- Wzrost o co najmniej 30% w jeden dzień: 647 zdarzeń na 225 kontraktach (2020: 6, 2021: 251, 2022: 82, 2023: 61,
  2024: 158, 2025 do maja: 89), z tego 390 w oknie OOS. Przy progu 50%: 154 zdarzenia na 99 kontraktach, 105 w oknie
  OOS. Wszystkie kontrakty mają dane o fundingu.
- Bez stopu przy progu 30% weszłoby 485 shortów przy H = 10 (średnio 2,5 naraz) i 406 przy H = 30 (6,1 naraz, limit
  pełny w 3% dni). Przy progu 50%: 136 i 130 shortów (0,7 i 1,9 naraz).

### Wspólne zasady rundy 10
- Jak w rundzie 9: dane badawcze do 31.05.2025, holdout zablokowany, świece dzienne perpetual USDⓈ-M z archiwum
  Binance, decyzja na zamknięciu dnia UTC i wykonanie na otwarciu następnego dnia. Taker 0,05% za stronę, poślizg 2 pb
  dla BTC i ETH i 10 pb dla pozostałych, rzeczywisty funding, test z kosztami ×2. Brak świecy wymusza wyjście
  (poślizg 2%), a kontrakty wycofane zostają w danych.
- Rebalans z pasmem 20% wagi docelowej każdego kontraktu. Wejścia i wyjścia zawsze się wykonują.
- Miara: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał.
- Ocena jak w rundach 5–9: walk-forward 24/3 od 01.2020 (OOS 01.2022–05.2025), PBO (CSCV, 16 bloków), koszty ×2,
  obsunięcie ≤ 25%, Sharpe ≥ 0 w hossie, bessie i konsolidacji (reżim bez pozycji liczę jako Sharpe 0), Sharpe OOS
  ≥ 1 i wyższy niż benchmarki, DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 204), co
  wymaga Sharpe OOS ok. 2,1 rocznie.
- Koszyk altcoinów jak w R8A–R9A: point-in-time top 20 bez BTC, kontraktów indeksowych i stablecoinów, w równych
  wagach, bez kontraktów shortowanych tego dnia i bez kontraktów bez świecy, o nominale równym sumie shortów.
- Runner odmawia przebiegu, gdy któryś kontrakt nie ma danych o fundingu.

### R10A — short po wzroście o co najmniej X w jeden dzień
- Rynek: płynne universe point-in-time bez limitu liczby kontraktów (wiek kontraktu ≥ 90 dni, średni obrót z 30 dni
  ≥ 20 mln USDT), z kontraktami wycofanymi, bez BTC, kontraktów indeksowych, stablecoinów, znanych zmian nazwy i
  kontraktów z datą wygaśnięcia.
- Zdarzenie: dzień t, w którym kontrakt należący do universe na zamknięciu dnia t − 1 zamknął się co najmniej o X
  wyżej niż w dniu t − 1.
- Wejście: na zamknięciu dnia t + 1, gdy obrót z tego dnia wynosi co najmniej 20 mln USDT, z wykonaniem na otwarciu
  dnia t + 2.
- Rozmiar i limity jak w R9A: −0,05 kapitału na short, najwyżej 20 naraz i jeden short na kontrakt. Zdarzenie na
  kontrakcie, który jest już shortowany, przepada. Wśród zdarzeń z tego samego dnia pierwszeństwo ma większy obrót.
- Wyjście: po H dniach w pozycji, gdy kontrakt przestaje być notowany, albo na stopie z R9A, gdy zamknięcie dnia
  wynosi co najmniej dwukrotność ceny wejścia (zamknięcie na następnym otwarciu). Po stopie kontrakt nie wraca do
  shortu przez H dni. Kolejny wzrost tego samego kontraktu po zamknięciu pozycji jest nowym zdarzeniem.
- X ∈ {30%, 50%}, H ∈ {10, 30} dni. **4 warianty.**
- Raportuję też liczbę stopów, łączny funding i podział wyniku na lata (tylko diagnostyka).

Razem 4 próby, więc N w programie wzrośnie z 200 do 204. Jeśli R10A da Sharpe OOS ≥ 1 i obsunięcie do 25%, ale nie
przejdzie DSR, nie nazwę jej kandydatem, a decyzję o potwierdzeniu przedstawię właścicielowi, jak w rundzie 9. Jeśli
obie rodziny, R9A i R10A, okażą się trafami, wspólny portfel będzie osobną próbą z własną prerejestracją. W przeciwnym
razie przejdę do kolejnej rundy z nową prerejestracją, bez strojenia progów i okien tej rundy.

## Runda 11 — short przy skrajnie ujemnym fundingu starszych tokenów (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Runda 10 wykluczyła jedno z wyjaśnień przewagi R9A: po gwałtownym wzroście płynnych, starszych tokenów short traci
(R10A: Sharpe OOS −0,90), więc nie chodzi o ogólną przecenę aktywów o profilu loterii. Drugie wyjaśnienie dotyczy
kosztu krótkiej sprzedaży. Na nowych kontraktach shorty zwykle płacą funding (funding ujemny), a cena spada bardziej,
niż wynosi ta opłata. Teoria Millera (1977) mówi, że przy ograniczonej krótkiej sprzedaży cenę wyznaczają optymiści,
więc aktywo jest przewartościowane. W akcjach spółki drogie do shortowania wypadają gorzej o więcej niż opłata za
pożyczkę (Jones i Lamont 2002; Cohen, Diether i Malloy 2007; Boehmer, Jones i Zhang 2008; Drechsler i Drechsler
2014). W kontraktach perpetual ujemny funding jest ceną shortu: shorty płacą longom, gdy kontrakt notowany jest poniżej
indeksu, a arbitraż wymagałby pożyczenia tokena na rynku spot, co dla altcoinów bywa trudne.

Jeśli to wyjaśnienie jest trafne, efekt powinien wystąpić także na starszych tokenach (wiek kontraktu co najmniej 90
dni, więc zdarzenia są rozłączne z R9A), gdy ich funding staje się skrajnie ujemny. Hipoteza: takie kontrakty wypadają
gorzej od koszyka altcoinów o więcej, niż shorty płacą w fundingu. Scenariusz przeciwny to squeeze zatłoczonych shortów
(np. ALPACA w kwietniu 2025 w R7B i R10A); test rozstrzyga, który przeważa. Hipoteza wynika z wyniku R9A, a konstrukcja
portfela (rozmiar, limit, stop, koszyk) pochodzi z R9A, więc to hipoteza inspirowana danymi, a DSR liczę z liczbą
wszystkich prób programu.

Przed tym zapisem sprawdziłem tylko liczebność zdarzeń i zajętość miejsc bez stopu, bez zwrotów po zdarzeniu:
- Funding dnia ≤ −0,3% (ok. −0,1% na 8 godzin, rocznie ok. −110%): 2292 dni zdarzeń na 237 kontraktach (2020: 79,
  2021: 298, 2022: 519, 2023: 433, 2024: 418, 2025 do maja: 545), z tego 1915 w oknie OOS. Przy progu −0,6%: 1087 dni
  zdarzeń na 186 kontraktach (21, 112, 203, 249, 202, 300), 954 w oknie OOS. Wszystkie 366 kontraktów płynnego
  universe ma dane o fundingu.
- Bez stopu przy progu 0,3% weszłoby 736 shortów przy H = 10 (średnio 4,0 naraz) i 570 przy H = 30 (8,7 naraz, limit
  pełny w 8% dni). Przy progu 0,6%: 420 i 359 shortów (2,2 i 5,3 naraz).

### Wspólne zasady rundy 11
- Jak w rundzie 10: dane badawcze do 31.05.2025, holdout zablokowany, świece dzienne perpetual USDⓈ-M z archiwum
  Binance, decyzja na zamknięciu dnia UTC i wykonanie na otwarciu następnego dnia. Taker 0,05% za stronę, poślizg 2 pb
  dla BTC i ETH i 10 pb dla pozostałych, rzeczywisty funding, test z kosztami ×2. Brak świecy wymusza wyjście
  (poślizg 2%), a kontrakty wycofane zostają w danych.
- Rebalans z pasmem 20% wagi docelowej każdego kontraktu. Wejścia i wyjścia zawsze się wykonują.
- Miara: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał.
- Ocena jak w rundach 5–10: walk-forward 24/3 od 01.2020 (OOS 01.2022–05.2025), PBO (CSCV, 16 bloków), koszty ×2,
  obsunięcie ≤ 25%, Sharpe ≥ 0 w hossie, bessie i konsolidacji (reżim bez pozycji liczę jako Sharpe 0), Sharpe OOS
  ≥ 1 i wyższy niż benchmarki, DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 208), co
  wymaga Sharpe OOS ok. 2,1 rocznie.
- Koszyk altcoinów jak w R8A–R10A: point-in-time top 20 bez BTC, kontraktów indeksowych i stablecoinów, w równych
  wagach, bez kontraktów shortowanych tego dnia i bez kontraktów bez świecy, o nominale równym sumie shortów.
- Runner odmawia przebiegu, gdy któryś kontrakt płynnego universe nie ma danych o fundingu.

### R11A — short po dniu z fundingiem nie wyższym niż −X
- Rynek: płynne universe jak w R10A (wiek kontraktu ≥ 90 dni, średni obrót z 30 dni ≥ 20 mln USDT), z kontraktami
  wycofanymi, bez BTC, kontraktów indeksowych, stablecoinów, znanych zmian nazwy i kontraktów z datą wygaśnięcia.
- Funding dnia t: suma stawek rozliczonych od 00:00 dnia t do 00:00 dnia t + 1 UTC (bez tej godziny), czyli koszt
  shortu trzymanego przez cały dzień, niezależnie od tego, czy kontrakt rozlicza funding co 8, 4 czy 1 godzinę.
- Zdarzenie: dzień t, w którym kontrakt należący do universe na zamknięciu dnia t − 1 ma funding dnia t ≤ −X.
- Wejście: na zamknięciu dnia t + 1, gdy obrót z tego dnia wynosi co najmniej 20 mln USDT, z wykonaniem na otwarciu
  dnia t + 2.
- Rozmiar i limity jak w R9A i R10A: −0,05 kapitału na short, najwyżej 20 naraz i jeden short na kontrakt. Zdarzenie
  na kontrakcie, który jest już shortowany, przepada. Wśród zdarzeń z tego samego dnia pierwszeństwo ma większy obrót.
- Wyjście: po H dniach w pozycji, gdy kontrakt przestaje być notowany, albo na stopie z R9A, gdy zamknięcie dnia
  wynosi co najmniej dwukrotność ceny wejścia (zamknięcie na następnym otwarciu). Po stopie kontrakt nie wraca do
  shortu przez H dni. Kolejne zdarzenie po zamknięciu pozycji otwiera nowy short, więc przy trwale skrajnym fundingu
  pozycja może zostać odnowiona.
- X ∈ {0,3%, 0,6%} dziennie, H ∈ {10, 30} dni. **4 warianty.**
- Raportuję też liczbę stopów, funding zapłacony przez shorty i podział wyniku na lata (tylko diagnostyka).

Razem 4 próby, więc N w programie wzrośnie z 204 do 208. Jeśli R11A da Sharpe OOS ≥ 1 i obsunięcie do 25%, ale nie
przejdzie DSR, nie nazwę jej kandydatem, a decyzję o potwierdzeniu przedstawię właścicielowi, jak w rundzie 9. Jeśli
R11A i R9A okażą się trafami, wspólny portfel będzie osobną próbą z własną prerejestracją. W przeciwnym razie przejdę
do kolejnej rundy z nową prerejestracją, bez strojenia progów i okien tej rundy.

## Runda 12 — short nowego kontraktu już na zamknięciu dnia debiutu (prerejestracja 2026-11-26, przed jakimkolwiek testem tej rundy)

Rundy 10 i 11 wykluczyły dwa ogólne wyjaśnienia przewagi R9A: przecenę gwałtownych wzrostów (R10A: Sharpe OOS −0,90)
i koszt krótkiej sprzedaży starszych tokenów (R11A: −0,83). Zostaje mechanizm samego debiutu: podaż tokenów, które
trafiają na rynek od pierwszych godzin notowań (airdropy, odblokowane przydziały, zapasy animatorów rynku,
zabezpieczanie zablokowanych tokenów). R9A decyduje dopiero na zamknięciu dnia D0 + 1, a wchodzi na otwarciu D0 + 2,
bo tę regułę przejęła z R7C, gdzie dzień debiutu jest niepełny. Jeśli podaż działa od razu, decyzja na zamknięciu
samego dnia D0 (wejście na otwarciu D0 + 1) powinna złapać dodatkowy dzień spadku. Ryzyko: pierwsze dni po debiucie
mają też najsilniejsze squeezy.

To hipoteza inspirowana danymi: R7C dała lepszy wynik przy d = 1 niż przy d = 7, a przesunięcie wejścia jeszcze
wcześniej jest ekstrapolacją tego wyniku, wspartą publiczną wiedzą o spadkach nowych tokenów w latach 2024–2025.
DSR liczę z liczbą wszystkich prób programu. R12A nie zastępuje R9A w prerejestrowanym teście końcowym na holdoucie,
który zostaje bez zmian. Jeśli R12A okaże się trafem, o jej potwierdzeniu zdecyduje właściciel osobno (handel
papierowy albo, jeśli holdout nie zostanie użyty dla R9A, osobny test końcowy z własną prerejestracją).

Przed tym zapisem sprawdziłem tylko liczebność zdarzeń i zajętość miejsc bez stopu, bez zwrotów po zdarzeniu:
- Z 470 zdarzeń R9A filtr obrotu 20 mln USDT w dniu debiutu przechodzi 361 (w R9A, w dniu D0 + 1: 383). Mediana obrotu
  w dniu debiutu to 66,9 mln USDT, w dniu D0 + 1 100,5 mln USDT.
- Bez stopu weszłoby 344 shortów, średnio 5,1 naraz (6,7 w oknie OOS), limit pełny w 2,9% dni. Reguła R9A daje 366
  shortów, średnio 5,4 naraz (7,0 w OOS) i limit pełny w 2,5% dni.

### Wspólne zasady rundy 12
- Jak w rundzie 9: zdarzenia to wspólna pula R7C i R8B z ich definicjami i wyłączeniami, dane badawcze do 31.05.2025,
  holdout zablokowany, świece dzienne perpetual USDⓈ-M z archiwum Binance, decyzja na zamknięciu dnia UTC i wykonanie
  na otwarciu następnego dnia. Taker 0,05% za stronę, poślizg 2 pb dla BTC i ETH i 10 pb dla pozostałych, rzeczywisty
  funding, test z kosztami ×2. Brak świecy wymusza wyjście (poślizg 2%), a kontrakty wycofane zostają w danych.
- Rebalans z pasmem 20% wagi docelowej każdego kontraktu. Wejścia i wyjścia zawsze się wykonują.
- Miara: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał.
- Ocena jak w rundach 5–11: walk-forward 24/3 od 01.2020 (OOS 01.2022–05.2025), PBO (CSCV, 16 bloków), koszty ×2,
  obsunięcie ≤ 25%, Sharpe ≥ 0 w hossie, bessie i konsolidacji (reżim bez pozycji liczę jako Sharpe 0), Sharpe OOS
  ≥ 1 i wyższy niż benchmarki, DSR ≥ 0,95 z wariancją 1/T i N = wszystkie próby programu (po tej rundzie 210), co
  wymaga Sharpe OOS ok. 2,1 rocznie.
- Koszyk altcoinów jak w R8A–R11A. Runner odmawia przebiegu, gdy któryś kontrakt nie ma danych o fundingu.

### R12A — wejście na zamknięciu dnia debiutu
- Zdarzenia: 470 zdarzeń R9A bez zmian.
- Wejście: na zamknięciu dnia D0, gdy obrót z dnia debiutu (od pierwszego notowania do 24:00 UTC) wynosi co najmniej
  20 mln USDT, z wykonaniem na otwarciu dnia D0 + 1.
- Rozmiar i limity jak w R9A: −0,05 kapitału na short, najwyżej 20 naraz; wśród zdarzeń z tego samego dnia
  pierwszeństwo ma większy obrót, a zdarzenie bez wolnego miejsca przepada.
- Wyjście: po 30 dniach w pozycji, gdy kontrakt przestaje być notowany, albo na stopie z R9A, gdy zamknięcie dnia
  wynosi co najmniej (1 + k) × cenę wejścia (zamknięcie na następnym otwarciu); kontrakt nie wraca do shortu.
- k ∈ {bez stopu, 1,0}. **2 warianty.**
- Raportuję też liczbę stopów, podział wyniku na lata i porównanie z wariantami R9A o tym samym k (tylko diagnostyka).

Razem 2 próby, więc N w programie wzrośnie z 208 do 210. Jeśli R12A da Sharpe OOS ≥ 1 i obsunięcie do 25%, ale nie
przejdzie DSR, nie nazwę jej kandydatem, a decyzję o potwierdzeniu przedstawię właścicielowi. W przeciwnym razie
przejdę do kolejnej rundy z nową prerejestracją, bez strojenia dnia wejścia i stopu tej rundy.

## Runda 13 — replikacja R9A na Hyperliquid: ta sama reguła na nowych kontraktach innej giełdy (prerejestracja 2026-11-27, przed pobraniem jakichkolwiek danych Hyperliquid)

R9A to jedyny trop programu (Sharpe OOS 1,30, obsunięcie −24,5%), ale nie przechodzi DSR ani PBO, a runda 12
pokazała, że inny dzień wejścia daje tę samą strategię (korelacja 0,93). Kolejne warianty na tych samych zdarzeniach
Binance niczego nie rozstrzygną. Rozstrzygnąć może reguła ustalona w rundzie 9, sprawdzona bez zmian na zdarzeniach
z innej giełdy. Hyperliquid notuje nowe kontrakty perpetual, często wcześniej niż Binance albo takie, których Binance
nie notuje, ma publiczne API z historią świec i fundingu i jest dostępny z Polski, w przeciwieństwie do Binance.
To runda potwierdzająca, nie szukanie kandydata: jedna próba, reguła R9A przeniesiona bez strojenia, z dwiema
zmianami wymuszonymi przez rynek (próg obrotu i koszty), ustalonymi przed obejrzeniem danych.

Przed tym zapisem nie pobrałem żadnych danych Hyperliquid, bo proxy środowiska blokowało api.hyperliquid.xyz. Z danych
badawczych Binance zapisałem tylko pierwszy dzień notowań każdego kontraktu USDT (research/data/
binance_perp_first_day.csv, 492 kontrakty, ostatni 30.05.2025), bez cen po zdarzeniach.

### Wspólne zasady rundy 13
- Dane: publiczne API Hyperliquid (api.hyperliquid.xyz/info): lista kontraktów perpetual z `meta` (z wycofanymi),
  świece dzienne UTC (`candleSnapshot`, 1d) i godzinowy funding (`fundingHistory`). Pobieram wyłącznie dane sprzed
  01.06.2025: zapytania kończą się 31.05.2025 23:59:59,999 UTC, a późniejsze wiersze odrzucam, więc holdout zostaje
  zamknięty także dla Hyperliquid. Każda odpowiedź API ma w manifeście sumę SHA-256, jak dane Deribit w rundzie 6.
  Klient: `cqs.data.hyperliquid`.
- Hyperliquid podaje wolumen w jednostkach tokena, więc obrót w USD przybliżam jako wolumen × (O + H + L + C) / 4
  świecy dziennej. Funding rozliczany co godzinę przypisuję do pełnej godziny (znaczniki czasu API odbiegają o kilka
  milisekund), a silnik nalicza go pozycjom trzymanym w chwili rozliczenia, jak na Binance.
- Wycofane kontrakty: `meta` zachowuje je z flagą isDelisted, a numer kontraktu rośnie z kolejnością dodania.
  Kontrakt bez świec sprzed 01.06.2025, którego numer jest niższy niż najwyższy numer kontraktu ze świecami z tego
  okresu, liczę jako brak danych. Jeśli braki przekroczą 10% takich kontraktów, wynik oznaczam jako narażony na błąd
  przeżywalności i nie wyciągam z niego wniosków. Płaskie świece bez obrotu przed pierwszym i po ostatnim handlu
  pomijam.
- Decyzja na zamknięciu dnia UTC, wykonanie na otwarciu następnego dnia, silnik jak w rundach 7–12 (pasmo rebalansu
  20% wagi docelowej, wejścia i wyjścia zawsze się wykonują). Brak świecy wymusza wyjście z poślizgiem 2%.
- Koszty: taker 0,045% za stronę (podstawowy próg Hyperliquid w 2025 r.; w latach 2023–2024 opłata była niższa, więc
  założenie jest ostrożne), poślizg 20 pb na każdym kontrakcie Hyperliquid (dwa razy więcej niż dla altcoinów na
  Binance, bo arkusze zleceń są płytsze), rzeczywisty godzinowy funding Hyperliquid i test z kosztami ×2. Runner
  odmawia przebiegu, gdy któryś handlowany kontrakt nie ma danych o fundingu.
- Miara jak w rundach 7–12: zwrot ponad bony 3M od depozytu równego 50% nominału brutto, najwyżej cały kapitał.

### R13A — reguła R9A na nowych kontraktach Hyperliquid
- Zdarzenie: pierwszy dzień D0 ze świecą dzienną o niezerowym wolumenie dla kontraktu z `meta`, D0 najpóźniej
  29.05.2025.
- Wyłączenia: BTC oraz stare tokeny, czyli kontrakty, których kontrakt USDⓈ-M USDT na Binance był notowany wcześniej
  niż 30 dni przed D0 na Hyperliquid. Wtedy rynek perpetual na ten token już istniał, a mechanizm R9A (nowa podaż
  i zniesienie ograniczeń krótkiej sprzedaży przy debiucie rynku) nie działa. Mapowanie nazw: nazwa + USDT; przedrostek
  k oznacza 1000 jednostek (kPEPE → 1000PEPEUSDT, a jeśli takiego kontraktu nie ma, PEPEUSDT). Kontrakty Binance
  będące nową nazwą starego tokena (lista RENAMES z rundy 7, np. RENDERUSDT, POLUSDT) oznaczają stary token. Nazwy bez
  dokładnego odpowiednika, dla których Binance ma kontrakt o innym mnożniku (np. 1000000MOGUSDT), wypisuję w raporcie,
  ale reguły nie zmieniam.
- Wejście: na zamknięciu D0 + 1, gdy obrót z dnia D0 + 1 wynosi co najmniej 2 mln USD, z wykonaniem na otwarciu
  D0 + 2. To próg R9A (20 mln USDT) podzielony przez 10, bo obrót perpetual na Hyperliquid był w latach 2023–2025 rzędu
  kilku do kilkunastu procent obrotu Binance; zlecenie 5% kapitału 100 tys. USD to przy nim najwyżej 0,25% obrotu dnia.
- Rozmiar, limity i wyjście jak w wariancie R9A z testu końcowego: −0,05 kapitału na short, najwyżej 20 naraz
  (pierwszeństwo ma większy obrót, zdarzenie bez wolnego miejsca przepada), 30 dni w pozycji, stop przy zamknięciu dnia
  co najmniej 2 × cena wejścia (k = 1,0) z zamknięciem na następnym otwarciu, bez ponownego shortu tego kontraktu.
- Zabezpieczenie jak w R8A–R12A, ale z kontraktów Hyperliquid: koszyk równoważony o tym samym nominale co shorty,
  z 20 największych kontraktów według średniego obrotu z 30 dni (co najmniej 2 mln USD, co najmniej 90 dni notowań),
  z których usuwam BTC; członek koszyka shortowany danego dnia z niego wypada, a bez członków koszyka nie ma shortów.
- **1 wariant**, bez walk-forward: reguła została ustalona w rundzie 9 na danych Binance, więc cały okres Hyperliquid
  jest dla niej poza próbą. Kod: `cqs.strategies.hl_listings`, `cqs.research.run_round13`.

### Ocena rundy 13
- Najpierw podaję liczbę kontraktów, zdarzeń, wyłączeń, przejść filtra obrotu i braków danych.
- Potem: Sharpe zwrotu ponad bony od pierwszego dnia z pozycją, CAGR, maksymalne obsunięcie, wynik przy kosztach ×2,
  wyniki roczne, reżimy z dziennego zamknięcia BTC na Hyperliquid (definicje jak w poprzednich rundach), PSR względem
  zera, DSR z wariancją 1/T i N = wszystkie próby programu (dla porządku), korelację dziennych zwrotów z R9A
  (research/results/R9A_oos.csv) we wspólnym okresie, liczbę stopów oraz podział transakcji na zdarzenia, przy których
  kontrakt Binance był już notowany w chwili decyzji, i pozostałe (diagnostyka tego samego przebiegu).
- Reguła rozstrzygnięcia, w tej kolejności: przy brakach danych powyżej 10% wynik jest narażony na błąd przeżywalności;
  przy mniej niż 30 shortach jest za mały na wniosek; Sharpe ponad bony ≤ 0 oznacza brak replikacji; Sharpe > 0
  z PSR(0) ≥ 0,90 i dodatnim Sharpe przy kosztach ×2 oznacza replikację; pozostałe przypadki są nierozstrzygnięte.
- R13A nie jest kandydatem, niezależnie od wyniku. Replikacja wzmocni argument za testem końcowym R9A i za handlem
  papierowym tej reguły na Hyperliquid, a jej brak osłabi trop R9A; w obu przypadkach decyzję o R9A podejmuje
  właściciel. Wspólna strategia Binance + Hyperliquid wymagałaby osobnej prerejestracji.

Razem 1 próba, więc N w programie wzrośnie z 210 do 211.

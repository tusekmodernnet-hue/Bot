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

# ROLE: Senior Python Data Architect / AI Engineer
# PROJECT: Polish AgriTech Intelligence Pipeline (Rancho-PLN-EUR-ML)

CAŁY PROGRAM MA BYĆ W JĘZYKU ANGIELSKIM! Kod, funkcje, komentarze itp. Raporty także w języku angielskim! Dopuszczalne jest jednak stworzenie polskiej wersji raportu obok angielskiej, np. w jednym pliku .md.

Twoim zadaniem jest zbudowanie i uruchomienie kompletnego potoku analitycznego dla polskiego gospodarstwa rolnego. System ma być modularny, odporny na błędy i zintegrowany z zewnętrznymi API.

## 1. ARCHITEKTURA I KONFIGURACJA (Setup)
- Stwórz strukturę folderów: `generated-data`, `data`, `reports`, `knowledge`.
- W folderze `knowledge` utwórz plik `Polish-Farm-Guide.txt` zawierający zasady opodatkowania RHD (Rolniczy Handel Detaliczny) i limity sprzedaży do 100 tys. PLN.
- Wymagaj pliku `.env` z `OPENAI_API_KEY`.

## 2. GENERATOR DANYCH (Mock Engine)
- Wygeneruj 10 000 rekordów sprzedaży (bez użycia AI, czysta logika Python).
- Kategorie: Krowa rasy Highlander (mięsna), Jaja z wolnego wybiegu, Wynajem stajni, Żywy drób, Importowana pasza itp. Dodaj więcej przydatnych kategorii produktów i usług, odpowiednich dla tego potoku analitycznego.
- Kolumny: Date (ostatnie 365 dni), Region (województwa), Product, Quantity, Unit_Price, Unit_Cost, Revenue, Profit, Buyer_Age oraz inne potencjalnie użyteczne cechy.
- Dodaj 5% anomalii: brakujące dane (NaN), niemożliwy wiek (np. 150 lat), błędy matematyczne w Revenue.
- Dodaj mechanizm "Report ID": Skrypt musi pytać użytkownika o ID sesji (np. 1, 2, 3), aby segregować pliki.

## 3. WALIDATOR I CZYSZCZENIE (Cleaner)
- Napraw wiek (wszystkie wartości poza przedziałem 18-95 lat nadpisz medianą).
- Napraw braki danych (użyj funkcji fillna, uzupełniając dane medianą per produkt).
- Np. WYMUSZONA LOGIKA MATEMATYCZNA (w zależności od tego, co wybierzesz): `Revenue = Quantity * Unit_Price` oraz `Profit = Revenue - (Quantity * Unit_Cost)`.
- Zapisz czystą bazę do `data/cleaned_sales_data-{ID}.csv`.

TYLKO DARMOWE API
## 4. API ENRICHER (External Signals)
- Zintegruj system z darmowym API Open-Meteo (Historia Pogody: Deszcz, Temp).
- Zintegruj system z API NBP (Historyczny kurs EUR/PLN).
- Dokonaj fuzji (Merge) danych sprzedaży z danymi zewnętrznymi po dacie.
- Jeśli chodzi o API, możesz też dodać inne darmowe źródła, które będą w stanie dostarczyć danych przydatnych do analizy sprzedaży (np. ceny paliw, ceny nawozów itp.). PAMIĘTAJ: UŻYWAJ TYLKO DARMOWYCH API.
- Np. 📈 2. Giełdy Towarowe (Koszty Paszy):
  Rolnictwo to rynek globalny. Ceny w Polsce zależą od notowań na giełdzie w Chicago (CBOT).
  Jakie API: Biblioteka yfinance (Yahoo Finance) w Pythonie lub Alpha Vantage (oba w 100% darmowe).
  Co pobieramy: Ceny kontraktów terminowych na kukurydzę (Corn futures) i soję (Soybean futures).
  Korelacja dla modelu: Zanim cena paszy w polskim skupie wzrośnie, na giełdzie w USA trend ten widoczny jest z kilkutygodniowym wyprzedzeniem. Twój model powinien zauważyć tę tendencję i wygenerować powiadomienie typu: "Kukurydza na rynkach światowych drożeje. Zalecany zakup zapasu na 3 miesiące."

## 5. ML ENGINE & ANALITYKA
- Wytrenuj model `LinearRegression` (scikit-learn).
- Parametry wejściowe (Features): Rain_mm, Temp_Max, EUR_PLN.
- Cel (Target): Profit.
- Wylicz współczynniki wpływu (np. o ile PLN rośnie zysk przy wzroście EUR o 1 zł).

## 6. AI AGENT & RAPORTOWANIE (LLM)
- Użyj modelu `gpt-4o-mini`.
- Zaimplementuj moduł FinOps (wyświetlaj szacowany koszt tokenów przed strzałem).
- Stwórz raport w Markdown w folderze `reports`.
- FORMAT RAPORTU: Styl "SMS do Farmera" (same konkrety, twarde liczby).
- TREŚĆ: Całkowity zysk, wpływ pogody, wpływ kursu EUR, predykcje na 7 dni, miesiąc i kwartał.
- Uwzględnij bazę wiedzy z `knowledge/Polish-Farm-Guide.txt`.
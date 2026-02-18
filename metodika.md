# Metodologická príloha 

Táto príloha opisuje metodológiu predikčného modelu pre zmeny v poplatkoch za komunálny odpad. 
Základom metodiky je analýza trendov medzi rokmi 2023-2025, keďže v roku 2023 prebehla prvá zmena poplatkov za KO.

## Dáta a východiská 

### Dáta 
Základom sú dáta o výbere poplatku za roky 2023-2025 v nasledovnej štruktúre:
- rok
- platiteľ (FO, PO, bytové domy)
- veľkosť nádoby (od 110/120l po 3000l)
- interval zvozu

Dáta následne obsahovali ako hodnoty:
- počet miest (adries)
- počet nádob (primárne pri PO a BD je na jednej adrese viac nádob)
- výnos z poplatku (v EUR ročne)

Následne sme ako doplňujúce údaje používali prieskumy naplnenosti nádob v IBV z roku 2025.

### Východiská 
Vo všeobecnosti vychádzame z toho, že správanie obyvateľstva po zmene poplatkov či iných zmien v oblasti komunálneho odpadu bude kopírovať reakcie v rokoch 2023-2025. 

Pri **fyzických osobách** (IBV) predpokladáme, že spôsob reakcie na zvýšenie poplatkov je primárne zmena frekvencie, keďže každá domácnosť musí mať aspoň jednu nádobu a väčšinou nemá viac ako jednu nádobu. Sledovať budeme aj to, či nenastáva zmena v presune k menšej nádobe (rozdiel medzi 120 a 240l).

Pri **PO a bytových domoch**, kde je väčšinou viac nádob na jednej adrese, budeme predpokladať zmeny primárne cez počet nádob. Zmena frekvencie bude skôr druhotná možnosť v prípade, že z administratívnych alebo praktických dôvodov nebude možné zníženie počtu nádob. 

Taktiež predpokladáme, že pri výraznejšom navýšení poplatku budú behaviorálne reakcie silnejšie ako pri nižšom náraste. Od ostatného zvýšenia teda budeme musieť trendy v tomto zmysle extrapolovať. 

## Pozorované trendy 2023-2025
Zmena poplatkov bola schválená v roku 2023, s účinnosťou od roku 2024. Efekt zmien je tak možné pozorovať na vývoji v porovnaní s rokom 2023.

### IBV
Pozorujeme výraznejší pokles týždenného intervalu, primárne v prospech mesačného. Mesačný interval bol totiž zavedený v tejto zmene v roku 2023. Porovnateľné množstvo adries ako prestalo využívať týždenný interval, začalo využívať mesačný inteval. Dvojtýždenný interval ostal z veľkej časti nezmenený. 

Ide však stále o relatívne malé pohyby (pri týždennom intervale ide o pokles dokopy o 720 nádob, -11% dokopy oproti 2023, pri 120l nádobách). 
Pri dvojtýždennom intervale a pri 240l nádobách celkovo pozorujeme medziročné zmeny na úrovni 1-2%. 

Celkový počet nádob sa v princípe nezmenil (celkový pokles v IBV o 11 nádob za dva roky).

### BD a PO
Pri BD a PO pozorujeme zmeny v celkovom počte nádob, bez výrazného nárastu v ktoromkoľvek intervale. 

Počet nádob pri BD klesol o takmer 400 za dva roky (takmer -5%). Pri PO bol prepad ešte o niečo väčší (-7% a okolo 900 nádob). Pokles pozorujeme v nejakej podobe u všetkých veľkostí nádob a intervaloch.

Najvýraznejší pokles nastal u BD pri najpočetnejšej skupine (1100l nádoba) pri týždňovej frekvencii. Pri PO nastal najväčší pokles pri 120l nádobe, dokopy o takmer 700 nádob (-19%). 

## Model 

Z pozorovaní z rokov 2023-2025 extrapolujeme model do budúcna:

### IBV
Pri IBV je medziročný presun modelovaný v rámci nasledovnej "matice":

|rok r / rok r+1 |120l, 1xM|120l, 2xM|120l, 4xM|240l, 2xM|240l, 4xM|
|-|---------|---------|---------|---------|---------|
|120l, 1xM| 1 | 0 | 0 | 0 | 0
|120l, 2xM| `f` | `1 - f` | 0 | 0 | 0
|120l, 4xM| `g*a` | `g*(1-a)` | `1 - g` | 0 | 0
|240l, 2xM| 0 | `f` | 0 | `1-f` | 0
|240l, 4xM| 0 | 0 | `g*a` | `g*(1-a)` | `1 - g`

Kde v jednotlivých bunkách sú hodnoty v percentách. Tie reprezentujú presun medzi kohortami medzi dvoma rokmi. 

Parametrizovaná je matica nasledovne:

`f` je faktor presunu z 2t na mesačný interval a tvorí sa ako `(d / 15) / (b ^ y)`, kde: 
- `d` = zmena poplatku v percentách
- `b` = parameter sily reakcie, v štandardnom scenári má hodnotu `b = 2`
- `y` = počet rokov, ktoré uplynuli od zmeny 

`g` je faktor presunu z týždenného na nižšie intervaly a počíta sa ako `(d / 2) / (b ^ y)`, kde `d`, `b` aj `y` reprezentujú rovnaké hodnoty ako vyššie. Parameter `a` pre rozdelenie do mesačného resp. dvojtýždenného intervalu je nastavený ako `a = 0.45`. 

Tieto hodnoty aj vzorce boli zvolené tak, aby čo najbližšie reprezentovali pozorovaný vývoj za roky 2023-2025. Cez parameter `b` je možné meniť intenzitu reakcie. Čím nižšia hodnota, tým silnejšia zmena (presun do nižších intervalov). Pre náš scenár výraznej reakcie má hodnotu `b = 1.5`. 

Rozdiel nastáva v momente, kde bude zrušená možnosť týždenného zvozu. Vteda sa dva riadky menia:

|rok r / rok r+1 |120l, 1xM|120l, 2xM|120l, 4xM|240l, 2xM|240l, 4xM|
|-|---------|---------|---------|---------|---------|
|120l, 1xM| 1 | 0 | 0 | 0 | 0
|120l, 2xM| `f` | `1 - f` | 0 | 0 | 0
|120l, 4xM| `0.2*(1-p)` | `0.8*(1-p)` | 0 | `p` | 0
|240l, 2xM| 0 | `f` | 0 | `1-f` | 0
|240l, 4xM| 0.05 | 0.15 | 0 | 0.8 | 0

Kde musia všetci s týždenným zvozom prejsť do inej kohorty. Odlišný je iba parameter `p`, ktorý označuje percento naplnených nádob. 

### BD a PO

Pri BD a PO modelujeme počet nádob na jedno miesto/adresu vo viacerých krokoch:

1. V prvom kroku je v každom roku potrebné vypočítať faktor reakcie. Ten je určený ako: `r = 1 - (d / (s ^ y))`. Parametre `d` a `y` reprezentujú rovnaké premenné ako pri IBV. Parameter `s` je parameter sily reakcie pre BD a PO. V štandardnom scenári má hodnotu `s = 6`.

2. Následne sa týmto faktorom `r` prenásobia východiskové pomery nádob a adries pre jednotlivé kombinácie veľkostí/intervalov. Faktor teda ovplyvňuje priemerný počet nádob na jednu adresu/miesto. V prípade, že tento nový priemer klesne pod 1, je potrebné premiestniť nejaké počet adries na menší interval:

3. Pre tie kohorty (kombinácie veľkostí a intervalov), kde je tento podiel menej ako 1, sa vypočíta počet adries, ktoré treba presunúť na menej frekventovaný interval, aby sa podiel vyrovnal číslu 1. 

4. Výsledný počet nádob v kohortách sa vypočíta ako súčin nových podielov a počtu miest (pri tých predpokladáme, že ostávajú rovnaké). 

Pri BD a PO môže dôjsť v modeli k zmene minimálneho počtu ľudí na nádobu. Táto zmena je expertným odhadom, ktorý vychádza z konzultácii s OŽP a je potrebné jej závery brať s istou mierou opatrnosti. Ak však v modeli túto zmenu zrealizujeme, ovplyvní faktor `r` nasledovne:

Upravený faktor je `rx = r * (1 - (log(m - 44) / (3.5 * s)))`, kde:
- `m` je nové minimum počtu ľudí na nádobu (dnešné východisko je 45)
- `s` je rovnaký parameter ako v pôvodnom faktore `r`
- `log` je logaritmus so základom 10

Ostatné kroky (2-4) prebehnú rovnako, akurát s novým faktorom. 

### Výpočet výnosov 

Tieto dva modely vypočítajú prognózu počtu nádob v jednotlivých kohortách (interval / veľkosť nádoby). Tieto počty sú následne prenásobené priemerným poplatkom na nádobu, ktorý zohľadňuje zvýšenia v nastaveniach modelu. 
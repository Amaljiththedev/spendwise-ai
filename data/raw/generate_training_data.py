import pandas as pd
import random
import pathlib

MERCHANTS = {
    "Food & Drink": [
        "TESCO EXPRESS", "TESCO METRO {ref}",
        "SAINSBURYS LOCAL", "SAINSBURYS {ref}",
        "MCDONALDS {city}", "MCD* {ref}",
        "COSTA COFFEE", "POS COSTA COFFEE {ref}",
        "COSTA* {ref} {city}", "pos txn costa coffee {ref}",
        "STARBUCKS {city} {ref}", "STARBUCKS* {ref}",
        "GREGGS", "GREGGS {city} {ref}",
        "DELIVEROO* {ref}", "DELIVEROO {ref}",
        "JUST EAT {ref}", "JUST-EAT* {ref}",
        "NANDOS {city}", "SUBWAY {city} {ref}",
        "PRET A MANGER {ref}", "PRET* {city}",
    ],
    "Travel": [
        "UBER BV {ref}", "UBER* TRIP {city}",
        "UBR* {city} {ref}", "UBER {ref}",
        "TFL TRAVEL {ref}", "TFL* {ref}",
        "TRAINLINE {ref}", "NATIONAL RAIL {ref}",
        "EASYJET {ref}", "RYANAIR {ref}",
        "BP PETROL {ref}", "SHELL {city} {ref}",
        "PARKING {city} {ref}", "NCP {city}",
    ],
    "Shopping": [
        "AMZN MKTP {ref}", "AMAZON {ref}",
        "AMZN MKTP UK* {ref}", "payment amazo {ref}",
        "CARD PAYMENT AMAZON {ref}",
        "ASOS {ref}", "ASOS* {ref}",
        "PRIMARK {city}", "H&M {city} {ref}",
        "EBAY* {ref}", "EBAY {ref}",
        "ARGOS {ref}", "NEXT {city} {ref}",
        "BOOTS {city} {ref}", "SUPERDRUG {ref}",
    ],
    "Entertainment": [
        "NETFLIX {ref}", "NETFLIX.COM {ref}",
        "card* payment netfli {ref}",
        "SPOTIFY {ref}", "SPOTIFY AB {ref}",
        "APPLE.COM/BILL {ref}", "APPLE {ref}",
        "VUE CINEMA {city} {ref}",
        "ODEON {city} {ref}",
        "STEAM GAMES {ref}", "TWITCH {ref}",
        "DISNEY+ {ref}", "NOW TV {ref}",
    ],
    "Rent": [
        "RENT PAYMENT", "rent payment {ref}",
        "LANDLORD {ref}", "rent oh{ref} landlord",
        "fps rent to landlord", "bank transfer rent",
        "BANK TRF {ref} RENT", "STO RENT PMT",
        "sto payment/{ref}", "bank trf {ref}",
        "STANDING ORDER RENT", "SO RENT {ref}",
        "STUDENT ACCOMMODATION {ref}",
        "UNITE STUDENTS {ref}",
    ],
    "Utilities": [
        "BRITISH GAS {ref}", "BG {ref}",
        "THAMES WATER {ref}", "SEVERN TRENT {ref}",
        "BT GROUP {ref}", "BT BROADBAND {ref}",
        "EDF ENERGY {ref}", "OCTOPUS ENERGY {ref}",
        "SKY BROADBAND {ref}", "VIRGIN MEDIA {ref}",
        "VODAFONE {ref}", "O2 {ref}",
        "EE {ref}", "THREE {ref}",
    ],
    "Health & Fitness": [
        "PURE GYM {ref}", "PUREGYM {city} {ref}",
        "DAVID LLOYD {ref}", "ANYTIME FITNESS {ref}",
        "BOOTS PHARMACY {ref}", "BOOTS {ref}",
        "HOLLAND BARRETT {ref}",
        "NHS PRESCRIPTION {ref}",
        "HEADSPACE {ref}", "CALM {ref}",
        "NUFFIELD HEALTH {ref}",
    ],
}

CITIES = ["BHM", "LDN", "MAN", "LIV", "BRS", "LEE", "NEW"]
REFS   = ["58DR", "87WX", "6T4UFE", "0RJI", "2B9IQ", 
          "LPYS28", "IXA11", "OH9SDB", "G98GEY", "ZJN4"]

def generate_description(template):
    city = random.choice(CITIES)
    ref  = random.choice(REFS)
    return template.format(city=city, ref=ref)

rows = []
for category, templates in MERCHANTS.items():
    for _ in range(100):          # 100 rows per category
        template = random.choice(templates)
        desc     = generate_description(template)
        amount   = round(random.uniform(3, 800), 2)
        rows.append({
            "Date":        "2026-01-01",
            "Description": desc,
            "Amount":      amount,
            "Type":        "Expense",
            "Category":    category
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = pathlib.Path(__file__).parent / "real_training_data.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} rows → {out}")
print(df["Category"].value_counts())

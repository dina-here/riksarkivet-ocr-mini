# OCR Experiment – Arbetsprov

Detta repository innehåller ett arbetsprov där OCR-tekniker har testats på
olika typer av dokument som är relevanta för arkivmaterial.

Syftet har varit att undersöka vad som fungerar i praktiken för text­extraktion,
samt att tydligt redovisa både fungerande och icke-fungerande resultat.

---
## Dataset & struktur

```text
data/
├─ raw/
│  ├─ printed_forms/         # Modern tryckt text (böcker, artiklar, formulär)
│  ├─ handwritten_notes/     # Moderna handskrivna anteckningar (rutpapper)
│  └─ handwritten_letters/   # Historiska handskrivna brev
├─ processed/
│  ├─ printed_forms/
│  ├─ handwritten_notes/
│  └─ handwritten_letters/
└─ results/
   ├─ printed_forms/
   ├─ handwritten_notes/
   └─ handwritten_letters/

## Metoder som testats

- **Tesseract OCR**  
  Klassisk OCR-motor med stöd för språkspecifika modeller (svenska).

- **TrOCR (handwritten)**  
  Transformer-baserad OCR-modell testad som HTR-baseline för handskrivna dokument.

Samtliga script för preprocessing, OCR och experiment finns kvar i repositoryt
för transparens, även där resultaten inte blev användbara.

---

## Resultat

### ✅ printed_forms (modern tryckt text)
**Användbart resultat**

- Metod: **Tesseract OCR**
- Språk: svenska (`swe`)
- Preprocessing: mild gråskala (ingen hård binarisering)
- Layout: enspaltig text
- Page Segmentation Mode: `psm 4`

Denna kombination gav stabil och korrekt OCR-kvalitet för modern tryckt text.
Resultaten finns i `results/printed_forms/`.

---

### ❌ handwritten_notes
**Inga användbara slutresultat**

- Testad metod: TrOCR (handwritten)
- Resultat: otillräcklig kvalitet

Orsaker:
- varierande handstil
- rutpapper som stör segmentering
- avsaknad av språkspecifik styrning i modellen

---

### ❌ handwritten_letters
**Inga användbara slutresultat**

- Testad metod: TrOCR (handwritten)
- Resultat: otillräcklig kvalitet

Orsaker:
- historisk handskrift
- ornament, ligaturer och varierande bokstavsformer
- begränsningar i OCR-baserade metoder utan domänanpassad träning

Mapparna finns kvar för transparens men innehåller inga färdiga resultat.

---

## Slutsats

**Tryckt OCR fungerar med Tesseract efter korrekt preprocessing och layoutanpassning.  
Den testade HTR-baselinen (TrOCR) är inte tillräcklig för handskrivna eller historiska
dokument utan vidare träning och domänanpassning.**

För dessa dokumenttyper krävs i praktiken en dedikerad HTR-pipeline med
radsegmentering och annoterad träningsdata.

---

## Miljö & körning (kort)

Projektet kördes på Windows.

- Tesseract installerades lokalt
- `tesseract.exe` och `tessdata` angavs explicit i tesseract_ocr.py
- Python-miljö:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

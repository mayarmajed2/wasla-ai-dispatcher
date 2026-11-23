# MOKA Restaurant — Menu Extraction Report

**Date:** 2026-11-23  
**Source:** 7 actual menu images located in the workspace root (`WhatsApp Image *.jpeg`)  
**Generated Data Files:**
- `menu/menu.json` (combined canonical dataset)
- `menu/categories.json` (23 categories)
- `menu/products.json` (145 products)
- `menu/modifiers.json` (7 explicit extras/modifiers)

> **Source of Truth Policy:** No products, prices, sizes, or modifiers were invented. Dynamic stock/availability is intentionally set to `null` across all products because real-time availability must come exclusively from the authoritative Backend.

---

## 1. Extraction Summary by Source Image

| # | Source Image | Sections Extracted | Products | Modifiers |
|---|---|---|---|---|
| 1 | `WhatsApp Image 2026-11-23 at 13.55.04 PM.jpeg` | Crepes (10), Special Crepes (11) | 21 | 4 |
| 2 | `WhatsApp Image 2026-11-23 at 3.54.56 PM.jpeg` | Meals / وجبات موكا (3) | 3 | 0 |
| 3 | `WhatsApp Image 2026-11-23 at 3.55.02 PM.jpeg` | Pasta / الباستا (4) | 4 | 0 |
| 4 | `WhatsApp Image 2026-11-23 at 3.55.04 PM.jpeg` | Pizza Cheese (3), Pizza Beef (4), Pizza Chicken (7), Sandwich Pizza (7) | 21 | 3 |
| 5 | `WhatsApp Image 2026-11-23 at 3.55.202 PM.jpeg` | Beef Burger (6), Chicken Burger (9), Appetizer (8) | 23 | 0 |
| 6 | `WhatsApp Image 2026-11-23 at 3.556.05 PM.jpeg` | Hot Drinks (8), Mix Soda (4), Milk Shake (9), Unlabeled Fruit Drinks (6), Iced Coffee (6), Frappuccino (3), Red Bull Mix (3) | 39 | 0 |
| 7 | `WhatsApp Image 2026-11-323 at 3.55.05 PM.jpeg` | Pan Cake (7), Kashtota (5), Dessert (6), Tawgen (7), Waffle (9) | 34 | 0 |
| **Total** | **7 images** | **23 categories** | **145** | **7** |

---

## 2. Items & Details Requiring Your Confirmation (DO NOT ASSUME)

The following items are unclear, missing labels, or potentially contain menu printing typos. They have been marked with `"needs_confirmation": true` in the JSON files and require your confirmation before finalizing business rules:

### 2.1. Unlabeled Category in Cafe Drinks Menu
- **Source Image:** `WhatsApp Image 2026-11-23 at 3.556.05 PM.jpeg`
- **Details:** Directly below `MILK SHAKE`, there is an orange category header bar with **no text printed on it**, followed by 6 items:
  - `كيوي` (80 EGP)
  - `مانجو` (80 EGP)
  - `فراولة` (80 EGP)
  - `بلو بيري` (80 EGP)
  - `باشون فروت` (100 EGP)
  - `خوخ` (80 EGP)
- **Question:** What is the exact category name for these 6 items? (e.g., **Smoothie / سموذي**, **Fresh Juice / عصائر فريش**, or **Fruit Milkshake**?)

### 2.2. Possible Typo in Pizza Extra: `جبه كامله` (+20 EGP)
- **Source Image:** `WhatsApp Image 2026-11-23 at 3.55.04 PM.jpeg`
- **Details:** Listed under Pizza `EXTRA` as `جبه كامله` — 20 EGP.
- **Question:** Is this a printing typo for **`جبنة كاملة`** (Extra Cheese / Full Mozzarella)?

### 2.3. Possible Description Typo on `كريب كوردن بلو` (140 EGP)
- **Source Image:** `WhatsApp Image 2026-11-23 at 13.55.04 PM.jpeg`
- **Details:** Under `SPECIAL CREPES`, `كريب كوردن بلو` (140 EGP) has the exact same printed description as `كريب سموك كرانشي` (130 EGP): `(ستربس فريش + صوص باربيكيو + بيف مدخن)`.
- **Question:** What are the actual ingredients of `كريب كوردن بلو` so the AI answers customer ingredient questions accurately?

### 2.4. Sauce Choices for `صوص من اختيارك` (+20 EGP) in Crepes
- **Source Image:** `WhatsApp Image 2026-11-23 at 13.55.04 PM.jpeg`
- **Details:** Listed under Crepe `EXTRA` without enumerating the available sauces.
- **Question:** Which exact sauces can customers select for this modifier? (e.g., شيدر، رانش، باربيكيو، سويت شيلي، بيج تيستي، سموكي، توميه؟)

### 2.5. Pizza Extras Pricing Rules (`حشو اطراف` +40 EGP, `اكسترا بسطرمه / بيبروني` +50 EGP)
- **Source Image:** `WhatsApp Image 2026-11-23 at 3.55.04 PM.jpeg`
- **Question:**
  1. Do these extra prices apply identically to both **Medium (M)** and **Large (L)** pizzas?
  2. Is `اكسترا (بسطرمه، بيبروني)` **+50 EGP per topping** (e.g., +50 for Pastrami OR +50 for Pepperoni) or +50 for both together?

### 2.6. Missing Modifier / Customization Rules on Other Categories
- **Burgers (Beef & Chicken):** Are paid extras (extra patty, extra cheese, caramelized onion, sauces, combo upgrades) allowed, or only free removals (e.g., بدون بصل / بدون مخلل)?
- **Pasta (`باستا تشكن الفريدو` & `باستا وايت ستربس`):** Both descriptions explicitly include `+ بطاطس` on the image. Confirm this is intentional.
- **Cafe & Desserts:** Do drinks/desserts have required customization choices (e.g., sugar level for `قهوه تركي`, topping/sauce for `تشيز كيك`, `فته وافل`, `كب وافل`)?
- **Delivery Zones, Delivery Fees, Taxes / Service Fees:** Not listed on the menu images. Confirm whether these will be supplied exclusively by the Backend, and whether we should use placeholder test zones in the Mock Backend.

---

## 3. Notable Menu Structure Observations for AI Design

1. **Duplicate / Overlapping Short Names Across Categories:**
   - `"تروبيكال"`, `"باشون فروت"`, `"مانجو لوكو"` appear in both **Mix Soda** (70/80 EGP) and **Red Bull Mix** (110 EGP).
   - `"نوتيلا"`, `"لوتس"`, `"كيندر"`, `"بستاشيو"` appear across **Milk Shake**, **Pan Cake**, **Kashtota**, **Tawgen**, and **Waffle**.
   - `"كلاسيك"` appears in **Beef Burger**, **Chicken Burger**, and **Mix Soda**.
   - **AI Implication:** The AI Dispatcher MUST ask for clarification when a customer says an ambiguous name (e.g., *"عايز واحد نوتيلا"* or *"واحد كلاسيك"*), rather than guessing the category.
2. **Mandatory Size / Option Disambiguation:**
   - All **Pizzas** require size (`M` / `L`).
   - All **Burgers** require size (`S` Single / `D` Double).
   - All **Pan Cakes** require count (`10 قطع` / `20 قطعة`).
   - `بيتزا تشكن رانش/باربكيو` requires choosing **Ranch** or **BBQ**.
   - `ميكس صودا كلاسيك` requires choosing a flavor (`فراوله` / `صن شاين` / `كيوي` / `مانجو` / `توت` / `خوخ`).

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

## 2. Owner-Confirmed Clarifications (RESOLVED — 0 Pending Items)

All 6 previously flagged menu ambiguities were explicitly confirmed by the project owner on **2026-11-23** and applied to the canonical JSON files (`items_requiring_confirmation = 0`):

1. **Fruit Milk Shake (`cat_fruit_milk_shake`):** The 6 fruit items under Milk Shake (`كيوي 80`, `مانجو 80`, `فراولة 80`, `بلو بيري 80`, `باشون فروت 100`, `خوخ 80`) are confirmed as **Milkshake drinks** (`ميلك شيك فواكه`).
2. **Whole Wheat Pizza Crust (`mod_pizza_whole_wheat_crust`):** Confirmed as **`حبة كاملة`** (Whole Wheat flour crust, +20 EGP) — NOT a typo for cheese.
3. **`كريب كوردن بلو` (140 EGP):** Confirmed to contain **كوردن بلو** (Cordon Bleu); the printed description matching Smoke Crunchy was a menu print typo.
4. **`صوص من اختيارك` (+20 EGP):** Confirmed to allow **any sauce available in the restaurant**.
5. **Pizza Extras Pricing (`حشو اطراف` +40, `اكسترا بسطرمه/بيبروني` +50, `حبة كاملة` +20):** Confirmed to apply at the **same price** for both Medium (`M`) and Large (`L`) sizes.
6. **Other Categories (Burgers, Pasta, Cafe):** Confirmed **no additional paid extras** beyond what is printed on the menu (free removals/notes such as "من غير بصل" remain supported).

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
